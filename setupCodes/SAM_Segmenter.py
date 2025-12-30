import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
import platform
import multiprocessing
from multiprocessing import Process, Queue, Value, Lock, Manager
import queue
from multiprocessing import Value
# Assuming these are available in your environment
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import tempfile
import argparse
import threading
import queue
import gradio as gr
from multiprocessing import Queue
import os
import threading

def parse_args():
    parser = argparse.ArgumentParser(description='SAM Segmenter with Gradio Interface')
    parser.add_argument('--share', action='store_true', 
                       help='Enable Gradio live sharing (default: False)')
    return parser.parse_args()

# --- Configuration (Adjust paths as needed) ---
import os

BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Create paths that work on all platforms
SAM2_CHECKPOINT = os.path.join(BASE_MODEL_DIR, "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = os.path.join(BASE_MODEL_DIR, "sam2.1_hiera_l.yaml")
GROUNDING_DINO_CONFIG = os.path.join(BASE_MODEL_DIR, "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(BASE_MODEL_DIR, "groundingdino_swint_ogc.pth")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
MASKS_DIR = "generated_masks"

# Add new constant for negative masks directory
NEGATIVE_MASKS_DIR = "generated_negative_masks"

# --- Make input_folder truly global ---
input_folder = ""  # Initialize globally

# --- Synchronization ---
debug = True
stop_processing = Value('b', False)  # Initialize as a multiprocessing Value
worker_timings = {}

def scan_existing_masks(input_folder: str, output_masks_folder: str) -> tuple[list, list]:
    """
    Scan input and output folders to identify which files can be skipped.
    """
    input_files = [
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and
        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ]
    
    files_to_skip = []
    files_to_process = []
    
    for input_file in input_files:
        base_name = os.path.splitext(input_file)[0]
        mask_path = os.path.join(output_masks_folder, f"mask_{base_name}.png")
        
        if os.path.exists(mask_path):
            files_to_skip.append(input_file)
        else:
            # Add full path to files_to_process
            files_to_process.append(os.path.join(input_folder, input_file))
            
    return files_to_process, files_to_skip

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def open_folder_mask():
    open_folder_path = os.path.abspath("generated_masks")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def open_folder_negative_mask():
    open_folder_path = os.path.abspath("generated_negative_masks")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

# --- Model Loading (For Single Image Processing) ---
# Check if files exist
if not os.path.exists(SAM2_CHECKPOINT):
    print(f"Error: SAM2 checkpoint file not found at {SAM2_CHECKPOINT}")
    exit()
if not os.path.exists(SAM2_MODEL_CONFIG):
    print(f"Error: SAM2 model config file not found at {SAM2_MODEL_CONFIG}")
    exit()
if not os.path.exists(GROUNDING_DINO_CONFIG):
    print(f"Error: Grounding DINO config file not found at {GROUNDING_DINO_CONFIG}")
    exit()
if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
    print(f"Error: Grounding DINO checkpoint file not found at {GROUNDING_DINO_CHECKPOINT}")
    exit()


if os.name != 'nt':  # 'nt' is the os.name value for Windows
    SAM2_MODEL_CONFIG = '/' + SAM2_MODEL_CONFIG


# --- Data Classes ---

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    ymax: int
    xmax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

# --- Helper Functions ---

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    return approx_contour.reshape(-1, 2).tolist()

def polygon_to_mask(polygon: List, image_shape: tuple) -> np.ndarray:
    if not polygon:
        return np.zeros(image_shape, dtype=np.uint8)
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(1,))
    return mask

def get_boxes_from_gdino(results: tuple) -> List[List[float]]:
    boxes = []
    for box in results[0]:
        xyxy = box.tolist()
        boxes.append(xyxy)
    return boxes
    
def refine_masks(masks: np.ndarray, image_shape: tuple) -> List[np.ndarray]:
    refined_masks = []
    for mask in masks:
        # Ensure the mask is a boolean array
        mask_bool = mask > 0

        # Resize the mask to the original image shape
        mask_resized = cv2.resize(mask_bool.astype(np.uint8) * 255, (image_shape[1], image_shape[0]))

        # Threshold the resized mask to ensure binary values (0 or 255)
        _, mask_thresholded = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Convert the mask back to boolean array (True/False)
        mask_bool = mask_thresholded > 0
        
        refined_masks.append(mask_bool)

    return refined_masks

# --- Main Processing Functions ---
def detect(
    temp_image_path: str,
    labels: List[str],
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    local_grounding_model = None
) -> List[DetectionResult]:
    image_source, image = load_image(temp_image_path)
    h, w, _ = image_source.shape
    
    text_prompt = ". ".join(labels) + "."
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16): # Use updated autocast
        boxes, confidences, phrases = predict(
            model= local_grounding_model if local_grounding_model else grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    
    # Scale boxes by image dimensions
    boxes = boxes * torch.Tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    detections = []
    for box, score, label in zip(boxes, confidences, phrases):
        detections.append(
            DetectionResult(
                score=score,
                label=label,
                box=BoundingBox(
                    xmin=int(box[0]),
                    ymin=int(box[1]),
                    xmax=int(box[2]),
                    ymax=int(box[3])
                )
            )
        )
    return detections

def segment(
    temp_image_path: str,
    detection_results: List[DetectionResult],
    local_sam2_predictor = None
) -> List[DetectionResult]:
    image_source, _ = load_image(temp_image_path)
    local_sam2_predictor.set_image(image_source)

    # Get boxes in correct format
    boxes = np.array([det.box.xyxy for det in detection_results])
    
    # Use SAM2 predictor
    masks, _, _ = local_sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    
    # Handle mask dimensionality
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # Assign masks to detection results
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask > 0  # Convert to boolean mask
        
    return detection_results
    
def check_exif_orientation(image):
    """Check if image has EXIF orientation that needs correction."""
    try:
        if hasattr(image, '_getexif'):  # Check if image has EXIF data
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274 is the orientation tag
                # Return True only if orientation exists and is not 1 (normal)
                return orientation is not None and orientation != 1, orientation
    except Exception as e:
        print(f"Error checking EXIF orientation: {e}")
    return False, None

def apply_exif_orientation(image, orientation):
    """Apply the rotation and flip specified in the EXIF orientation tag."""
    try:
        # Create new image with correct dimensions first
        if orientation in [5, 6, 7, 8]:
            # Swap dimensions for 90/270 degree rotations
            new_width = image.height
            new_height = image.width
            image = image.resize((new_width, new_height))

        ORIENTATION_CONFIGS = {
            2: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),    # Mirror horizontal
            3: lambda x: x.rotate(180, expand=True),            # Rotate 180
            4: lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),    # Mirror vertical
            5: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),  # Mirror horizontal and rotate 270
            6: lambda x: x.rotate(-90, expand=True),            # Rotate 90
            7: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True), # Mirror horizontal and rotate 90
            8: lambda x: x.rotate(90, expand=True),             # Rotate 270
        }
        if orientation in ORIENTATION_CONFIGS:
            return ORIENTATION_CONFIGS[orientation](image)
    except Exception as e:
        print(f"Error applying EXIF orientation: {e}")
    return image

def save_image_without_exif(img, save_path):
    """Save image with EXIF orientation already applied (like Paint.NET)."""
    try:
        if hasattr(img, '_getexif'):  # Check if image has EXIF data
            exif = img._getexif()
            if exif is not None and 274 in exif:  # 274 is orientation tag
                orientation = exif[274]
                if orientation != 1:
                    ORIENTATION_CONFIGS = {
                        2: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
                        3: lambda x: x.rotate(180, expand=True),
                        4: lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
                        5: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
                        6: lambda x: x.rotate(-90, expand=True),
                        7: lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True),
                        8: lambda x: x.rotate(90, expand=True),
                    }
                    if orientation in ORIENTATION_CONFIGS:
                        img = ORIENTATION_CONFIGS[orientation](img)
    except Exception as e:
        print(f"Error handling EXIF: {e}")
    
    # Save without EXIF data
    img.convert("RGB").save(save_path, format='PNG')

def process_image(input_image, text_prompt, box_threshold, text_threshold, save_mask, local_sam2_predictor, local_grounding_model, output_dir, masks_dir, processed_files_queue, processed_count, lock, image_path=None, negative_prompt=None, negative_masks_dir=None):
    global input_folder
    temp_image_path = None

    try:
        timestamp = int(time.time() * 1000000)
        pid = multiprocessing.current_process().pid

        # Create temp_fix_image directory in system's temp directory
        temp_fix_dir = os.path.join(tempfile.gettempdir(), "temp_fix_image")
        os.makedirs(temp_fix_dir, exist_ok=True)

        # Create temp directory only if output_dir is not None
        temp_images_dir = os.path.join(output_dir, "temp_images") if output_dir else None

        if processed_files_queue is not None:
            # Batch mode: Use image_path
            img = Image.open(image_path)
            # Create temp file only if image has EXIF data
            if hasattr(img, '_getexif') and img._getexif() is not None:
                temp_filename = f"temp_stripped_{timestamp}_{pid}.png"
                if temp_images_dir:
                    os.makedirs(temp_images_dir, exist_ok=True)
                    temp_image_path = os.path.join(temp_images_dir, temp_filename)
                else:
                    temp_image_path = os.path.join(temp_fix_dir, temp_filename)
                save_image_without_exif(img, temp_image_path)
                input_image = Image.open(temp_image_path)
            else:
                temp_image_path = image_path
                input_image = img.convert("RGB")
        else:
            # Single image mode
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            temp_filename = f"temp_{timestamp}_{pid}.png"
            if temp_images_dir:
                os.makedirs(temp_images_dir, exist_ok=True)
                temp_image_path = os.path.join(temp_images_dir, temp_filename)
            else:
                temp_image_path = os.path.join(temp_fix_dir, temp_filename)
            save_image_without_exif(input_image, temp_image_path)

        # Detect and segment using temp_image_path
        labels = [text_prompt]
        detections = detect(temp_image_path, labels, box_threshold, text_threshold, local_grounding_model=local_grounding_model)

        if not detections:
            print(f"No detections found for image with prompt '{text_prompt}'")
            if temp_image_path != image_path and temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return np.array(input_image), None, [], temp_image_path, None

        detections = segment(temp_image_path, detections, local_sam2_predictor=local_sam2_predictor)

        # Create visualization
        segmented_image = annotate(input_image, detections)

        # Save the segmented image with a unique name only if output_dir exists
        if processed_files_queue is None and output_dir:
            output_image_filename = f"output_{timestamp}_{text_prompt.replace(' ', '_')}_{box_threshold:.2f}_{pid}.png"
            output_image_path = os.path.join(output_dir, output_image_filename)
            Image.fromarray(segmented_image).save(output_image_path)

        # Handle mask generation and coordinates
        mask_image = None
        mask_coordinates = []
        negative_mask_image = None

        if save_mask and detections:
            combined_mask = np.zeros(np.array(input_image).shape[:2], dtype=np.uint8)
            for detection in detections:
                if detection.mask is not None:
                    combined_mask |= detection.mask.astype(np.uint8)
            
            # Process negative prompt if provided
            if negative_prompt and negative_prompt.strip():
                negative_labels = [negative_prompt]
                negative_detections = detect(temp_image_path, negative_labels, box_threshold, text_threshold, local_grounding_model=local_grounding_model)
                
                if negative_detections:
                    negative_detections = segment(temp_image_path, negative_detections, local_sam2_predictor=local_sam2_predictor)
                    
                    # Create negative mask
                    negative_mask = np.zeros(np.array(input_image).shape[:2], dtype=np.uint8)
                    for detection in negative_detections:
                        if detection.mask is not None:
                            negative_mask |= detection.mask.astype(np.uint8)
                    
                    # Save negative mask separately
                    negative_mask_image = Image.fromarray(negative_mask * 255)
                    
                    # Save negative mask if directory exists
                    if processed_files_queue is None and negative_masks_dir:
                        os.makedirs(negative_masks_dir, exist_ok=True)
                        negative_mask_filename = f"negative_mask_{timestamp}_{negative_prompt.replace(' ', '_')}_{box_threshold:.2f}_{pid}.png"
                        negative_mask_image.save(os.path.join(negative_masks_dir, negative_mask_filename))
                    
                    # Debug info
                    print(f"Positive mask pixels: {np.sum(combined_mask)}")
                    print(f"Negative mask pixels: {np.sum(negative_mask)}")
                    
                    # Ensure masks are binary (0 or 1) with proper type casting
                    combined_mask_bin = (combined_mask > 0).astype(np.uint8)
                    negative_mask_bin = (negative_mask > 0).astype(np.uint8)
                    
                    # Subtract negative mask from combined mask using binary operations
                    # This ensures areas in the negative mask are removed from the positive mask
                    combined_mask = combined_mask_bin & ~negative_mask_bin
                    
                    # Alternative mathematical approach (commented out for reference)
                    # combined_mask = np.clip(combined_mask_bin - negative_mask_bin, 0, 1)
                    
                    print(f"Final mask pixels after subtraction: {np.sum(combined_mask)}")

            mask_image = Image.fromarray(combined_mask * 255)
            mask_coords = np.where(combined_mask > 0)
            mask_coordinates = list(zip(mask_coords[0].tolist(), mask_coords[1].tolist()))

            # Save mask only if masks_dir exists
            if processed_files_queue is None and masks_dir:
                os.makedirs(masks_dir, exist_ok=True)
                mask_filename = f"mask_{timestamp}_{text_prompt.replace(' ', '_')}_{box_threshold:.2f}_{pid}.png"
                mask_image.save(os.path.join(masks_dir, mask_filename))

        # Cleanup temporary file only if we created it
        if temp_image_path != image_path and temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return segmented_image, mask_image, mask_coordinates, temp_image_path, negative_mask_image

    except Exception as e:
        print(f"Error processing image: {e}")
        if temp_image_path and temp_image_path != image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return np.array(input_image), None, [], temp_image_path, None

def annotate(image: Image.Image, detection_results: List[DetectionResult]) -> np.ndarray:
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        color = np.random.randint(0, 256, size=3)

        if mask is not None:
            overlay = image_cv2.copy()
            mask_display = (mask * 255).astype(np.uint8)
            overlay[mask == 1] = color
            image_cv2 = cv2.addWeighted(image_cv2, 0.7, overlay, 0.3, 0)

            contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(
            image_cv2,
            f'{label}: {score:.2f}',
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2
        )

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def distribute_tasks(image_files, gpu_ids, batch_size):
    num_gpus = len(gpu_ids)
    num_workers_per_gpu = batch_size
    total_workers = num_gpus * num_workers_per_gpu

    # Distribute tasks evenly among all workers
    tasks_per_worker = [[] for _ in range(total_workers)]
    for i, image_file in enumerate(image_files):
        worker_index = i % total_workers
        tasks_per_worker[worker_index].append(image_file)

    # Group tasks by GPU
    tasks_per_gpu = [tasks_per_worker[i:i+num_workers_per_gpu] for i in range(0, total_workers, num_workers_per_gpu)]

    return tasks_per_gpu
def worker_process(
    gpu_id: int,
    worker_id: int,
    tasks: List[str],
    output_segmented_folder: str,
    output_masks_folder: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    results_queue: Queue,
    processed_files_queue: Queue,
    processed_count: Value,
    stop_processing: Value,
    status_queue: Queue,
    big_lock: Lock,
    negative_prompt: str = None,
    negative_masks_folder: str = None
):
    try:
        device = f"cuda:{gpu_id}"
        status_queue.put(f"INFO:Worker {worker_id} starting on GPU {gpu_id}")
        print(f"Worker {worker_id} on GPU {gpu_id} starting on device {device}")

        # Load models
        try:
            local_sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
            local_sam2_predictor = SAM2ImagePredictor(local_sam2_model)
            local_grounding_model = load_model(
                model_config_path=GROUNDING_DINO_CONFIG,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
                device=device
            )
            status_queue.put(f"INFO:Models loaded successfully on GPU {gpu_id}")
        except Exception as e:
            error_msg = f"Failed to load models on GPU {gpu_id}: {str(e)}"
            status_queue.put(f"ERROR:{error_msg}")
            results_queue.put(error_msg)
            return

        total_tasks = len(tasks)
        start_time = time.time()

        for task_idx, image_path in enumerate(tasks, 1):
            if stop_processing.value:
                status_queue.put(f"STOPPED:Worker {worker_id} on GPU {gpu_id}")
                print(f"Worker {worker_id} on GPU {gpu_id} stopping due to user request")
                return

            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                segmented_image_path = os.path.join(output_segmented_folder, f"segmented_{base_name}.png")
                mask_image_path = os.path.join(output_masks_folder, f"mask_{base_name}.png")
                negative_mask_image_path = os.path.join(negative_masks_folder, f"negative_mask_{base_name}.png") if negative_masks_folder and negative_prompt else None

                # Check if input image exists
                if not os.path.exists(image_path):
                    error_msg = f"Image not found: {image_path}"
                    status_queue.put(f"ERROR:{error_msg}")
                    results_queue.put(error_msg)
                    continue

                # Process image
                try:
                    input_image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    error_msg = f"Failed to open image {base_name}: {str(e)}"
                    status_queue.put(f"ERROR:{error_msg}")
                    results_queue.put(error_msg)
                    continue

                try:
                    segmented_image, mask_image, _, _, negative_mask_image = process_image(
                        np.array(input_image),
                        text_prompt,
                        box_threshold,
                        text_threshold,
                        save_mask=True,
                        local_sam2_predictor=local_sam2_predictor,
                        local_grounding_model=local_grounding_model,
                        output_dir=output_segmented_folder,
                        masks_dir=output_masks_folder,
                        processed_files_queue=processed_files_queue,
                        processed_count=processed_count,
                        lock=big_lock,
                        image_path=image_path,
                        negative_prompt=negative_prompt,
                        negative_masks_dir=negative_masks_folder
                    )

                    # Save outputs
                    Image.fromarray(segmented_image).save(segmented_image_path)
                    if mask_image:
                        mask_image.save(mask_image_path)
                    if negative_mask_image and negative_masks_folder:
                        negative_mask_image.save(negative_mask_image_path)

                    # Update progress
                    with big_lock:
                        processed_count.value += 1
                    
                    # Calculate progress statistics
                    elapsed_time = time.time() - start_time
                    images_per_second = task_idx / elapsed_time if elapsed_time > 0 else 0
                    remaining_tasks = total_tasks - task_idx
                    eta = remaining_tasks / images_per_second if images_per_second > 0 else 0

                    status_msg = (
                        f"PROGRESS:"
                        f"Worker {worker_id} | "
                        f"Processed: {task_idx}/{total_tasks} | "
                        f"Speed: {images_per_second:.2f} img/s | "
                        f"ETA: {eta:.1f}s | "
                        f"Current: {base_name}"
                    )
                    status_queue.put(status_msg)
                    processed_files_queue.put(segmented_image_path)
                    results_queue.put(f"Successfully processed {base_name}")

                except Exception as e:
                    error_msg = f"Failed to process {base_name}: {str(e)}"
                    status_queue.put(f"ERROR:{error_msg}")
                    results_queue.put(error_msg)
                    continue

            except Exception as e:
                error_msg = f"Unexpected error processing {image_path}: {str(e)}"
                status_queue.put(f"ERROR:{error_msg}")
                results_queue.put(error_msg)
                continue

        # Final status update
        completion_msg = f"COMPLETED:Worker {worker_id} finished processing {len(tasks)} images"
        status_queue.put(completion_msg)
        print(completion_msg)

    except Exception as e:
        error_msg = f"Critical error in worker {worker_id} on GPU {gpu_id}: {str(e)}"
        status_queue.put(f"ERROR:{error_msg}")
        results_queue.put(error_msg)
        raise

    finally:
        # Cleanup
        try:
            del local_sam2_model
            del local_sam2_predictor
            del local_grounding_model
            torch.cuda.empty_cache()
            status_queue.put(f"INFO:Worker {worker_id} cleaned up successfully")
        except Exception as e:
            status_queue.put(f"ERROR:Cleanup failed for worker {worker_id}: {str(e)}")

def batch_process_images(
    input_folder_local,
    output_segmented_folder,
    output_masks_folder,
    text_prompt,
    box_threshold,
    text_threshold,
    batch_size,
    gpu_ids,
    status_queue,
    skip_existing=False,
    negative_prompt=None
):
    global input_folder, stop_processing
    input_folder = input_folder_local
    
    # Create negative masks folder
    negative_masks_folder = os.path.join(os.path.dirname(output_masks_folder), NEGATIVE_MASKS_DIR)

    try:
        # Create output directories
        os.makedirs(output_segmented_folder, exist_ok=True)
        os.makedirs(output_masks_folder, exist_ok=True)
        if negative_prompt and negative_prompt.strip():
            os.makedirs(negative_masks_folder, exist_ok=True)
        else:
            negative_masks_folder = None

        # Scan for existing masks if skip_existing is enabled
        if skip_existing:
            image_files_to_process, skipped_files = scan_existing_masks(
                input_folder, output_masks_folder
            )
            if skipped_files:
                status_queue.put(
                    f"INFO:Skipping {len(skipped_files)} existing files: {', '.join(skipped_files)}"
                )
                print(f"Skipping {len(skipped_files)} files with existing masks")
        else:
            image_files_to_process = [
                os.path.join(input_folder, f)
                for f in os.listdir(input_folder)
                if os.path.isfile(os.path.join(input_folder, f))
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
            ]

        total_tasks = len(image_files_to_process)

        if total_tasks == 0:
            status_msg = "No new images to process" if skip_existing else "No images found in input folder"
            status_queue.put(f"ERROR:{status_msg}")
            return status_msg

        status_queue.put(f"INFO:Found {total_tasks} images to process")
        print(f"Found {total_tasks} images to process")

        # Fix: Use image_files_to_process instead of image_files
        tasks_per_gpu = distribute_tasks(image_files_to_process, gpu_ids, batch_size)

        # Create a new Manager for each batch processing run
        manager = Manager()
        processes = []
        
        try:
            # Create shared objects using the manager
            results_queue = manager.Queue()
            processed_files_queue = manager.Queue()
            processed_count = manager.Value('i', 0)
            stop_processing = manager.Value('b', False)
            big_lock = manager.Lock()

            # Start worker processes
            for gpu_index, gpu_id in enumerate(gpu_ids):
                for worker_index, tasks in enumerate(tasks_per_gpu[gpu_index]):
                    if not tasks:  # Skip if no tasks for this worker
                        continue
                    worker_id = gpu_index * batch_size + worker_index
                    p = Process(
                        target=worker_process,
                        args=(
                            gpu_id,
                            worker_id,
                            tasks,
                            output_segmented_folder,
                            output_masks_folder,
                            text_prompt,
                            box_threshold,
                            text_threshold,
                            results_queue,
                            processed_files_queue,
                            processed_count,
                            stop_processing,
                            status_queue,
                            big_lock,
                            negative_prompt,
                            negative_masks_folder
                        )
                    )
                    p.start()
                    processes.append(p)
                    
            # Monitor progress and calculate statistics
            start_time = time.time()
            while processed_count.value < total_tasks and any(p.is_alive() for p in processes):
                if stop_processing.value:
                    status_queue.put("STOPPED:Processing stopped by user")
                    break
                    
                try:
                    result = results_queue.get(timeout=1)
                    print(result)

                    # Calculate and display statistics
                    elapsed_time = time.time() - start_time
                    current_processed_count = processed_count.value
                    processing_rate = current_processed_count / elapsed_time if elapsed_time > 0 else 0
                    remaining_tasks = total_tasks - current_processed_count
                    estimated_time_remaining = remaining_tasks / processing_rate if processing_rate > 0 else 0
                    
                    status_message = (
                        f"PROGRESS:Processed: {current_processed_count}/{total_tasks} | "
                        f"Speed: {processing_rate:.2f} images/s | "
                        f"ETA: {estimated_time_remaining:.2f} s | "
                        f"Active workers: {sum(1 for p in processes if p.is_alive())}"
                    )
                    status_queue.put(status_message)
                    print(status_message)

                except queue.Empty:
                    continue

            return_message = "Processing stopped by user." if stop_processing.value else f"Completed processing {processed_count.value} out of {total_tasks} images"
            status_queue.put(f"{'STOPPED:' if stop_processing.value else 'COMPLETED:'}{return_message}")
            return return_message

        finally:
            # Clean up processes
            stop_processing.value = True
            for p in processes:
                p.terminate()
                p.join(timeout=1)
            
            # Clean up manager
            #manager.shutdown()

    except Exception as e:
        error_message = f"Batch processing failed: {str(e)}"
        status_queue.put(f"ERROR:{error_message}")
        return error_message

# Update the Gradio interface to handle the new return value
def create_gradio_interface():
    with gr.Blocks(title="SAM 2 Batch Segmentation APP Experimental") as interface:
        gr.Markdown("# SAM 2 Batch Segmentation APP Experimental")
        with gr.Tab("Single Image Processing"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="numpy", label="Input Image", height=500)
                    text_prompt = gr.Textbox(
                        label="Enter object to segment. Make sure it is lowercase and each word ends with . Also plural or singular makes difference",
                        value="eye. lips. ear.",
                        placeholder="Enter object to segment. Make sure it is lowercase and each word ends with . ",
                    )
                    negative_prompt = gr.Textbox(
                        label="Enter objects to exclude from mask (negative prompt). Same format as positive prompt.",
                        value="",
                        placeholder="e.g., nose. hair.",
                    )
                    box_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.35,
                        label="Box Threshold",
                    )
                    text_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                        label="Text Threshold",
                    )
                    run_button = gr.Button("Process Image")
                with gr.Column():
                    output_image = gr.Image(type="numpy", label="Output Image", height=500)
                    btn_open_outputs = gr.Button("Open Outputs Folder")
                    btn_open_outputs.click(fn=open_folder)
                    output_mask = gr.Image(type="numpy", label="Output Mask", height=500)
                    btn_open_outputs_mask = gr.Button("Open Generated Masks Folder")
                    btn_open_outputs_mask.click(fn=open_folder_mask)
                    output_negative_mask = gr.Image(type="numpy", label="Negative Mask", height=500, visible=False)
                    btn_open_neg_masks = gr.Button("Open Negative Masks Folder", visible=False)
                    btn_open_neg_masks.click(fn=open_folder_negative_mask)
                    mask_coordinates_output = gr.Textbox(label="Mask Coordinates", visible=False)

        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_folder_textbox = gr.Textbox(
                        label="Input Folder",
                        placeholder="Enter path to input folder",
                        interactive=True
                    )
                    output_segmented_folder = gr.Textbox(
                        label="Output Segmented Images Folder",
                        placeholder="Enter path to output segmented images folder",
                        value="",
                        interactive=True
                    )
                    output_masks_folder = gr.Textbox(
                        label="Output Masks Folder",
                        placeholder="Enter path to output masks folder",
                        value="",
                        interactive=True
                    )
                    # Add this in the first Column of Batch Processing tab
                    skip_existing = gr.Checkbox(
                        label="Skip Processing if Mask Exists",
                        value=False,
                        interactive=True
                    )
                    batch_text_prompt = gr.Textbox(
                        label="Enter object to segment",
                        value="eye. lips. ear.",
                        placeholder="Enter object to segment (lowercase, end with periods)",
                        interactive=True
                    )
                    batch_negative_prompt = gr.Textbox(
                        label="Enter objects to exclude from mask (negative prompt)",
                        value="",
                        placeholder="e.g., nose. hair.",
                        interactive=True
                    )
                    batch_box_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.35,
                        label="Box Threshold",
                        interactive=True
                    )
                    batch_text_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                        label="Text Threshold",
                        interactive=True
                    )
                    batch_size_input = gr.Number(
                        label="Batch Size",
                        value=2,
                        precision=0,
                        interactive=True
                    )
                    gpu_ids_input = gr.Textbox(
                        label="GPU IDs (comma-separated)",
                        placeholder="e.g., 0,1",
                        value="0",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    status_display = gr.HTML(
                        label="Status Display",
                        value="<div>Ready to start processing...</div>"
                    )
                    gr.HTML("<div>Progress:</div>")  # Add label as separate component
                    progress_bar = gr.Progress()
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        max_lines=10
                    )
                    error_display = gr.Textbox(
                        label="Errors",
                        interactive=False,
                        visible=False
                    )
                    with gr.Row():
                        batch_run_button = gr.Button(
                            "Start Batch Processing",
                            variant="primary"
                        )
                        stop_batch_button = gr.Button(
                            "Stop Processing",
                            variant="stop",
                            interactive=False
                        )
                    clear_button = gr.Button("Clear Status")

        # Helper functions for UI updates
        def update_ui_status(msg):
            if msg.startswith("ERROR:"):
                return {
                    error_display: gr.update(value=msg[6:], visible=True),
                    status_display: gr.update(value="<div style='color: red;'>Error occurred</div>"),
                    batch_run_button: gr.update(interactive=True),
                    stop_batch_button: gr.update(interactive=False)
                }
            elif msg.startswith("PROGRESS:"):
                return {
                    status_display: gr.update(value=f"<div>{msg[9:]}</div>"),
                    error_display: gr.update(visible=False),
                    batch_run_button: gr.update(interactive=False),
                    stop_batch_button: gr.update(interactive=True)
                }
            elif msg.startswith("STOPPED:"):
                return {
                    status_display: gr.update(value="<div style='color: orange;'>Processing stopped by user</div>"),
                    batch_status: gr.update(value="Processing stopped by user"),
                    batch_run_button: gr.update(interactive=True),
                    stop_batch_button: gr.update(interactive=False),
                    error_display: gr.update(visible=False)
                }
            elif msg.startswith("COMPLETED:"):
                return {
                    status_display: gr.update(value="<div style='color: green;'>Processing completed</div>"),
                    batch_status: gr.update(value=msg[10:]),
                    batch_run_button: gr.update(interactive=True),
                    stop_batch_button: gr.update(interactive=False),
                    error_display: gr.update(visible=False)
                }
            else:
                return {
                    status_display: gr.update(value=f"<div>{msg}</div>"),
                    error_display: gr.update(visible=False),
                    batch_run_button: gr.update(interactive=False),
                    stop_batch_button: gr.update(interactive=True)
                }


        def clear_status():
            return {
                status_display: gr.update(value="<div>Ready to start processing...</div>"),
                batch_status: gr.update(value=""),
                error_display: gr.update(value="", visible=False),
                batch_run_button: gr.update(interactive=True),
                stop_batch_button: gr.update(interactive=False)
            }

        def start_processing(input_folder_local, output_segmented_folder, output_masks_folder,
                            text_prompt, box_threshold, text_threshold, batch_size, gpu_ids_str,
                            skip_existing, negative_prompt):
            try:
                if not input_folder_local or not os.path.exists(input_folder_local):
                    return {
                        error_display: gr.update(value="Input folder does not exist", visible=True),
                        batch_run_button: gr.update(interactive=True),
                        stop_batch_button: gr.update(interactive=False),
                        status_display: gr.update(value="<div>Ready to start processing...</div>")
                    }

                # Create output directories
                os.makedirs(output_segmented_folder, exist_ok=True)
                os.makedirs(output_masks_folder, exist_ok=True)
                if negative_prompt and negative_prompt.strip():
                    negative_masks_folder = os.path.join(os.path.dirname(output_masks_folder), NEGATIVE_MASKS_DIR)
                    os.makedirs(negative_masks_folder, exist_ok=True)

                # Update UI state
                yield {
                    batch_run_button: gr.update(interactive=False),
                    stop_batch_button: gr.update(interactive=True),
                    status_display: gr.update(value="<div>Initializing processing...</div>"),
                    error_display: gr.update(visible=False)
                }

                status_queue = Queue()
                processing_thread = threading.Thread(
                    target=batch_process_images,
                    args=(
                        input_folder_local,
                        output_segmented_folder,
                        output_masks_folder,
                        text_prompt,
                        box_threshold,
                        text_threshold,
                        int(batch_size),
                        [int(x.strip()) for x in gpu_ids_str.split(',')],
                        status_queue,
                        skip_existing,
                        negative_prompt
                    )
                )
                processing_thread.start()

                while processing_thread.is_alive():
                    try:
                        status = status_queue.get(timeout=1)
                        if status.startswith("STOPPED:"):
                            yield {
                                status_display: gr.update(value="<div style='color: orange;'>Processing stopped by user</div>"),
                                batch_status: gr.update(value="Processing stopped by user"),
                                batch_run_button: gr.update(interactive=True),
                                stop_batch_button: gr.update(interactive=False),
                                error_display: gr.update(visible=False)
                            }
                            return
                        yield update_ui_status(status)
                    except queue.Empty:
                        continue

                # Wait for final status update from the queue
                try:
                    final_status = status_queue.get(timeout=5)
                    yield update_ui_status(final_status)
                except queue.Empty:
                    if stop_processing.value:
                        yield {
                            status_display: gr.update(value="<div style='color: orange;'>Processing stopped by user</div>"),
                            batch_status: gr.update(value="Processing stopped by user"),
                            batch_run_button: gr.update(interactive=True),
                            stop_batch_button: gr.update(interactive=False),
                            error_display: gr.update(visible=False)
                        }
                    else:
                        yield {
                            status_display: gr.update(value="<div style='color: green;'>Processing completed</div>"),
                            batch_status: gr.update(value="All images processed successfully"),
                            batch_run_button: gr.update(interactive=True),
                            stop_batch_button: gr.update(interactive=False),
                            error_display: gr.update(visible=False)
                        }

            except Exception as e:
                yield {
                    status_display: gr.update(value="<div style='color: red;'>Error occurred</div>"),
                    batch_status: gr.update(value=f"Error: {str(e)}"),
                    batch_run_button: gr.update(interactive=True),
                    stop_batch_button: gr.update(interactive=False),
                    error_display: gr.update(value=str(e), visible=True)
                }
            finally:
                # Ensure UI components are reset properly
                if 'processing_thread' in locals() and processing_thread.is_alive():
                    processing_thread.join(timeout=1)

        def stop_processing_handler():
            global stop_processing
            try:
                if stop_processing is not None:
                    stop_processing.value = True
                    return {
                        status_display: gr.update(value="<div style='color: orange;'>Stopping processing...</div>"),
                        batch_status: gr.update(value="Stopping processing... Please wait for current tasks to complete."),
                        stop_batch_button: gr.update(interactive=False),
                        error_display: gr.update(visible=False),
                        batch_run_button: gr.update(interactive=False)
                    }
            except Exception as e:
                return {
                    status_display: gr.update(value="<div style='color: red;'>Error while stopping: {str(e)}</div>"),
                    batch_status: gr.update(value=f"Error while stopping: {str(e)}"),
                    error_display: gr.update(visible=False),
                    batch_run_button: gr.update(interactive=True),
                    stop_batch_button: gr.update(interactive=False)
                }

        # Event handlers for single image processing
        def process_single_image(input_image, prompt, negative_prompt, box_threshold, text_threshold, save_mask):
            try:
                # Initialize models for single image processing
                local_sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
                local_sam2_predictor = SAM2ImagePredictor(local_sam2_model)
                local_grounding_model = load_model(
                    model_config_path=GROUNDING_DINO_CONFIG,
                    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
                    device=DEVICE
                )

                # Create negative masks directory if needed
                negative_masks_dir = None
                if negative_prompt and negative_prompt.strip():
                    negative_masks_dir = NEGATIVE_MASKS_DIR
                    os.makedirs(negative_masks_dir, exist_ok=True)
                    btn_open_neg_masks.visible = True
                    output_negative_mask.visible = True

                output_img, mask_img, mask_coords, _, negative_mask = process_image(
                    input_image, prompt, box_threshold, text_threshold, save_mask,
                    local_sam2_predictor, local_grounding_model, OUTPUT_DIR, MASKS_DIR,
                    None, None, None, negative_prompt=negative_prompt, negative_masks_dir=negative_masks_dir
                )
                
                # Show negative mask UI elements if negative mask exists
                neg_mask_visible = negative_mask is not None
                
                return output_img, mask_img, str(mask_coords), gr.update(visible=neg_mask_visible), negative_mask, gr.update(visible=neg_mask_visible)

            except Exception as e:
                return None, None, f"Error: {str(e)}", gr.update(visible=False), None, gr.update(visible=False)

        # Wire up all event handlers
        batch_run_button.click(
            fn=start_processing,
            inputs=[
                input_folder_textbox,
                output_segmented_folder,
                output_masks_folder,
                batch_text_prompt,
                batch_box_threshold,
                batch_text_threshold,
                batch_size_input,
                gpu_ids_input,
                skip_existing,
                batch_negative_prompt
            ],
            outputs=[
                status_display,
                batch_status,
                error_display,
                batch_run_button,
                stop_batch_button
            ]
        )

        stop_batch_button.click(
            fn=stop_processing_handler,
            outputs=[
                status_display,
                batch_status,
                error_display,
                batch_run_button,
                stop_batch_button
            ]
        )

        clear_button.click(
            fn=clear_status,
            outputs=[
                status_display,
                batch_status,
                error_display,
                batch_run_button,
                stop_batch_button
            ]
        )

        run_button.click(
            fn=process_single_image,
            inputs=[
                input_image,
                text_prompt,
                negative_prompt,
                box_threshold,
                text_threshold,
                gr.Checkbox(True, label="Save Mask")
            ],
            outputs=[
                output_image,
                output_mask,
                mask_coordinates_output,
                btn_open_neg_masks,
                output_negative_mask,
                output_negative_mask
            ]
        )

    return interface

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Try to set start method
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # If already set, get a spawn context
        ctx = multiprocessing.get_context('spawn')
        # Replace Process usage with ctx.Process throughout the code
        globals()['Process'] = ctx.Process
        globals()['Queue'] = ctx.Queue
        globals()['Value'] = ctx.Value
        globals()['Lock'] = ctx.Lock
        globals()['Manager'] = ctx.Manager
    
    demo = create_gradio_interface()
    demo.launch(share=args.share, inbrowser=True)
import os
import cv2
import shutil
import time
from PIL import Image, ExifTags, ImageOps
from ultralytics import YOLO
import imagehash
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import string
import argparse
import logging
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np
from io import StringIO, BytesIO
import torch  
import multiprocessing
from multiprocessing import Process, Queue, Value, Lock, Manager
import threading
import sys

logging.basicConfig(
    level=logging.ERROR,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.__stdout__)
    ]
)


class ProcessingState:
    def __init__(self):
        self.stop_flag = None
        self.manager = None
        self.cancel_event = threading.Event()
        self.current_executor = None

    def set_stop_flag(self, flag):
        self.stop_flag = flag
        
    def set_manager(self, manager):
        self.manager = manager
        
    def set_executor(self, executor):
        self.current_executor = executor
        
    def stop_processing(self):
        self.cancel_event.set()
        if self.stop_flag is not None:
            try:
                self.stop_flag.value = True
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        
        if self.current_executor is not None:
            try:
                self.current_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    
    def is_cancelled(self):
        if self.cancel_event.is_set():
            return True
        
        if self.stop_flag is not None:
            try:
                return self.stop_flag.value
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        
        return False
    
    def reset(self):
        self.cancel_event.clear()
        self.current_executor = None
        if self.stop_flag is not None:
            try:
                self.stop_flag.value = False
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

processing_state = ProcessingState()

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    ctx = multiprocessing.get_context('spawn')
    Process = ctx.Process
    Queue = ctx.Queue
    Value = ctx.Value
    Lock = ctx.Lock
    Manager = ctx.Manager

debug = False

import os

from .SAM_Segmenter_final import (
    build_sam2,
    SAM2ImagePredictor,
    load_model,
    process_image as sam2_process_image,
    resolve_model_paths as resolve_sam2_model_paths,
    validate_model_paths as validate_sam2_model_paths,
)


Image.MAX_IMAGE_PIXELS = None

os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

def normalize_aspect_ratios(aspect_ratios):
    """Accept '1024x1024,768x1024' OR ['1024x1024'] OR [(1024,1024)] and return list[str] like ['1024x1024']"""
    if aspect_ratios is None:
        return []
    if isinstance(aspect_ratios, str):
        return [ar.strip() for ar in aspect_ratios.split(',') if ar.strip()]
    if isinstance(aspect_ratios, (list, tuple)):
        out = []
        for ar in aspect_ratios:
            if isinstance(ar, str):
                s = ar.strip()
                if s:
                    out.append(s)
            elif isinstance(ar, (list, tuple)) and len(ar) == 2:
                out.append(f"{int(ar[0])}x{int(ar[1])}")
            else:
                s = str(ar).strip()
                if s:
                    out.append(s)
        return out
    return [str(aspect_ratios).strip()]

def parse_aspect_ratio_tuples(aspect_ratios):
    ars = normalize_aspect_ratios(aspect_ratios)
    tuples = []
    for ar in ars:
        w, h = ar.lower().split('x')
        tuples.append((int(w), int(h)))
    return tuples


img_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.heic', '.raw',
'.cr2', '.nef', '.arw', '.svg', '.ico', '.psd', '.ai', '.eps', '.pdf', '.indd',
'.jfif', '.pjpeg', '.pjp', '.hdr', '.exr', '.dib', '.svgz', '.avif', '.dng',
'.orf', '.rw2', '.sr2', '.3fr', '.srf', '.psb', '.emf', '.wmf', '.cdr',
'.cgm', '.pgm', '.ppm', '.pbm', '.pnm', '.sgi', '.ras', '.xbm', '.xpm')

YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'yolov11l-face.pt'
]

def filter_existing_files(image_files, output_folder, aspect_ratios, save_yolo, yolo_folder, save_as_png, overwrite):
    """Filter out files that already exist in output folders unless overwrite is True."""
    if overwrite:
        return image_files, []
        
    files_to_process = []
    skipped_files = []
    
    aspect_ratio_list = parse_aspect_ratio_tuples(aspect_ratios)
    
    for filename in image_files:
        base_name = os.path.splitext(filename)[0]
        extension = '.png' if save_as_png else '.jpg'
        
        existing_aspects = []
        missing_aspects = []
        
        for target_width, target_height in aspect_ratio_list:
            aspect_folder = os.path.join(output_folder, f"{target_width}x{target_height}")
            output_path = os.path.join(aspect_folder, base_name + extension)
            
            if os.path.exists(output_path):
                existing_aspects.append(f"{target_width}x{target_height}")
            else:
                missing_aspects.append(f"{target_width}x{target_height}")
        
        yolo_exists = False
        if save_yolo and yolo_folder:
            yolo_path = os.path.join(yolo_folder, base_name + extension)
            if os.path.exists(yolo_path):
                yolo_exists = True
        
        if missing_aspects:
            files_to_process.append(filename)
            if existing_aspects:
                skipped_files.append(f"{filename} (exists for: {', '.join(existing_aspects)})")
        else:
            skipped_files.append(f"{filename} (exists for all aspects{' and YOLO' if yolo_exists else ''})")
            
    return files_to_process, skipped_files


def draw_debug_bbox(image, bbox, output_path):
    """Draw bounding box on image and save for debugging"""
    if not debug:
        return
        
    os.makedirs("debug_bounding_box", exist_ok=True)
    img_debug = image.copy()
    cv2.rectangle(img_debug, 
                 (int(bbox[0]), int(bbox[1])), 
                 (int(bbox[2]), int(bbox[3])), 
                 (0, 255, 0), 2)
    
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    cv2.circle(img_debug, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
    
    cv2.imwrite(os.path.join("debug_bounding_box", output_path), cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))

def save_bounded_region(image, bbox, output_path):
    """Extract and save just the bounded region of the image"""
    if not debug:
        return
        
    os.makedirs("debug_bounding_box", exist_ok=True)
    
    x1, y1, x2, y2 = map(int, bbox)
    
    bounded_region = image[y1:y2, x1:x2]
    
    cv2.imwrite(os.path.join("debug_bounding_box", output_path), cv2.cvtColor(bounded_region, cv2.COLOR_RGB2BGR))

def resize_bbox_to_dimensions(bbox, target_width, target_height, img_width, img_height, is_sam=False, padding_value=0, padding_unit="percent"):
    """
    Resize bbox to target dimensions while maintaining subject focus and minimal expansion
    """
    x1, y1, x2, y2 = bbox
    subject_width = x2 - x1
    subject_height = y2 - y1
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    target_aspect = target_width / target_height
    current_aspect = subject_width / subject_height
    
    if debug:
        print(f"\nDebug resize_bbox_to_dimensions:")
        print(f"Original bbox: {[x1, y1, x2, y2]}")
        print(f"Target aspect: {target_aspect:.3f}")
        print(f"Current aspect: {current_aspect:.3f}")
    
    if current_aspect > target_aspect:
        new_width = subject_width
        new_height = new_width / target_aspect
    else:
        new_height = subject_height
        new_width = new_height * target_aspect
    
    padding_x = (new_width - subject_width) / 2
    padding_y = (new_height - subject_height) / 2
    
    if padding_value > 0:
        if padding_unit == "percent":
            padding_x_val = new_width * (padding_value / 100)
            padding_y_val = new_height * (padding_value / 100)
        else: 
            padding_x_val = padding_value
            padding_y_val = padding_value
            
        new_width += 2 * padding_x_val
        new_height += 2 * padding_y_val
        
        if debug:
            print(f"Applied {padding_value}{padding_unit} padding: width+{2*padding_x_val:.1f}, height+{2*padding_y_val:.1f}")
    
    new_x1 = center_x - (new_width / 2)
    new_y1 = center_y - (new_height / 2)
    new_x2 = center_x + (new_width / 2)
    new_y2 = center_y + (new_height / 2)
    
    if new_x1 < 0:
        new_x2 += abs(new_x1)
        new_x1 = 0
    if new_x2 > img_width:
        new_x1 -= (new_x2 - img_width)
        new_x2 = img_width
    if new_y1 < 0:
        new_y2 += abs(new_y1)
        new_y1 = 0
    if new_y2 > img_height:
        new_y1 -= (new_y2 - img_height)
        new_y2 = img_height
        
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)
    
    if debug:
        print(f"Final bbox: {[int(new_x1), int(new_y1), int(new_x2), int(new_y2)]}")
        
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def crop_images(input_folder, output_folder, aspect_ratios, yolo_folder, save_yolo, batch_size, gpu_ids, overwrite, selected_class, save_as_png, sam2_prompt, debug_mode=False, skip_no_detection=False, padding_value=0, padding_unit="percent", model_dir="models"):
    try:
        global debug
        debug = debug_mode
        
        # Create a new Manager for shared resources
        manager = Manager()
        processing_state.set_manager(manager)
        stop_processing = manager.Value('b', False)
        processing_state.set_stop_flag(stop_processing)
        
        results_queue = manager.Queue()
        processed_files_queue = manager.Queue()
        processed_count = manager.Value('i', 0)
        status_queue = manager.Queue()
        big_lock = manager.Lock()

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(img_formats)]
        
        # Filter out existing files and get skip information
        filtered_files, skipped_files = filter_existing_files(
            image_files, 
            output_folder, 
            aspect_ratios, 
            save_yolo, 
            yolo_folder, 
            save_as_png, 
            overwrite
        )
        
        if skipped_files:
            yield f"Skipping {len(skipped_files)} existing files:"
            for skip_info in skipped_files:
                yield f"• {skip_info}"
        
        if not filtered_files:
            yield "No new files to process. All files already exist in output folders."
            return
            
        yield f"Processing {len(filtered_files)} new files..."
        
        gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        
        tasks_per_gpu = distribute_tasks(filtered_files, gpu_ids, batch_size)

        processes = []
        
        try:
            for gpu_index, gpu_id in enumerate(gpu_ids):
                for worker_index, tasks in enumerate(tasks_per_gpu[gpu_index]):
                    if not tasks:
                        continue
                    worker_id = gpu_index * batch_size + worker_index
                    p = Process(
                        target=worker_process,
                        args=(
                            gpu_id, worker_id, tasks, input_folder, output_folder,
                            yolo_folder, aspect_ratios, save_yolo, overwrite,
                            selected_class, save_as_png, sam2_prompt, results_queue,
                            processed_files_queue, processed_count, stop_processing,
                            status_queue, big_lock, debug_mode, skip_no_detection,
                            padding_value, padding_unit, model_dir
                        )
                    )
                    p.start()
                    processes.append(p)

            total_files = len(filtered_files)
            while processed_count.value < total_files and any(p.is_alive() for p in processes):
                if stop_processing.value:
                    # Wait for processes to finish current tasks (5sec)
                    for p in processes:
                        p.join(timeout=5) 
                    yield "Processing stopped by user"
                    break
                
                try:
                    status = status_queue.get(timeout=1)
                    yield status
                except queue.Empty:
                    continue

            return_message = "Processing stopped by user." if stop_processing.value else f"Completed processing {processed_count.value} out of {total_files} images"
            yield return_message

        finally:
            stop_processing.value = True  
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=1)
            
            manager.shutdown()

    except Exception as e:
        yield f"Error: {str(e)}"

def add_padding_to_bbox(bbox, img_width, img_height, padding_percent=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    padding_x = width * padding_percent
    padding_y = height * padding_percent
    
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(img_width, x2 + padding_x)
    y2 = min(img_height, y2 + padding_y)
    
    return [int(x1), int(y1), int(x2), int(y2)]

def try_reach_aspect_ratio(bbox, target_aspect, original_width, original_height, min_match_percentage=50):
    """
    Iteratively try to reach target aspect ratio, falling back to lower percentages if needed.
    Returns (success, new_bbox, achieved_percentage)
    """
    def calculate_current_aspect(bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width / height if height != 0 else float('inf')
    
    def aspect_ratio_diff_percentage(current_aspect, target_aspect):
        return abs(1 - (current_aspect / target_aspect)) * 100

    def expand_bbox_by_one(bbox, direction):
        new_bbox = list(bbox)
        if direction == 'width':
            if new_bbox[0] > 0:
                new_bbox[0] -= 1
            if new_bbox[2] < original_width:
                new_bbox[2] += 1
        else:  
            if new_bbox[1] > 0:
                new_bbox[1] -= 1
            if new_bbox[3] < original_height:
                new_bbox[3] += 1
        return new_bbox

    best_bbox = list(bbox)
    best_percentage = 0
    current_target_percentage = 100

    while current_target_percentage >= min_match_percentage:
        if debug:
            print(f"Trying to reach {current_target_percentage}% of target aspect ratio {target_aspect}")
        
        current_bbox = list(bbox)
        max_iterations = max(original_width, original_height) * 2
        iteration = 0
        
        while iteration < max_iterations:
            current_aspect = calculate_current_aspect(current_bbox)
            diff_percentage = aspect_ratio_diff_percentage(current_aspect, target_aspect)
            match_percentage = 100 - diff_percentage
            
            if match_percentage > best_percentage:
                best_bbox = list(current_bbox)
                best_percentage = match_percentage

            if match_percentage >= current_target_percentage:
                return True, current_bbox, match_percentage

            if current_aspect < target_aspect:
                
                current_bbox = expand_bbox_by_one(current_bbox, 'width')
            else:
                current_bbox = expand_bbox_by_one(current_bbox, 'height')
            
            iteration += 1
            
            width = current_bbox[2] - current_bbox[0]
            height = current_bbox[3] - current_bbox[1]
            if (width >= original_width and current_aspect < target_aspect) or \
               (height >= original_height and current_aspect > target_aspect):
                break

        
        current_target_percentage -= 1
        
    return False, best_bbox, best_percentage

def process_image(args):
    filename, input_folder, output_folder, aspect_ratios, yolo_folder, save_yolo, overwrite, selected_class, save_as_png, sam2_prompt, model, sam2_predictor, grounding_model, debug_mode, skip_no_detection, padding_value, padding_unit = args
    
    debug = debug_mode
    def parse_aspect_ratios(aspect_ratios):
        """Helper function to parse aspect ratios from different input formats"""
        try:
            if isinstance(aspect_ratios, list) and all(isinstance(x, tuple) for x in aspect_ratios):
                return aspect_ratios
            if isinstance(aspect_ratios, str):
                return [tuple(map(int, ar.strip().split('x'))) for ar in aspect_ratios.split(',')]
            if isinstance(aspect_ratios, list) and all(isinstance(x, str) for x in aspect_ratios):
                return [tuple(map(int, ar.strip().split('x'))) for ar in aspect_ratios]
            raise ValueError(f"Invalid aspect_ratios format: {aspect_ratios}")
        except Exception as e:
            raise ValueError(f"Failed to parse aspect ratios: {str(e)}")

    try:
        img_path = os.path.join(input_folder, filename)
        
        try:
            if debug:
                logging.critical(f"\n=== Processing {filename} ===")
            
            # Read image via PIL and apply EXIF orientation (many phone images rely on this)
            with Image.open(img_path) as _im_raw:
                _im = ImageOps.exif_transpose(_im_raw).convert('RGB')
                img_array = np.array(_im)

            if img_array is None or img_array.size == 0:
                raise ValueError(f"Could not read image {img_path}")
            if not save_as_png:
                if debug:
                    logging.critical(f"Converting {filename} to JPG format before processing")
                
                with Image.open(img_path) as original_pil_raw:
                    original_pil = ImageOps.exif_transpose(original_pil_raw).copy()
                    if original_pil.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', original_pil.size, (255, 255, 255))
                        if original_pil.mode == 'P':
                            original_pil = original_pil.convert('RGBA')
                        # Paste image on white background using alpha channel as mask
                        background.paste(original_pil, mask=original_pil.split()[-1] if original_pil.mode in ('RGBA', 'LA') else None)
                        pil_img = background
                    else:
                        pil_img = original_pil.convert('RGB')
                    
                    jpg_buffer = BytesIO()
                    pil_img.save(jpg_buffer, format='JPEG', quality=100)
                    jpg_buffer.seek(0)
                    
                    # Reload the temp JPG
                    pil_img_jpg = Image.open(jpg_buffer).convert('RGB')
                    img_array = np.array(pil_img_jpg)
                
                if debug:
                    logging.critical(f"JPG conversion completed for {filename}")
            
            original_height, original_width = img_array.shape[:2]
            if debug:
                logging.critical(f"1. Original image size: {original_width}x{original_height}")

            # SAM2 Processing starts here
            if sam2_prompt:
                if debug:
                    logging.critical("\n=== SAM2 Processing ===")
                    logging.critical(f"6. Input array shape to SAM2: {img_array.shape}")
                
                segmented_image, mask_image, mask_coordinates, _, _ = sam2_process_image(
                    img_array,
                    sam2_prompt,
                    0.35,
                    0.25,
                    True,
                    sam2_predictor,
                    grounding_model,
                    None,
                    yolo_folder,
                    None,
                    None,
                    None,
                    img_path
                )
                
                if debug:
                    logging.critical(f"7. Output segmented image shape: {segmented_image.shape}")
                
                segmented_pil = Image.fromarray(segmented_image)
                
                if mask_coordinates and len(mask_coordinates) > 0:
                    if debug:
                        logging.critical(f"9. Number of coordinates: {len(mask_coordinates)}")
                        logging.critical(f"9.1. First few coordinates: {mask_coordinates[:5] if len(mask_coordinates) >= 5 else mask_coordinates}")
                    
                    try:
                        x_coords = [p[1] for p in mask_coordinates if len(p) >= 2]
                        y_coords = [p[0] for p in mask_coordinates if len(p) >= 2]
                    except (IndexError, TypeError) as e:
                        if debug:
                            logging.critical(f"Error extracting coordinates: {e}")
                            logging.critical(f"Mask coordinates format: {type(mask_coordinates)}, sample: {mask_coordinates[:3] if mask_coordinates else 'Empty'}")
                        return f"Invalid coordinate format for {filename}", 0, 1
                    
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        if debug:
                            logging.critical("Empty coordinate arrays")
                        return f"No valid coordinates found for {filename}", 0, 1
                    
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    bbox = [int(x) for x in bbox]
                    
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        if debug:
                            logging.critical(f"Invalid bbox dimensions: {bbox}")
                        return f"Invalid bbox for {filename}", 0, 1
                    
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = min(original_width, bbox[2])
                    bbox[3] = min(original_height, bbox[3])
                    
                    if debug:
                        logging.critical(f"10. Initial bbox: {bbox}")
                        draw_debug_bbox(img_array, bbox, f"viz_initial_bbox_{filename}")
                        save_bounded_region(img_array, bbox, f"initial_bbox_{filename}")
                else:
                    if debug:
                        logging.critical("No mask coordinates found")
                    if skip_no_detection:
                        return f"Skipped {filename} (no SAM2 detection)", 0, 0

                    # Fallback (keep image count stable):
                    # If SAM2 fails to return a mask and the caller did NOT ask to skip,
                    # use the full image as the crop bbox so downstream steps still get an output.
                    bbox = [0, 0, original_width, original_height]
                    segmented_pil = Image.fromarray(img_array)
                    print(f"FALLBACK: no SAM2 mask for {filename}; using full-image bbox {bbox}")
                    logging.critical(f"FALLBACK: no SAM2 mask for {filename}; using full-image bbox {bbox}")

            else:
                if debug:
                    logging.critical("\n=== YOLO Processing ===")
                results = model(img_array)
                detections = results[0].boxes
                
                if detections is not None and len(detections) > 0:
                    detected_classes = [model.names[int(box.cls)] for box in detections]
                    print(f"DETECTED CLASSES: {detected_classes}")
                    print(f"AVAILABLE MODEL CLASSES: {list(model.names.values())}")
                    print(f"LOOKING FOR CLASS: '{selected_class}'")
                
                class_detections = [box for box in detections if model.names[int(box.cls)] == selected_class]
                
                if not class_detections and detections is not None and len(detections) > 0:
                    face_classes = [box for box in detections if 'face' in model.names[int(box.cls)].lower()]
                    if face_classes:
                        class_detections = face_classes
                        print(f"FOUND FACE CLASS: {[model.names[int(box.cls)] for box in face_classes]}")
                    elif 'face' in selected_class.lower():
                        # For face models, all detections are faces, so use the first one
                        class_detections = [detections[0]]
                        print(f"USING FIRST DETECTION FROM FACE MODEL")
                
                if class_detections:
                    bbox = class_detections[0].xyxy[0].tolist()
                    bbox = [int(x) for x in bbox]
                    print(f"YOLO DETECTION FOUND: {bbox} for class '{selected_class}'")  # Always show
                    logging.critical(f"YOLO DETECTION FOUND: {bbox} for class '{selected_class}'")  # Always log
                    if debug:
                        logging.critical(f"12. Original YOLO bbox: {bbox}")
                        draw_debug_bbox(img_array, bbox, f"viz_yolo_bbox_{filename}")
                else:
                    print(f"NO '{selected_class.upper()}' CLASS FOUND IN DETECTIONS for {filename}")
                    logging.critical(f"NO '{selected_class.upper()}' CLASS FOUND IN DETECTIONS for {filename}")

                    if skip_no_detection:
                        if debug:
                            logging.critical(f"No {selected_class} detected")
                        return f"No {selected_class} detected in {filename}", 0, 1

                    # Retry once with a lower confidence threshold to reduce false negatives
                    # (only triggered on failures, so the common-case runtime is unaffected).
                    try:
                        results_retry = model(img_array, conf=0.10)
                        detections_retry = results_retry[0].boxes
                        class_detections_retry = [box for box in detections_retry if model.names[int(box.cls)] == selected_class] if detections_retry is not None else []
                        if class_detections_retry:
                            results = results_retry
                            detections = detections_retry
                            bbox = [int(x) for x in class_detections_retry[0].xyxy[0].tolist()]
                            print(f"YOLO RETRY DETECTION FOUND: {bbox} for class '{selected_class}' in {filename}")
                            logging.critical(f"YOLO RETRY DETECTION FOUND: {bbox} for class '{selected_class}' in {filename}")
                        else:
                            raise ValueError("retry did not find target class")
                    except Exception:
                        # Final fallback: keep the image in the dataset by using the full-image bbox.
                        bbox = [0, 0, original_width, original_height]
                        print(f"FALLBACK: no '{selected_class}' detection for {filename}; using full-image bbox {bbox}")
                        logging.critical(f"FALLBACK: no '{selected_class}' detection for {filename}; using full-image bbox {bbox}")

            processed = 0
            skipped = 0
            
            if debug:
                logging.critical("\n=== Processing Crops ===")
            img = Image.fromarray(img_array)
            
            try:
                parsed_aspect_ratios = parse_aspect_ratios(aspect_ratios)
            except ValueError as e:
                logging.critical(f"Error parsing aspect ratios: {str(e)}")
                return f"Error processing {filename}: Invalid aspect ratios", 0, 1

            for target_width, target_height in parsed_aspect_ratios:
                print(f"PROCESSING ASPECT RATIO: {target_width}x{target_height}")  # Always show this
                logging.critical(f"PROCESSING ASPECT RATIO: {target_width}x{target_height}")  # Always log this
                if debug:
                    logging.critical(f"\n--- Processing aspect ratio {target_width}x{target_height}: ---")
                
                if target_width <= 0 or target_height <= 0:
                    print(f"INVALID ASPECT RATIO: {target_width}x{target_height}")
                    logging.critical(f"Invalid aspect ratio values: {target_width}x{target_height}")
                    continue

                # <<< FIX: Replaced the flawed iterative expansion with a direct calculation
                # that creates the tightest possible crop for the target aspect ratio.
                final_bbox = resize_bbox_to_dimensions(
                    bbox, target_width, target_height, original_width, original_height,
                    False, padding_value, padding_unit
                )
                
                # Validate final_bbox - if invalid, fall back to original bbox
                if not final_bbox or len(final_bbox) != 4:
                    logging.critical(f"Invalid final_bbox: {final_bbox}")
                    logging.critical(f"Original bbox: {bbox}")
                    logging.critical(f"Falling back to original bbox")
                    final_bbox = bbox.copy() if isinstance(bbox, list) else list(bbox)
                
                # Ensure final_bbox is within bounds and valid
                final_bbox[0] = max(0, int(final_bbox[0]))
                final_bbox[1] = max(0, int(final_bbox[1]))
                final_bbox[2] = min(original_width, int(final_bbox[2]))
                final_bbox[3] = min(original_height, int(final_bbox[3]))
                
                # Check if final bbox has valid dimensions
                if final_bbox[2] <= final_bbox[0] or final_bbox[3] <= final_bbox[1]:
                    logging.critical(f"Invalid final bbox dimensions: {final_bbox}")
                    logging.critical(f"Original bbox: {bbox}, Image size: {original_width}x{original_height}")
                    logging.critical(f"Using original detection bbox instead")
                    final_bbox = list(bbox)
                    # Re-validate the original bbox
                    if final_bbox[2] <= final_bbox[0] or final_bbox[3] <= final_bbox[1]:
                        logging.critical(f"Original bbox also invalid, skipping")
                        continue
                
                if debug:
                    logging.critical(f"15. Final bbox: {final_bbox}")
                    draw_debug_bbox(img_array, final_bbox, f"viz_final_bbox_{target_width}x{target_height}_{filename}")
                    save_bounded_region(img_array, final_bbox, f"final_bbox_{target_width}x{target_height}_{filename}")
                
                try:
                    cropped_img = img.crop(final_bbox)
                    if debug:
                        logging.critical(f"16. Cropped image size: {cropped_img.size}")
                except Exception as e:
                    logging.critical(f"Error cropping image: {str(e)}")
                    logging.critical(f"Bbox used for cropping: {final_bbox}")
                    continue

                aspect_folder = os.path.join(output_folder, f"{target_width}x{target_height}")
                os.makedirs(aspect_folder, exist_ok=True)
                save_path = os.path.join(aspect_folder, os.path.splitext(filename)[0] + ('.png' if save_as_png else '.jpg'))

                try:
                    if save_as_png:
                        cropped_img.save(save_path, 'PNG')
                    else:
                        cropped_img.save(save_path, 'JPEG', quality=100)
                    processed += 1
                    print(f"SUCCESSFULLY SAVED: {save_path}")  # Always show this
                    logging.critical(f"SUCCESSFULLY SAVED: {save_path}")  # Always log this
                    if debug:
                        logging.critical(f"17. Saved crop to: {save_path}")
                except Exception as e:
                    print(f"ERROR SAVING: {str(e)} | Path: {save_path}")  # Always show errors
                    logging.critical(f"Error saving cropped image: {str(e)}")
                    logging.critical(f"Save path: {save_path}")
                    logging.critical(f"Cropped image size: {cropped_img.size if 'cropped_img' in locals() else 'N/A'}")
                    skipped += 1

            if save_yolo and yolo_folder:
                if debug:
                    logging.critical("\n=== Saving YOLO/masked image ===")
                yolo_save_path = os.path.join(yolo_folder, os.path.splitext(filename)[0] + ('.png' if save_as_png else '.jpg'))
                
                try:
                    if sam2_prompt:
                        segmented_pil.save(yolo_save_path, format='PNG' if save_as_png else 'JPEG', quality=100)
                    else:
                        result_img = Image.fromarray(results[0].plot())
                        result_img.save(yolo_save_path, format='PNG' if save_as_png else 'JPEG', quality=100)
                    processed += 1
                except Exception as e:
                    logging.critical(f"Error saving YOLO/SAM2 output: {str(e)}")
                    skipped += 1

            if debug:
                logging.critical(f"\n=== Processing Complete ===")
                logging.critical(f"Processed: {processed}, Skipped: {skipped}")
            
            return f"Processed {filename}", processed, skipped

        finally:
            try:
                pass
            except Exception as e:
                logging.critical(f"Warning: Could not clean up temporary directory: {str(e)}")

    except Exception as e:
        if debug:
            logging.critical(f"\n=== Error Occurred ===")
            logging.critical(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        return f"Error processing {filename}: {str(e)}", 0, 1
    
def resize_image(args):
    input_path, resolutions, output_root, save_as_png, face_model_path, overwrite, no_crop = args
    try:
        image = Image.open(input_path)
        processed = 0
        skipped = 0
        face_detected = False
        face_locations = None

        for resolution in resolutions:
            # Use output_root directly without subfolders
            output_folder = output_root
            os.makedirs(output_folder, exist_ok=True)

            output_filename = os.path.basename(input_path)
            if save_as_png:
                output_filename = os.path.splitext(output_filename)[0] + '.png'
            else:
                output_filename = os.path.splitext(output_filename)[0] + '.jpg'
            output_path = os.path.join(output_folder, output_filename)

            if not overwrite and os.path.exists(output_path):
                logging.info(f"Skipping {output_filename} for {resolution[0]}x{resolution[1]} as it already exists.")
                skipped += 1
                continue

            if no_crop:
                new_image = Image.new('RGB', resolution, (255, 255, 255))
                aspect_ratio = min(resolution[0] / image.width, resolution[1] / image.height)
                new_size = (int(image.width * aspect_ratio), int(image.height * aspect_ratio))
                resized_image = image.resize(new_size, Image.LANCZOS)
                position = ((resolution[0] - new_size[0]) // 2, (resolution[1] - new_size[1]) // 2)
                new_image.paste(resized_image, position)
                resized_image = new_image
            else:
                if not face_detected:
                    face_model = YOLO(face_model_path)
                    results = face_model(input_path)
                    detections = results[0].boxes

                    if len(detections) > 0:
                        face_detected = True
                        face = detections[0]
                        x1, y1, x2, y2 = face.xyxy[0].tolist()
                        face_locations = [(int(y1), int(x2), int(y2), int(x1))]

                desired_aspect_ratio = resolution[0] / resolution[1]
                image_aspect_ratio = image.width / image.height

                if image_aspect_ratio > desired_aspect_ratio:
                    new_width = int(image.height * desired_aspect_ratio)
                    new_height = image.height
                else:
                    new_width = image.width
                    new_height = int(image.width / desired_aspect_ratio)

                left = (image.width - new_width) / 2
                top = (image.height - new_height) / 2
                right = (image.width + new_width) / 2
                bottom = (image.height + new_height) / 2

                if face_detected and face_locations:
                    face_top, face_right, face_bottom, face_left = face_locations[0]
                    face_center_x = (face_left + face_right) // 2
                    face_center_y = (face_top + face_bottom) // 2

                    left = min(max(0, face_center_x - new_width // 2), image.width - new_width)
                    top = min(max(0, face_center_y - new_height // 2), image.height - new_height)
                    right = left + new_width
                    bottom = top + new_height

                resized_image = image.crop((left, top, right, bottom))
                resized_image = resized_image.resize(resolution, Image.LANCZOS)

            if save_as_png:
                resized_image.save(output_path, format='PNG')
            else:
                resized_image.save(output_path, format='JPEG', quality=100)

            processed += 1

        return processed, skipped
    except Exception as e:
        logging.critical(f"Error processing {input_path}: {e}")
        return 0, 0

def resize_images(Model_Dir, input_folder, output_folder, resolutions, save_as_png, num_threads=1, overwrite=False, no_crop=False):
    resolutions = [tuple(map(int, res.strip().split('x'))) for res in resolutions.split(',')]
    face_model_path = os.path.join(Model_Dir, "face_yolov9c.pt")

    image_paths = []
    for resolution in resolutions:
        resolution_folder = os.path.join(input_folder, f"{resolution[0]}x{resolution[1]}")
        if os.path.exists(resolution_folder):
            resolution_images = [
                os.path.join(resolution_folder, fname)
                for fname in os.listdir(resolution_folder)
                if fname.lower().endswith(img_formats) and os.path.isfile(os.path.join(resolution_folder, fname))
            ]
            image_paths.extend(resolution_images)
            logging.info(f"Found {len(resolution_images)} images in {resolution_folder}")
        else:
            error_msg = f"Folder not found for resolution {resolution[0]}x{resolution[1]}. Skipping this resolution."
            logging.critical(error_msg)
            yield error_msg

    total_images = len(image_paths)
    logging.info(f"Total images to process: {total_images}")

    if total_images == 0:
        yield "No images found to process in the specified resolution folders."
        return

    num_threads = min(num_threads, total_images)
    logging.info(f"Using {num_threads} threads for processing")

    processed_count = 0
    skipped_count = 0
    lock = threading.Lock()
    start_time = time.time()

    def update_count(processed, skipped):
        nonlocal processed_count, skipped_count
        with lock:
            processed_count += processed
            skipped_count += skipped
            return processed_count + skipped_count

    # Reset cancellation state and prepare for new processing
    processing_state.reset()
    progress_bar = tqdm(total=total_images, unit='image', desc="Resizing images", file=sys.stdout)

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Set the current executor for cancellation
        processing_state.set_executor(executor)
        
        future_to_image = {executor.submit(resize_image, (path, resolutions, output_folder, save_as_png, face_model_path, overwrite, no_crop)): path
                           for path in image_paths}

        for future in as_completed(future_to_image):
            # Check for cancellation using the new robust method
            if processing_state.is_cancelled():
                yield "Resizing cancelled by user."
                # Cancel remaining futures
                for f in future_to_image:
                    f.cancel()
                break
            try:
                processed, skipped = future.result()
                progress_bar.update(processed + skipped)
                count = update_count(processed, skipped)
                elapsed_time = time.time() - start_time
                images_per_second = count / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (total_images - count) / images_per_second if images_per_second > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                progress_msg = f"Processed: {processed_count}, Skipped: {skipped_count}, Total: {count}/{total_images} images. ETA: {eta_str}"
                print(progress_msg)
                yield progress_msg

                if count >= total_images:
                    break
            except Exception as exc:
                logging.critical(f"Generated an exception: {exc}")

    progress_bar.close()
    total_time = time.time() - start_time
    yield f"Processing complete! Processed: {processed_count}, Skipped: {skipped_count}, Total: {processed_count + skipped_count}/{total_images} images in {time.strftime('%H:%M:%S', time.gmtime(total_time))}"

def move_low_res_files(folder_a, folder_b, folder_c, folder_d, min_width, min_height):
    os.makedirs(folder_c, exist_ok=True)
    os.makedirs(folder_d, exist_ok=True)

    image_files = [f for f in os.listdir(folder_a) if f.endswith(img_formats)]
    total_files = len(image_files)
    files_processed = 0
    files_moved = 0

    for filename in image_files:
        image_path_a = os.path.join(folder_a, filename)
        image_path_b = os.path.join(folder_b, filename)

        if os.path.exists(image_path_b):
            with Image.open(image_path_a) as img:
                width, height = img.size

            if width < min_width or height < min_height:
                shutil.move(image_path_b, folder_c)
                shutil.move(image_path_a, folder_d)
                files_moved += 1

        files_processed += 1
        yield f"Processed: {files_processed}/{total_files}, Moved: {files_moved}"

    yield f"Processing complete! Processed: {files_processed}/{total_files}, Moved: {files_moved}"

def rename_files(folder_path, start_number):
    def random_string(length=10):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

    def assign_random_names(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                random_name = f"{random_string()}.{filename.split('.')[-1]}"
                random_file_path = os.path.join(path, random_name)
                os.rename(file_path, random_file_path)
                yield f"Assigned random name: {random_name}"

    def rename_files_sequentially(path, start):
        i = start
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                new_filename = f"img{i}.{filename.split('.')[-1]}"
                new_file_path = os.path.join(path, new_filename)
                os.rename(file_path, new_file_path)
                i += 1
                yield f"Renamed to: {new_filename}"

    yield from assign_random_names(folder_path)
    yield from rename_files_sequentially(folder_path, start_number)
    yield "Renaming complete!"

def process_image_face(args):
    input_path, output_root, padding, save_as_png, face_model = args
    try:
        with Image.open(input_path) as img:
            png_path = os.path.splitext(input_path)[0] + '_temp.png'
            img.save(png_path, 'PNG')

        image = cv2.imread(png_path)

        if image is None:
            raise ValueError(f"Could not read image {input_path} even after conversion to PNG")

        results = face_model(png_path)
        detections = results[0].boxes

        if len(detections) == 0:
            os.remove(png_path)
            return f"No face detected in {input_path}", 0, 1

        face = detections[0]
        x1, y1, x2, y2 = map(int, face.xyxy[0].tolist())

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        face_image = image[y1:y2, x1:x2]

        os.makedirs(output_root, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + ('.png' if save_as_png else '.jpg')
        output_path = os.path.join(output_root, output_filename)

        if save_as_png:
            cv2.imwrite(output_path, face_image)
        else:
            cv2.imwrite(output_path, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        os.remove(png_path)
        return f"Processed: {input_path}", 1, 0
    except Exception as e:
        if 'png_path' in locals():
            os.remove(png_path)
        return f"Error processing {input_path}: {e}", 0, 0

def extract_faces(Model_Dir, input_folder, output_folder, padding, save_as_png, num_threads):
    
    face_model_path = os.path.join(Model_Dir, "face_yolov9c.pt")
    face_model = YOLO(face_model_path)

    image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(img_formats)]
    total_images = len(image_paths)
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    lock = threading.Lock()

    def update_progress(processed, skipped):
        nonlocal processed_count, skipped_count
        with lock:
            processed_count += processed
            skipped_count += skipped
            return processed_count + skipped_count

    # Reset cancellation state and prepare for new processing
    processing_state.reset()
    progress_bar = tqdm(total=total_images, unit='image', desc="Extracting faces", file=sys.stdout)

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        processing_state.set_executor(executor)
        futures = [executor.submit(process_image_face, (path, output_folder, padding, save_as_png, face_model)) for path in image_paths]
        for future in as_completed(futures):
            if processing_state.is_cancelled():
                yield "Face extraction cancelled by user."
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
            result, processed, skipped = future.result()
            progress_bar.update(processed + skipped)
            total_processed = update_progress(processed, skipped)
            elapsed_time = time.time() - start_time
            images_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
            images_left = total_images - total_processed
            eta_seconds = images_left / images_per_second if images_per_second > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            progress_msg = f"Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_processed}/{total_images} images. ETA: {eta_str}"
            print(progress_msg)
            yield progress_msg

            if total_processed >= total_images:
                break

    progress_bar.close()
    total_time = time.time() - start_time
    completion_msg = f"Face extraction complete! Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_processed}/{total_images} images in {time.strftime('%H:%M:%S', time.gmtime(total_time))}"
    print(completion_msg)
    yield completion_msg

def calculate_color_hash(img, bins=32):
    """Calculates a color histogram hash for an image."""
    hist = cv2.calcHist([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_color_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    """Compares two color histograms using the specified method."""
    return cv2.compareHist(hist1, hist2, method)

def extract_kaze_features(image):
    """Extracts KAZE features from an image."""
    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2, method='BF', ratio_test=0.75):
    """Matches features between two sets of descriptors."""
    if method == 'BF':
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc1, desc2, k=2)
    elif method == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(desc1, desc2, k=2)
    else:
        raise ValueError("Invalid matching method. Choose 'BF' or 'FLANN'.")

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    return good_matches

def process_image_duplicates(args):
    filename, folder_path, destination_folder, hash_algorithms, hash_size, cutoff, move_duplicates = args
    image_path = os.path.join(folder_path, filename)

    try:
        if isinstance(hash_algorithms, str):
            hash_algorithms = [hash_algorithms]

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            image_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            image_hashes = {}
            duplicates_found = set()

            if 'kaze' in hash_algorithms:
                try:
                    keypoints, descriptors = extract_kaze_features(image_gray)
                    serializable_keypoints = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
                    image_hashes['kaze'] = (serializable_keypoints, descriptors)
                except Exception as e:
                    logging.critical(f"Error processing {filename} with kaze: {str(e)}")

            for alg in hash_algorithms:
                if alg == 'kaze':
                    continue
                try:
                    if alg == 'ahash':
                        hash_obj = imagehash.average_hash(img, hash_size=int(hash_size))
                    elif alg == 'dhash':
                        hash_obj = imagehash.dhash(img, hash_size=int(hash_size))
                    elif alg == 'phash':
                        hash_obj = imagehash.phash(img, hash_size=int(hash_size))
                    elif alg == 'whash':
                        hash_obj = imagehash.whash(img, hash_size=int(hash_size))
                    elif alg == 'colorhash':
                        hash_obj = calculate_color_hash(img)
                    image_hashes[alg] = hash_obj
                except Exception as e:
                    logging.critical(f"Error processing {filename} with {alg}: {str(e)}")
                    continue

            if not image_hashes:
                return f"Failed to generate any hashes for {filename}", 0, 0, filename, []

            return "Processed: {}".format(filename), 0, 0, filename, image_hashes

    except Exception as e:
        logging.critical(f"Error processing {filename}: {str(e)}")
        return f"Error processing {filename}: {str(e)}", 0, 0, filename, {}

def get_all_images(folder_path):
    """Get all images from folder and subfolders"""
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(img_formats):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                image_files.append((full_path, rel_path))
    return image_files

def move_duplicate_file(source_path, dest_folder, source_root):
    """Move duplicate file maintaining folder structure"""
    try:
        rel_path = os.path.relpath(source_path, source_root)
        dest_path = os.path.join(dest_folder, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(source_path, dest_path)
        return dest_path
    except Exception as e:
        logging.critical(f"Error moving file {source_path}: {str(e)}")
        return None

def find_duplicates_multi_hash(folder_path, destination_folder, move_duplicates, hash_algorithms, hash_size, cutoff, num_threads):
    os.makedirs(destination_folder, exist_ok=True)

    # Reset cancellation state and prepare for new processing
    processing_state.reset()
    
    image_files = get_all_images(folder_path)
    total_files = len(image_files)
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    results_list = []

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_image_duplicates,
                          (full_path, folder_path, destination_folder, hash_algorithms,
                           hash_size, cutoff, move_duplicates))
            for full_path, _ in image_files
        ]

        processing_state.set_executor(executor)
        for future in as_completed(futures):
            if processing_state.is_cancelled():
                yield "Duplicate detection cancelled by user."
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
            try:
                result, processed, skipped, filename, image_hashes = future.result()
                results_list.append((filename, image_hashes))

                processed_count += 1

                elapsed_time = time.time() - start_time
                remaining_files = total_files - processed_count
                files_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = remaining_files / files_per_second if files_per_second > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                progress_msg = f"{result} (Processed: {processed_count}, Skipped: {skipped_count}, Total: {processed_count}/{total_files}, ETA: {eta_str})"
                print(progress_msg)
                yield progress_msg

            except Exception as exc:
                logging.critical(f"Generated an exception: {exc}")

    uf = UnionFind()
    all_relationships = []
    all_image_hashes = {}

    for filename, image_hashes in results_list:
        all_image_hashes[filename] = image_hashes
        for alg, current_hash in image_hashes.items():
            if alg not in ['kaze', 'colorhash']:
                current_hash = str(current_hash)

            for other_filename, other_hashes in all_image_hashes.items():
                if filename == other_filename:
                    continue

                if alg in other_hashes:
                    other_hash = other_hashes[alg]
                    if alg not in ['kaze', 'colorhash']:
                        other_hash = str(other_hash)

                    hash_cutoff = float(cutoff.get(alg, 30)) if alg == 'colorhash' else int(cutoff.get(alg, 2))

                    if alg == 'kaze':
                        try:
                            matches = match_features(image_hashes['kaze'][1], other_hashes['kaze'][1])
                            if len(matches) > int(cutoff.get(alg, 30)):
                                uf.union(filename, other_filename)
                                all_relationships.append((filename, other_filename, alg))
                        except Exception as e:
                            logging.critical(f"Error comparing KAZE features: {str(e)}")
                    elif alg == 'colorhash':
                        try:
                            if compare_color_histograms(current_hash, other_hash) < hash_cutoff:
                                uf.union(filename, other_filename)
                                all_relationships.append((filename, other_filename, alg))
                        except Exception as e:
                            logging.critical(f"Error comparing ColorHash: {str(e)}")
                    else:
                        if abs(int(current_hash, 16) - int(other_hash, 16)) < hash_cutoff:
                            uf.union(filename, other_filename)
                            all_relationships.append((filename, other_filename, alg))

    groups = {}
    for file in uf.parent:
        root = uf.find(file)
        if root not in groups:
            groups[root] = set()
        groups[root].add(file)

    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    moved_files = set()

    if move_duplicates:
        moved_count = 0
        for group in duplicate_groups:
            group_list = sorted(list(group))
            original = group_list[0]
            
            for duplicate_file in group_list[1:]:
                if duplicate_file not in moved_files:
                    try:
                        move_duplicate_file(duplicate_file, destination_folder, folder_path)
                        moved_files.add(duplicate_file)
                        moved_count += 1
                        yield f"Moved {duplicate_file} to destination folder (Original: {original})"
                    except Exception as e:
                        logging.critical(f"Error moving duplicate {duplicate_file}: {str(e)}")

        yield f"Moved {moved_count} duplicate files to destination folder"

    with open(os.path.join(destination_folder, 'duplicates.txt'), 'w', encoding='utf-8') as f:
        f.write("Direct relationships found:\n")
        for rel in all_relationships:
            f.write(f"{rel[0]} is a duplicate of {rel[1]} (Method: {rel[2]})\n")

        f.write("\nFinal duplicate groups:\n")
        for group in duplicate_groups:
            sorted_group = sorted(list(group))
            original = sorted_group[0]
            duplicates = sorted_group[1:]
            f.write(f"\nGroup with original: {original}\n")
            for duplicate in duplicates:
                rel_path = os.path.relpath(duplicate, folder_path)
                f.write(f"  - {duplicate} (relative path: {rel_path}) is a duplicate of {original}\n")

    total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
    final_msg = f"Found {total_duplicates} duplicates in {len(duplicate_groups)} groups. Results saved to duplicates.txt"
    print(final_msg)
    yield final_msg

# New function: Generate Tiled Images
def process_single_tiled_image(fname, input_folder, output_folder, tile_width, tile_height, grid_dim, save_as_png, jpeg_quality, overwrite):
    try:
        input_path = os.path.join(input_folder, fname)
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            if width < tile_width or height < tile_height:
                return f"Skipping {fname}: image too small for the tile resolution.", False
            if grid_dim > 1:
                x_step = (width - tile_width) / (grid_dim - 1)
                y_step = (height - tile_height) / (grid_dim - 1)
            else:
                x_step = 0
                y_step = 0
            for i in range(grid_dim):
                for j in range(grid_dim):
                    left = int(round(j * x_step))
                    top = int(round(i * y_step))
                    right = left + tile_width
                    bottom = top + tile_height
                    tile = img.crop((left, top, right, bottom))
                    base_name = os.path.splitext(fname)[0]
                    ext = ".png" if save_as_png else ".jpg"
                    out_filename = f"{base_name}_tile_{i}_{j}{ext}"
                    out_path = os.path.join(output_folder, out_filename)
                    if not overwrite and os.path.exists(out_path):
                        continue
                    if save_as_png:
                        tile.save(out_path, "PNG")
                    else:
                        tile.save(out_path, "JPEG", quality=jpeg_quality)
            return f"Processed {fname}", True
    except Exception as e:
        return f"Error processing {fname}: {str(e)}", False


def generate_tiled_images(input_folder, output_folder, grid_count, tile_resolution, save_as_png, jpeg_quality, num_threads, overwrite):
    import math
    try:
        tile_width, tile_height = map(int, tile_resolution.lower().split('x'))
    except Exception as e:
        yield f"Error: Invalid tile resolution format: {tile_resolution}. Use e.g., 1024x1024."
        return

    grid_count_val = int(grid_count)
    sqrt_val = math.isqrt(grid_count_val)
    if sqrt_val * sqrt_val != grid_count_val:
        yield "Error: Grid count must be a perfect square (e.g., 4, 9, 16)."
        return
    grid_dim = sqrt_val

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(img_formats)]
    total_images = len(image_files)
    if total_images == 0:
        yield "No images found in input folder."
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Reset cancellation state and prepare for new processing
    processing_state.reset()
    
    start_time = time.time()
    processed_count = 0
    lock = threading.Lock()

    progress_bar = tqdm(total=total_images, unit='image', desc="Generating Tiled Images", file=sys.stdout)

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        processing_state.set_executor(executor)
        futures = []
        for fname in image_files:
            if processing_state.is_cancelled():
                yield "Processing cancelled by user."
                break
            futures.append(
                executor.submit(
                    process_single_tiled_image,
                    fname, input_folder, output_folder,
                    tile_width, tile_height, grid_dim,
                    save_as_png, jpeg_quality, overwrite
                )
            )
        for future in as_completed(futures):
            msg, success = future.result()
            with lock:
                processed_count += 1
            progress_bar.update(1)
            elapsed_time = time.time() - start_time
            speed = processed_count / elapsed_time if elapsed_time > 0 else 0
            eta = (total_images - processed_count) / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            yield f"{msg} | Processed: {processed_count}/{total_images} | Speed: {speed:.2f} img/s | ETA: {eta_str}"
            if processing_state.is_cancelled():
                yield "Processing cancelled by user."
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
        progress_bar.close()
        total_time = time.time() - start_time
        yield f"Tile generation complete! Processed {processed_count}/{total_images} images in {time.strftime('%H:%M:%S', time.gmtime(total_time))}."

# UnionFind class (for grouping duplicates)
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

def initialize_sam2_models(model_dir, device):
    """Initialize SAM2 + GroundingDINO using explicit model_dir passed from run_preprocess."""
    try:
        paths = resolve_sam2_model_paths(model_dir)
        validate_sam2_model_paths(paths)

        sam2_model = build_sam2(paths["SAM2_MODEL_CONFIG"], paths["SAM2_CHECKPOINT"], device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        grounding_model = load_model(
            model_config_path=paths["GROUNDING_DINO_CONFIG"],
            model_checkpoint_path=paths["GROUNDING_DINO_CHECKPOINT"],
            device=device
        )
        return {
            "sam2_predictor": sam2_predictor,
            "grounding_model": grounding_model,
        }
    except Exception as e:
        raise Exception(f"Failed to initialize SAM2 models (model_dir={model_dir}): {str(e)}")

def initialize_yolo_models(model_dir, device, selected_class="yolov11l-face.pt"):
    try:
        # Use specific face model if selected
        if selected_class == "yolov11l-face.pt":
            model_path = os.path.join(model_dir, "yolov11l-face.pt")
            model = YOLO(model_path)
        else:
            model_path = os.path.join(model_dir, "yolo11x.pt")
            model = YOLO(model_path)
        model.to(device)
        return {'yolo_model': model}
    except Exception as e:
        raise Exception(f"Failed to initialize YOLO model: {str(e)}")

def process_single_image(image_path, input_folder, output_folder, yolo_folder, 
                        aspect_ratios, save_yolo, overwrite, selected_class,
                        save_as_png, sam2_prompt, models, device, debug_mode, skip_no_detection,
                        padding_value, padding_unit):
    try:
        filename = os.path.basename(image_path)
        aspect_ratios = parse_aspect_ratio_tuples(aspect_ratios)
        
        if sam2_prompt:
            sam2_predictor = models['sam2_predictor']
            grounding_model = models['grounding_model']
            result = process_image((
                filename, input_folder, output_folder, aspect_ratios,
                yolo_folder, save_yolo, overwrite, selected_class,
                save_as_png, sam2_prompt, None, sam2_predictor, grounding_model, debug_mode, skip_no_detection,
                padding_value, padding_unit
            ))
        else:
            yolo_model = models['yolo_model']
            result = process_image((
                filename, input_folder, output_folder, aspect_ratios,
                yolo_folder, save_yolo, overwrite, selected_class,
                save_as_png, None, yolo_model, None, None, debug_mode, skip_no_detection,
                padding_value, padding_unit
            ))
        
        return result
    except Exception as e:
        raise Exception(f"Error processing {image_path}: {str(e)}")

def process_single_image_with_stop_check(image_path, input_folder, output_folder, yolo_folder, 
                                       aspect_ratios, save_yolo, overwrite, selected_class,
                                       save_as_png, sam2_prompt, models, device, stop_processing, debug_mode, skip_no_detection,
                                       padding_value, padding_unit):
    if stop_processing.value:
        return None

    try:
        filename = os.path.basename(image_path)
        result = process_single_image(
            image_path, input_folder, output_folder, yolo_folder,
            aspect_ratios, save_yolo, overwrite, selected_class,
            save_as_png, sam2_prompt, models, device, debug_mode, skip_no_detection,
            padding_value, padding_unit
        )
        
        if stop_processing.value:
            return None
            
        return result
        
    except Exception as e:
        if not stop_processing.value:
            raise Exception(f"Error processing {image_path}: {str(e)}")
        return None

def worker_process(gpu_id, worker_id, tasks, input_folder, output_folder, yolo_folder, 
                  aspect_ratios, save_yolo, overwrite, selected_class, save_as_png, 
                  sam2_prompt, results_queue, processed_files_queue, processed_count, 
                  stop_processing, status_queue, big_lock, debug_mode, skip_no_detection,
                  padding_value, padding_unit, model_dir):

    try:
        global debug
        debug = debug_mode
        if debug:
            logging.critical(f"Worker {worker_id} starting on GPU {gpu_id} with debug mode: {debug}")
        for task_idx, image_path in enumerate(tasks, 1):
            if stop_processing.value:
                status_queue.put(f"STOPPED:Worker {worker_id} on GPU {gpu_id} stopping")
                return

        device = f"cuda:{gpu_id}"
        status_queue.put(f"Worker {worker_id} starting on GPU {gpu_id}")

        try:
            if sam2_prompt:
                models = initialize_sam2_models(model_dir, device)
            else:
                models = initialize_yolo_models(model_dir, device, selected_class)
        except Exception as e:
            status_queue.put(f"ERROR:Failed to initialize models on GPU {gpu_id}: {str(e)}")
            return

        total_tasks = len(tasks)
        start_time = time.time()
        worker_processed = 0

        for task_idx, image_path in enumerate(tasks, 1):
            if stop_processing.value:
                status_queue.put(f"STOPPED:Worker {worker_id} on GPU {gpu_id} stopping")
                return

            try:
                if not stop_processing.value:
                    result = process_single_image_with_stop_check(
                        image_path, input_folder, output_folder, yolo_folder,
                        aspect_ratios, save_yolo, overwrite, selected_class,
                        save_as_png, sam2_prompt, models, device, stop_processing, debug, skip_no_detection,
                        padding_value, padding_unit
                    )

                    if stop_processing.value:
                        return

                    with big_lock:
                        processed_count.value += 1
                        worker_processed += 1

                        elapsed_time = time.time() - start_time
                        current_speed = task_idx / elapsed_time if elapsed_time > 0 else 0
                        eta = (total_tasks - task_idx) / current_speed if current_speed > 0 else 0

                        status_msg = (
                            f"PROGRESS:Worker {worker_id} | "
                            f"GPU {gpu_id} | "
                            f"Processed: {worker_processed}/{total_tasks} | "
                            f"Speed: {current_speed:.2f} img/s | "
                            f"ETA: {eta:.1f}s | "
                            f"Current: {os.path.basename(image_path)}"
                        )
                        status_queue.put(status_msg)
                        results_queue.put(result)
                        processed_files_queue.put(image_path)

            except Exception as e:
                if not stop_processing.value:
                    status_queue.put(f"ERROR:Error processing {image_path} on GPU {gpu_id}: {str(e)}")

    except Exception as e:
        if not stop_processing.value:
            status_queue.put(f"ERROR:Critical error in worker {worker_id} on GPU {gpu_id}: {str(e)}")

def distribute_tasks(image_files, gpu_ids, batch_size):
    num_gpus = len(gpu_ids)
    num_workers_per_gpu = batch_size
    total_workers = num_gpus * num_workers_per_gpu

    tasks_per_worker = [[] for _ in range(total_workers)]
    for i, image_file in enumerate(image_files):
        worker_index = i % total_workers
        tasks_per_worker[worker_index].append(image_file)

    tasks_per_gpu = [
        tasks_per_worker[i:i+num_workers_per_gpu] 
        for i in range(0, total_workers, num_workers_per_gpu)
    ]

    if debug:
        print("\n=== Task Distribution ===")
        for gpu_idx, gpu_tasks in enumerate(tasks_per_gpu):
            print(f"\nGPU {gpu_ids[gpu_idx]}:")
            for worker_idx, worker_tasks in enumerate(gpu_tasks):
                print(f"  Worker {worker_idx}: {len(worker_tasks)} tasks")

    return tasks_per_gpu

def print_task_distribution(tasks_per_gpu, gpu_ids):
    print("\nTask Distribution Summary:")
    total_tasks = 0
    for gpu_idx, gpu_tasks in enumerate(tasks_per_gpu):
        gpu_total = sum(len(worker_tasks) for worker_tasks in gpu_tasks)
        total_tasks += gpu_total
        print(f"\nGPU {gpu_ids[gpu_idx]}:")
        for worker_idx, worker_tasks in enumerate(gpu_tasks):
            print(f"  Worker {worker_idx}: {len(worker_tasks)} tasks")
        print(f"  Total for GPU {gpu_ids[gpu_idx]}: {gpu_total} tasks")
    print(f"\nTotal tasks across all GPUs: {total_tasks}")

# Cancel handler functions for various tabs
def cancel_tiled_handler():
    processing_state.stop_processing()
    time.sleep(1)
    return "Tiled Image generation cancelled."

def cancel_resizer_handler():
    processing_state.stop_processing()
    time.sleep(1)
    return "Image resizing cancelled."

def cancel_faces_handler():
    processing_state.stop_processing()
    time.sleep(1)
    return "Face extraction cancelled."

def cancel_duplicates_handler():
    processing_state.stop_processing()
    time.sleep(1)
    return "Duplicate detection cancelled."


def detect_face_or_object(img_array, selected_class, sam2_prompt, model, sam2_predictor, grounding_model, debug_mode=False):
    """
    Shared detection function that can be used by both Image Cropper and Extract Faces tabs.
    Returns bbox coordinates or None if no detection found.
    """
    debug = debug_mode
    original_height, original_width = img_array.shape[:2]
    
    if sam2_prompt:
        if debug:
            logging.critical("\n=== SAM2 Detection ===")
            logging.critical(f"Input array shape to SAM2: {img_array.shape}")
        
        # Create a temporary image path for SAM2 processing
        temp_img = Image.fromarray(img_array)
        temp_path = f"temp_detection_{os.getpid()}.png"
        temp_img.save(temp_path)
        
        try:
            segmented_image, mask_image, mask_coordinates, _, _ = sam2_process_image(
                img_array,
                sam2_prompt,
                0.35,
                0.25,
                True,
                sam2_predictor,
                grounding_model,
                None,
                None,
                None,
                None,
                None,
                temp_path
            )
            
            if mask_coordinates and len(mask_coordinates) > 0:
                if debug:
                    logging.critical(f"Number of SAM2 coordinates: {len(mask_coordinates)}")
                
                try:
                    x_coords = [p[1] for p in mask_coordinates if len(p) >= 2]
                    y_coords = [p[0] for p in mask_coordinates if len(p) >= 2]
                except (IndexError, TypeError) as e:
                    if debug:
                        logging.critical(f"Error extracting SAM2 coordinates: {e}")
                    return None
                
                if len(x_coords) == 0 or len(y_coords) == 0:
                    if debug:
                        logging.critical("Empty SAM2 coordinate arrays")
                    return None
                
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                bbox = [int(x) for x in bbox]
                
                # Validate bbox dimensions
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    if debug:
                        logging.critical(f"Invalid SAM2 bbox dimensions: {bbox}")
                    return None
                
                # Ensure bbox is within image bounds
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(original_width, bbox[2])
                bbox[3] = min(original_height, bbox[3])
                
                if debug:
                    logging.critical(f"SAM2 final bbox: {bbox}")
                
                return bbox
            else:
                if debug:
                    logging.critical("No SAM2 mask coordinates found")
                return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    else:
        if debug:
            logging.critical("\n=== YOLO Detection ===")
        
        results = model(img_array)
        detections = results[0].boxes
        
        # Special handling for yolov11l-face.pt model
        if selected_class == "yolov11l-face.pt":
            # For face model, take the first detection (all detections are faces)
            if len(detections) > 0:
                bbox = detections[0].xyxy[0].tolist()
                bbox = [int(x) for x in bbox]
                if debug:
                    logging.critical(f"YOLO Face Model bbox: {bbox}")
                return bbox
            else:
                if debug:
                    logging.critical("No face detected with YOLO face model")
                return None
        else:
            # Standard YOLO class detection
            class_detections = [box for box in detections if model.names[int(box.cls)] == selected_class]
            
            if class_detections:
                bbox = class_detections[0].xyxy[0].tolist()
                bbox = [int(x) for x in bbox]
                if debug:
                    logging.critical(f"YOLO bbox: {bbox}")
                return bbox
            else:
                if debug:
                    logging.critical(f"No {selected_class} detected with YOLO")
                return None
    
    return None

def process_image_face_enhanced(args):
    """
    Enhanced face extraction function that supports both YOLO and SAM2 detection methods.
    """
    input_path, output_root, padding, save_as_png, selected_class, sam2_prompt, device, debug_mode, model_dir = args
    
    try:
        debug = debug_mode
        if debug:
            logging.critical(f"\n=== Enhanced Face Processing: {input_path} ===")
        
        # Initialize models inside worker process to avoid serialization issues
        if sam2_prompt:
            if debug:
                logging.critical("Initializing SAM2 models in worker process...")
            models = initialize_sam2_models(model_dir, device)
            sam2_predictor = models.get('sam2_predictor')
            grounding_model = models.get('grounding_model')
            yolo_model = None
        else:
            if debug:
                logging.critical("Initializing YOLO models in worker process...")
            models = initialize_yolo_models(model_dir, device, selected_class)
            yolo_model = models.get('yolo_model')
            sam2_predictor = None
            grounding_model = None
        
        # Load and convert image
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
        
        # Detect face/object using shared detection function
        bbox = detect_face_or_object(img_array, selected_class, sam2_prompt, yolo_model, sam2_predictor, grounding_model, debug_mode)
        
        if bbox is None:
            detection_method = "SAM2" if sam2_prompt else "YOLO"
            target = sam2_prompt if sam2_prompt else selected_class
            return f"No {target} detected with {detection_method} in {os.path.basename(input_path)}", 0, 1
        
        # Apply padding
        x1, y1, x2, y2 = bbox
        img_height, img_width = img_array.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Extract face region
        face_image = img_array[y1:y2, x1:x2]
        
        # Save extracted face
        os.makedirs(output_root, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + ('.png' if save_as_png else '.jpg')
        output_path = os.path.join(output_root, output_filename)
        
        # Convert back to PIL for saving
        face_pil = Image.fromarray(face_image)
        
        if save_as_png:
            face_pil.save(output_path, 'PNG')
        else:
            face_pil.save(output_path, 'JPEG', quality=100)
        
        if debug:
            logging.critical(f"Saved face to: {output_path}")
        
        return f"Processed: {os.path.basename(input_path)}", 1, 0
        
    except Exception as e:
        if debug:
            logging.critical(f"Error processing {input_path}: {str(e)}")
        return f"Error processing {os.path.basename(input_path)}: {str(e)}", 0, 1

def extract_faces_enhanced(input_folder, output_folder, padding, save_as_png, num_threads, selected_class, sam2_prompt, device="cuda:0", debug_mode=False, model_dir="model"):
    """
    Enhanced face extraction function that supports both YOLO and SAM2 detection methods.
    """
    try:
        debug = debug_mode
        if debug:
            logging.critical("Starting enhanced face extraction...")
        
        # Get image paths
        image_paths = [
            os.path.join(input_folder, fname) 
            for fname in os.listdir(input_folder) 
            if fname.lower().endswith(img_formats)
        ]
        
        total_images = len(image_paths)
        if total_images == 0:
            yield "No images found in input folder."
            return
        
        # Reset cancellation state and prepare for new processing
        processing_state.reset()
        
        processed_count = 0
        skipped_count = 0
        start_time = time.time()
        lock = threading.Lock()
        
        def update_progress(processed, skipped):
            nonlocal processed_count, skipped_count
            with lock:
                processed_count += processed
                skipped_count += skipped
                return processed_count + skipped_count
        
        progress_bar = tqdm(total=total_images, unit='image', desc="Extracting faces (enhanced)", file=sys.stdout)
        
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_image_face_enhanced,
                    (path, output_folder, padding, save_as_png, selected_class, sam2_prompt, device, debug_mode, model_dir)
                )
                for path in image_paths
            ]
            
            processing_state.set_executor(executor)
            for future in as_completed(futures):
                if processing_state.is_cancelled():
                    yield "Face extraction cancelled by user."
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                    
                try:
                    result, processed, skipped = future.result()
                    progress_bar.update(processed + skipped)
                    total_processed = update_progress(processed, skipped)
                    
                    elapsed_time = time.time() - start_time
                    images_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                    images_left = total_images - total_processed
                    eta_seconds = images_left / images_per_second if images_per_second > 0 else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    
                    detection_method = "SAM2" if sam2_prompt else "YOLO"
                    target = sam2_prompt if sam2_prompt else selected_class
                    
                    progress_msg = f"[{detection_method}] Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_processed}/{total_images} images. Target: {target}, ETA: {eta_str}"
                    print(progress_msg)
                    yield progress_msg
                    
                    if total_processed >= total_images:
                        break
                        
                except Exception as exc:
                    logging.critical(f"Error in face extraction: {exc}")
                    yield f"Error: {str(exc)}"
        
        progress_bar.close()
        total_time = time.time() - start_time
        completion_msg = f"Enhanced face extraction complete! Processed: {processed_count}, Skipped: {skipped_count}, Total: {processed_count + skipped_count}/{total_images} images in {time.strftime('%H:%M:%S', time.gmtime(total_time))}"
        print(completion_msg)
        yield completion_msg
        
    except Exception as e:
        error_msg = f"Error in enhanced face extraction: {str(e)}"
        logging.critical(error_msg)
        yield error_msg


def _print_generator(gen):
    """Consume a generator that yields progress strings (or dicts) and print them."""
    for item in gen:
        if isinstance(item, dict):
            txt = None
            for v in item.values():
                if isinstance(v, str):
                    txt = v
                    break
            if txt:
                print(txt, flush=True)
            continue
        print(str(item), flush=True)

def main():
    """
    CLI entrypoint (Gradio removed).
    Mirrors the Gradio tabs as subcommands.
    """
    global debug

    parser = argparse.ArgumentParser(
        description="Image processing CLI (crop/resize/etc) - Gradio removed"
    )
    parser.add_argument("--model-dir", default="model", help="Directory containing model files (default: model)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- crop ---
    p_crop = sub.add_parser("crop", help="Crop images using YOLO (and optionally SAM2) to given aspect ratios")
    p_crop.add_argument("--input", required=True, help="Input folder with raw images")
    p_crop.add_argument("--output", required=True, help="Output folder for cropped images")
    p_crop.add_argument("--aspect-ratios", default="1x1", help="Comma-separated, e.g. 3x4,4x5,1x1")
    p_crop.add_argument("--yolo-folder", default="", help="Folder to save YOLO/SAM2 visualizations (optional)")
    p_crop.add_argument("--save-yolo", action="store_true", help="Save YOLO/SAM2 outputs to --yolo-folder")
    p_crop.add_argument("--batch-size", type=int, default=1, help="Images per batch")
    p_crop.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids, e.g. 0 or 0,1")
    p_crop.add_argument("--overwrite", action="store_true", help="Overwrite existing cropped files")
    p_crop.add_argument("--class", dest="selected_class", default="person", help="YOLO class (default: person)")
    p_crop.add_argument("--png", action="store_true", help="Save outputs as PNG (default is JPG)")
    p_crop.add_argument("--sam2-prompt", default="", help="If set, use SAM2 segmenter prompt (e.g. 'face.').")
    p_crop.add_argument("--debug", action="store_true", help="Enable verbose logging")
    p_crop.add_argument("--skip-no-detection", action="store_true",
                        help="Skip images where no detection is found (instead of falling back to center crop)")
    p_crop.add_argument("--padding-value", type=float, default=0.0, help="Extra padding around detected box")
    p_crop.add_argument("--padding-unit", choices=["percent", "pixels"], default="percent",
                        help="Interpret padding as percent of bbox size or absolute pixels (default: percent)")

    # --- resize ---
    p_resize = sub.add_parser("resize", help="Resize images to one or more resolutions")
    p_resize.add_argument("--input", required=True,
                          help="Input folder; expects subfolders per resolution, e.g. input/1024x1024/*.png")
    p_resize.add_argument("--output", required=True, help="Output folder root")
    p_resize.add_argument("--resolutions", default="1024x1024",
                          help="Comma-separated, e.g. 1024x1024,1280x720")
    p_resize.add_argument("--png", action="store_true", help="Save outputs as PNG (default is JPG)")
    p_resize.add_argument("--threads", type=int, default=4, help="Number of worker processes")
    p_resize.add_argument("--overwrite", action="store_true", help="Overwrite existing resized files")
    p_resize.add_argument("--no-crop", action="store_true",
                          help="Do not crop; letterbox with white background to match target aspect ratio")

    # --- move low-res ---
    p_mlr = sub.add_parser("move-low-res", help="Move low-resolution images between folders")
    p_mlr.add_argument("--folder-a", required=True, help="Folder A (cropped images)")
    p_mlr.add_argument("--folder-b", required=True, help="Folder B (raw images)")
    p_mlr.add_argument("--folder-c", required=True, help="Folder C (raw low-res destination)")
    p_mlr.add_argument("--folder-d", required=True, help="Folder D (cropped low-res destination)")
    p_mlr.add_argument("--min-width", type=int, default=1536)
    p_mlr.add_argument("--min-height", type=int, default=1536)

    # --- rename ---
    p_ren = sub.add_parser("rename", help="Rename images in a folder")
    p_ren.add_argument("--folder", required=True, help="Folder to rename files within")
    p_ren.add_argument("--start", type=int, default=1, help="Start number")

    # --- extract faces ---
    p_faces = sub.add_parser("extract-faces", help="Extract faces/objects from images (YOLO + optional SAM2)")
    p_faces.add_argument("--input", required=True, help="Input folder")
    p_faces.add_argument("--output", required=True, help="Output folder")
    p_faces.add_argument("--padding", type=float, default=0.0, help="Padding around detected bbox (percent)")
    p_faces.add_argument("--png", action="store_true", help="Save as PNG (default is JPG)")
    p_faces.add_argument("--threads", type=int, default=4, help="Number of worker processes")
    p_faces.add_argument("--class", dest="selected_class", default="person",
                         help="YOLO class to extract (default: person)")
    p_faces.add_argument("--sam2-prompt", default="", help="If set, run SAM2 segmentation with this prompt")
    p_faces.add_argument("--gpu-id", type=int, default=0, help="GPU id for SAM2 model (if used)")
    p_faces.add_argument("--debug", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.cmd == "crop":
        debug = bool(args.debug)
        if args.save_yolo and not args.yolo_folder:
            raise SystemExit("--save-yolo requires --yolo-folder")
        gen = crop_images(
            args.input,
            args.output,
            args.aspect_ratios,
            args.yolo_folder if args.yolo_folder else None,
            args.save_yolo,
            args.batch_size,
            args.gpu_ids,
            args.overwrite,
            args.selected_class,
            args.png,
            args.sam2_prompt,
            debug_mode=args.debug,
            skip_no_detection=args.skip_no_detection,  
            padding_value=args.padding_value,
            padding_unit=args.padding_unit,
            model_dir=args.model_dir,
        )
        _print_generator(gen)
        return

    if args.cmd == "resize":
        gen = resize_images(
            args.model_dir,
            args.input,
            args.output,
            args.resolutions,
            args.png,
            num_threads=args.threads,
            overwrite=args.overwrite,
            no_crop=args.no_crop,
        )
        _print_generator(gen)
        return

    if args.cmd == "move-low-res":
        result = move_low_res_files(
            args.folder_a, args.folder_b, args.folder_c, args.folder_d, args.min_width, args.min_height
        )
        print(result, flush=True)
        return

    if args.cmd == "rename":
        result = rename_files(args.folder, args.start)
        print(result, flush=True)
        return

    if args.cmd == "extract-faces":
        debug = bool(args.debug)
        gen = extract_faces_enhanced(
            args.input,
            args.output,
            args.padding,
            args.png,
            args.threads,
            args.selected_class,
            args.sam2_prompt,
            args.gpu_id,
            args.debug,
            model_dir=args.model_dir,
        )
        _print_generator(gen)
        return


if __name__ == "__main__":
    main()

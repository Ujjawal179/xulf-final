"""
run_final.py

Pipeline:
0) Clone Repo, cd into it, run this script with necessary args
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)
5) Resolve toml file
6) Launch kohya training using the resolved toml file
7) Upload trained model to R2 storage
8) Call webhook on completion/failure
"""

import argparse
import logging
import os
import zipfile
import urllib.request
import subprocess
import json
import hashlib
import hmac
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# R2/S3 upload
try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
    print("Warning: boto3 not installed, R2 upload will be skipped")

# HTTP requests for webhook
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    # Fallback to urllib
    import urllib.request as urllib_req

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# ============== R2 Upload Functions ==============

def get_r2_client():
    """Create R2/S3 client from environment variables."""
    if not HAS_BOTO:
        return None
    
    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    
    if not all([endpoint, access_key, secret_key]):
        logging.warning("R2 credentials not fully configured, skipping R2 upload")
        return None
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def upload_to_r2(local_path: Path, r2_key: str) -> Optional[str]:
    """
    Upload a file to R2 storage.
    Returns the public URL if successful, None otherwise.
    """
    client = get_r2_client()
    if not client:
        return None
    
    bucket = os.environ.get("R2_BUCKET")
    public_base = os.environ.get("R2_PUBLIC_BASE", "").rstrip("/")
    
    if not bucket:
        logging.error("R2_BUCKET not set")
        return None
    
    try:
        logging.info(f"Uploading {local_path} to R2: {r2_key}")
        
        # Determine content type
        suffix = local_path.suffix.lower()
        content_type = {
            ".safetensors": "application/octet-stream",
            ".bin": "application/octet-stream",
            ".json": "application/json",
            ".txt": "text/plain",
            ".log": "text/plain",
        }.get(suffix, "application/octet-stream")
        
        client.upload_file(
            str(local_path),
            bucket,
            r2_key,
            ExtraArgs={"ContentType": content_type},
        )
        
        public_url = f"{public_base}/{r2_key}" if public_base else None
        logging.info(f"Uploaded to R2: {r2_key}")
        if public_url:
            logging.info(f"Public URL: {public_url}")
        
        return public_url
    except Exception as e:
        logging.error(f"R2 upload failed: {e}")
        return None


def upload_training_outputs(saves_dir: Path, r2_out_key: str) -> Dict[str, Optional[str]]:
    """
    Upload all training outputs to R2.
    Returns dict of {filename: public_url}.
    """
    results = {}
    
    if not saves_dir.exists():
        logging.warning(f"Saves directory does not exist: {saves_dir}")
        return results
    
    # Find all safetensors and related files
    for pattern in ["*.safetensors", "*.json", "*.txt", "*.log"]:
        for file_path in saves_dir.glob(pattern):
            # Build R2 key - use the r2_out_key as base path
            r2_base = r2_out_key.rsplit("/", 1)[0] if "/" in r2_out_key else "lora"
            r2_key = f"{r2_base}/{file_path.name}"
            
            url = upload_to_r2(file_path, r2_key)
            results[file_path.name] = url
    
    # Also look for the main model file specifically
    main_model_patterns = ["*.safetensors"]
    for pattern in main_model_patterns:
        for file_path in saves_dir.glob(f"**/{pattern}"):
            if file_path.name not in results:
                r2_base = r2_out_key.rsplit("/", 1)[0] if "/" in r2_out_key else "lora"
                r2_key = f"{r2_base}/{file_path.name}"
                url = upload_to_r2(file_path, r2_key)
                results[file_path.name] = url
    
    return results


# ============== Webhook Functions ==============

def sign_webhook_payload(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def call_webhook(
    webhook_url: str,
    status: str,
    job_id: str,
    model_id: str = "",
    user_id: str = "",
    lora_url: Optional[str] = None,
    error_message: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Call the webhook endpoint with job status.
    Returns True if successful, False otherwise.
    """
    if not webhook_url:
        logging.warning("No webhook URL configured, skipping callback")
        return False
    
    webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
    
    payload = {
        "status": status,  # "completed", "failed", "error"
        "request_id": job_id,
        "job_id": job_id,
        "model_id": model_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "azure_ml",
    }
    
    if lora_url:
        payload["lora_url"] = lora_url
        # Also include in result format expected by existing webhook handler
        payload["result"] = {
            "diffusers_lora_file": {"url": lora_url},
        }
    
    if error_message:
        payload["error"] = error_message
    
    if extra_data:
        payload.update(extra_data)
    
    payload_json = json.dumps(payload, separators=(",", ":"))
    
    headers = {
        "Content-Type": "application/json",
    }
    
    if webhook_secret:
        signature = sign_webhook_payload(payload_json, webhook_secret)
        headers["X-Webhook-Signature"] = signature
        headers["X-Fal-Signature"] = signature  # Compatibility with existing handler
    
    try:
        logging.info(f"Calling webhook: {webhook_url}")
        logging.info(f"Payload: {payload_json[:500]}...")
        
        if HAS_REQUESTS:
            resp = requests.post(
                webhook_url,
                data=payload_json,
                headers=headers,
                timeout=30,
            )
            success = resp.status_code < 400
            logging.info(f"Webhook response: {resp.status_code} {resp.text[:200]}")
        else:
            # Fallback to urllib
            req = urllib_req.Request(
                webhook_url,
                data=payload_json.encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib_req.urlopen(req, timeout=30) as resp:
                success = resp.status < 400
                logging.info(f"Webhook response: {resp.status}")
        
        return success
    except Exception as e:
        logging.error(f"Webhook call failed: {e}")
        return False


# ============== Original Pipeline Functions ==============

def download_zip(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(zip_path))
    logging.info("Downloaded zip file to %s", zip_path)


def unzip_dataset(zip_path: Path, extract_path: Path) -> None:
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(extract_path))
    logging.info("Extracted zip file to %s", extract_path)


def pick_image_dir(root: Path) -> Path:
    """
    If root directly contains images: return root, If root contains exactly one subdir: recurse into it (common zip layout)
    Otherwise return root (for caller to handle)
    """
    if not root.exists():
        return root

    files = [p for p in root.iterdir() if p.is_file()]
    if any(p.suffix.lower() in IMG_EXTS for p in files):
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return pick_image_dir(subdirs[0])

    return root


def consume_generator(gen) -> None:
    for msg in gen:
        logging.info("%s", msg)


def run_cmd(cmd, check=True):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=check)

def main() -> None:
    parser = argparse.ArgumentParser()

    #Important arguments that will be given either in command or in azure input
        #Img Proc args
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--dest_root", type=str, default="Dataset/final-imgs", help='Dest Folder with concept)')
    parser.add_argument("--concept", default="1_ohwx_woman", help="Kohya concept folder name")

        #Kohya args
    parser.add_argument("--flux_dir", required=True, help="Azure input: flux weights directory")
    parser.add_argument("--saves_dir", required=True, help="Azure output: model save directory")
    parser.add_argument("--config", default="setupCodes/runConfig.toml", help="Base TOML config")

    # R2 and webhook arguments (passed from Azure job env or command line)
    parser.add_argument("--r2_out_key", type=str, default="", help="R2 key for output model (e.g. lora/user/model/model.safetensors)")
    parser.add_argument("--webhook_url", type=str, default="", help="Webhook URL to call on completion")
    parser.add_argument("--job_id", type=str, default="", help="Job ID for tracking")
    parser.add_argument("--user_id", type=str, default="", help="User ID for tracking")
    parser.add_argument("--model_id", type=str, default="", help="Model ID for tracking")
    parser.add_argument("--trigger_word", type=str, default="ohwx", help="Trigger word used in training")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")

    #These arguments can be left as default
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/temp-imgs")
    parser.add_argument("--img_output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
    

    parser.add_argument("--aspect_ratios", type=str, default="1024x1024", help='e.g. "1024x1024" or "3x4,4x5"')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--selected_class", type=str, default="person")
    parser.add_argument("--overwrite", action="store_true", default=True)

    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = base / zip_path

    extract_path = Path(args.extract_path)
    if not extract_path.is_absolute():
        extract_path = base / extract_path

    img_output_root = Path(args.img_output_root)
    if not img_output_root.is_absolute():
        img_output_root = base / img_output_root

    dest_root = Path(args.dest_root)
    if not dest_root.is_absolute():
        dest_root = base / dest_root

    dest_path = dest_root / args.concept
    dest_path.mkdir(parents=True, exist_ok=True)

    #Below maybe needs checking for azure
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("img_output_root  =", img_output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(img_output_root),
        aspect_ratios=str(args.aspect_ratios),
        yolo_folder=None,
        save_yolo=False,
        batch_size=int(args.batch_size),
        gpu_ids=str(args.gpu_ids),
        overwrite=bool(args.overwrite),
        selected_class=str(args.selected_class),
        save_as_png=False,
        sam2_prompt=False,
        debug_mode=False,
        skip_no_detection=False,
        padding_value=0,
        padding_unit="percent",
        model_dir=str(model_dir),
    )

    consume_generator(gen)
    logging.info("Cropped at:", img_output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(img_output_root),
            output_folder=str(dest_path),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", dest_root)


    ############KOHYA TRAINING ##############
    flux_dir = Path(args.flux_dir).resolve()
    saves_dir = Path(args.saves_dir).resolve()
    logging.info(f"FLUX_DIR={flux_dir}")
    logging.info(f"SAVES_DIR={saves_dir}")

    config_src = Path(args.config).read_text()

    resolved = (
        config_src
        .replace("${{inputs.flux_weights}}", str(flux_dir))
        .replace("${{inputs.train_data}}", str(dest_root))
        .replace("${{outputs.ModelSaves}}", str(saves_dir))
    )

    resolved_path = Path("runConfig.resolved.toml")
    resolved_path.write_text(resolved)

    print("Wrote", resolved_path)
    print("Resolved train_data_dir:", dest_root)
    print("Resolved pretrained model:", flux_dir / "flux1-dev.safetensors")

    script_dir = Path(__file__).parent
    # flux_train_script = script_dir / ".." / "kohya_ss" / "sd-scripts" / "flux_train_network.py"

    flux_train_script = "kohya_ss/sd-scripts/flux_train_network.py"
    
    # Track training success/failure for webhook
    training_success = False
    training_error = None
    
    try:
        run_cmd([
            "accelerate", "launch",
            "--num_machines", "1",
            "--num_processes", "1",
            "--mixed_precision", "bf16",
            "--dynamo_backend", "no",
            str(flux_train_script),
            "--config_file", str(resolved_path),
        ])
        training_success = True
        logging.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        training_error = f"Training failed with exit code {e.returncode}"
        logging.error(training_error)
    except Exception as e:
        training_error = f"Training failed: {str(e)}"
        logging.error(training_error)

    # ============== R2 Upload ==============
    lora_url = None
    uploaded_files = {}
    
    if training_success and args.r2_out_key:
        logging.info("=== Uploading outputs to R2 ===")
        try:
            uploaded_files = upload_training_outputs(saves_dir, args.r2_out_key)
            
            # Find the main model URL (first .safetensors file)
            for filename, url in uploaded_files.items():
                if filename.endswith(".safetensors") and url:
                    lora_url = url
                    break
            
            if lora_url:
                logging.info(f"Main LoRA model uploaded: {lora_url}")
            else:
                logging.warning("No .safetensors file found in outputs")
        except Exception as e:
            logging.error(f"R2 upload failed: {e}")
    elif not args.r2_out_key:
        logging.info("No R2 output key specified, skipping R2 upload")

    # ============== Webhook Callback ==============
    webhook_url = args.webhook_url or os.environ.get("WEBHOOK_URL", "")
    job_id = args.job_id or os.environ.get("JOB_ID", "")
    user_id = args.user_id or os.environ.get("USER_ID", "")
    model_id = args.model_id or os.environ.get("MODEL_ID", "")
    
    if webhook_url:
        logging.info("=== Calling webhook ===")
        
        extra_data = {
            "trigger_word": args.trigger_word,
            "steps": args.steps,
            "uploaded_files": {k: v for k, v in uploaded_files.items() if v},
        }
        
        if training_success:
            call_webhook(
                webhook_url=webhook_url,
                status="completed",
                job_id=job_id,
                model_id=model_id,
                user_id=user_id,
                lora_url=lora_url,
                extra_data=extra_data,
            )
        else:
            call_webhook(
                webhook_url=webhook_url,
                status="failed",
                job_id=job_id,
                model_id=model_id,
                user_id=user_id,
                error_message=training_error,
                extra_data=extra_data,
            )
    else:
        logging.info("No webhook URL configured, skipping callback")

    # Exit with appropriate code
    if not training_success:
        logging.error("Pipeline failed")
        exit(1)
    
    logging.info("=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()

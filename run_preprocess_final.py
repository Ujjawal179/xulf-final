"""
run_preprocess.py

To Dos:
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)
"""

import argparse
import logging
import os
import zipfile
import urllib.request
from pathlib import Path

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/Imgs")

    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
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

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base / output_root

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("output_root  =", output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(output_root),
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
    logging.info("Cropped at:", output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(output_root),
            output_folder=str(output_root),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", output_root)


if __name__ == "__main__":
    main()
"""
run_preprocess.py

To Dos:
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)
"""

import argparse
import logging
import os
import zipfile
import urllib.request
from pathlib import Path

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/Imgs")

    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
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

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base / output_root

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("output_root  =", output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(output_root),
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
    logging.info("Cropped at:", output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(output_root),
            output_folder=str(output_root),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", output_root)


if __name__ == "__main__":
    main()
"""
run_preprocess.py

To Dos:
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)
"""

import argparse
import logging
import os
import zipfile
import urllib.request
from pathlib import Path

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/Imgs")

    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
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

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base / output_root

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("output_root  =", output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(output_root),
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
    logging.info("Cropped at:", output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(output_root),
            output_folder=str(output_root),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", output_root)


if __name__ == "__main__":
    main()
"""
run_preprocess.py

To Dos:
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)
"""

import argparse
import logging
import os
import zipfile
import urllib.request
from pathlib import Path

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/Imgs")

    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
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

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base / output_root

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("output_root  =", output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(output_root),
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
    logging.info("Cropped at:", output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(output_root),
            output_folder=str(output_root),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", output_root)


if __name__ == "__main__":
    main()

# train_flux_kohya.py previously
import argparse
import os
import subprocess
from pathlib import Path
import sys

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
CAPTION_EXTS = {".txt"}


def run(cmd, check=True):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux_dir", required=True, help="Azure input: flux weights directory")
    parser.add_argument("--train_dir", required=True, help="Azure input: training images directory")
    parser.add_argument("--out_dir", required=True, help="Azure output: model save directory")
    parser.add_argument("--concept", default="1_ohwx", help="Kohya concept folder name")
    parser.add_argument("--config", default="runConfig.toml", help="Base TOML config")

    args = parser.parse_args()

    flux_dir = Path(args.flux_dir).resolve()
    train_dir = Path(args.train_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"FLUX_DIR={flux_dir}")
    print(f"TRAIN_DIR={train_dir}")
    print(f"OUT_DIR={out_dir}")

    # 1) Converts Data to Kohya dataset structure. Can apply repetitions + augmentations in the concept name. Repetitions not yet supported here.

    dataset_root = Path.cwd() / "kohya_dataset"
    img_parent = dataset_root / "img"
    concept_dir = img_parent / args.concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    print("Inspect TRAIN_DIR (first 50 files):")
    for i, p in enumerate(train_dir.iterdir()):
        if i >= 50:
            break
        print(" ", p.name)

    linked = 0
    for p in train_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS | CAPTION_EXTS:
            target = concept_dir / p.name
            if not target.exists():
                target.symlink_to(p)
                linked += 1

    print(f"Linked files: {linked}")

    images = [p for p in concept_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if len(images) == 0:
        print("ERROR: No images found for training.", file=sys.stderr)
        sys.exit(2)

    print(f"Total images in concept dir: {len(images)}")

    # 2) Corrects TOML with resolved paths from Azure

    config_src = Path(args.config).read_text()

    resolved = (
        config_src
        .replace("${{inputs.flux_weights}}", str(flux_dir))
        .replace("${{inputs.train_data}}", str(img_parent))
        .replace("${{outputs.ModelSaves}}", str(out_dir))
    )

    resolved_path = Path("runConfig.resolved.toml")
    resolved_path.write_text(resolved)

    print("Wrote", resolved_path)
    print("Resolved train_data_dir:", img_parent)
    print("Resolved pretrained model:", flux_dir / "flux1-dev.safetensors")

    # 3) Launches training with relative path from docker img working dir
    
    script_dir = Path(__file__).parent
    flux_train_script = script_dir / ".." / "kohya_ss" / "sd-scripts" / "flux_train_network.py"
    run([
        "accelerate", "launch",
        "--num_machines", "1",
        "--num_processes", "1",
        "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        str(flux_train_script),
        "--config_file", str(resolved_path),
    ])


if __name__ == "__main__":
    main()

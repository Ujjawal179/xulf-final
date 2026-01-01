#!/usr/bin/env bash
set -euo pipefail

export FLUX_DIR="${{inputs.flux_weights}}"
export YOLO_DIR="${{inputs.yolo_weights}}"
export OUT_DIR="${{outputs.ModelSaves}}"

DATASET_URL="https://media.looktara.com/tmp/Akanksha.zip"
CONCEPT_SUBFOLDER="1_ohwx_woman"   # e.g. 1_ohwx_woman OR 1_ohwx_man
ASPECTS="1024x1024"                 # e.g. 1024x1024
RESOS="1024x1024"                     # e.g. 1024x1024

echo "FLUX_DIR=$FLUX_DIR"
echo "YOLO_DIR=$YOLO_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "DATASET_URL=$DATASET_URL"
echo "CONCEPT_SUBFOLDER=$CONCEPT_SUBFOLDER"

echo "=== Sanity: list Flux weights ==="
ls -lah "$FLUX_DIR" || true
echo "Find flux1-dev.safetensors:"
find "$FLUX_DIR" -maxdepth 4 -name "flux1-dev.safetensors" -print || true
echo "Find t5xxl_fp16.safetensors:"
find "$FLUX_DIR" -maxdepth 6 -name "t5xxl_fp16.safetensors" -print || true

echo "=== Workdir ==="
WORKDIR="$(pwd)"
mkdir -p "$WORKDIR/work"
cd "$WORKDIR/work"

echo "=== Clone codebase ==="
git clone --depth 1 https://github.com/sudhin05/xulf-final xulf-final
cd xulf-final

echo "=== Locate app_no_gradio.py ==="
APP="$(find . -maxdepth 6 -name "app_no_gradio.py" -print -quit || true)"
if [[ -z "$APP" ]]; then
  echo "ERROR: app_no_gradio.py not found in repo"
  exit 2
fi
echo "APP=$APP"

echo "=== Download + unzip dataset ==="
mkdir -p data/raw
python - <<PY
import os, urllib.request, zipfile
from pathlib import Path

url = os.environ.get("DATASET_URL")
zip_path = Path("data/raw/dataset.zip")

print("Downloading:", url)
urllib.request.urlretrieve(url, zip_path)

extract_dir = Path("data/raw/extracted")
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("Extracted to:", extract_dir)
PY

echo "=== Find first folder that contains images ==="
RAW_IMG_DIR="$(python - <<'PY'
from pathlib import Path
root = Path("data/raw/extracted")
img_ext = {".png",".jpg",".jpeg",".webp"}
best = None
for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in img_ext]
    if imgs:
        best = d
        break
print(best if best else root)
PY
)"
echo "RAW_IMG_DIR=$RAW_IMG_DIR"
ls -lah "$RAW_IMG_DIR" | head -n 50 || true

echo "=== Preprocess: crop ==="
mkdir -p data/crops
python "$APP" crop \
  --input  "$RAW_IMG_DIR" \
  --output "data/crops" \
  --aspect-ratios "$ASPECTS" \
  --batch-size 4 \
  --gpu-ids "0" \
  --class person \
  --model-dir "$YOLO_DIR" \
  --overwrite

echo "=== Preprocess: resize ==="
mkdir -p data/resized
python "$APP" resize \
  --input  "data/crops" \
  --output "data/resized" \
  --resolutions "$RESOS" \
  --threads 8 \
  --model-dir "$YOLO_DIR" \
  --overwrite

echo "=== Build Kohya dataset structure ==="
KOHYA_PARENT="$WORKDIR/kohya_dataset/img"
CONCEPT_DIR="$KOHYA_PARENT/$CONCEPT_SUBFOLDER"
mkdir -p "$CONCEPT_DIR"

export TRAIN_DATA_FOR_TOML="$KOHYA_PARENT"

python - <<'PY'
import os, shutil
from pathlib import Path

resized = Path("data/resized")
raw = Path(os.environ["RAW_IMG_DIR"])
concept_dir = Path(os.environ["CONCEPT_DIR"])

img_ext = {".png", ".jpg", ".jpeg", ".webp"}

linked = 0
for img in resized.rglob("*"):
    if img.is_file() and img.suffix.lower() in img_ext:
        dst_img = concept_dir / img.name
        if not dst_img.exists():
            try:
                os.link(img, dst_img)
            except Exception:
                shutil.copy2(img, dst_img)
        linked += 1

print("Prepared images:", linked)
print("Concept dir:", concept_dir)
PY


echo "TRAIN_DATA_FOR_TOML=$TRAIN_DATA_FOR_TOML"
echo "Total images in concept dir:"
find "$CONCEPT_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l

echo "=== Resolve runConfig.toml placeholders (from job code folder, NOT repo) ==="

RUNCFG="$CODE_ROOT/runConfig.toml"
if [[ ! -f "$RUNCFG" ]]; then
  echo "ERROR: $RUNCFG not found. Put runConfig.toml next to run_all.sh in job code."
  ls -lah "$CODE_ROOT" || true
  exit 3
fi

export RUNCFG
export CODE_ROOT

python - << "PY"
import os, re
from pathlib import Path

flux = os.environ["FLUX_DIR"]
train_parent = os.environ["TRAIN_DATA_FOR_TOML"]
out = os.environ["OUT_DIR"]

tok_flux  = "${" + "{inputs.flux_weights}" + "}"
tok_train = "${" + "{inputs.train_data}" + "}"
tok_out   = "${" + "{outputs.ModelSaves}" + "}"

src_path = Path(os.environ["RUNCFG"])
src = src_path.read_text()

resolved = (
    src.replace(tok_flux, flux)
       .replace(tok_train, train_parent)   
       .replace(tok_out, out)
)

out_path = Path(os.environ["CODE_ROOT"]) / "runConfig.resolved.toml"
out_path.write_text(resolved)

print("Wrote", out_path)
print("Resolved pretrained model should be:", str(Path(flux) / "flux1-dev.safetensors"))
print("Resolved train_data_dir should be:", train_parent)
PY

echo "=== RUN TRAINING ==="
accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  /opt/kohya_ss/sd-scripts/flux_train_network.py \
  --config_file runConfig.resolved.toml

echo "=== DONE: outputs ==="
ls -lah "$OUT_DIR" || true
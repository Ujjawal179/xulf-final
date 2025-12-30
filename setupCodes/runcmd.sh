bash -lc '
set -euo pipefail
shopt -s nullglob

export FLUX_DIR="${{inputs.flux_weights}}"
export TRAIN_DIR="${{inputs.train_data}}"
export OUT_DIR="${{outputs.ModelSaves}}"

echo "FLUX_DIR=$FLUX_DIR"
echo "TRAIN_DIR=$TRAIN_DIR"
echo "OUT_DIR=$OUT_DIR"

echo "Listing FLUX_DIR:"
ls -lah "$FLUX_DIR" || true
echo "Find flux1-dev.safetensors:"
find "$FLUX_DIR" -maxdepth 3 -name "flux1-dev.safetensors" -print || true

# -----------------------------
# 1) Build Kohya-friendly dataset structure (writable)
#    Kohya wants: train_data_dir = parent_of_concept_folders
#    Example: <parent>/img/1_ohwx_man/*.png
# -----------------------------
KOHYA_DATASET_ROOT="$PWD/kohya_dataset"
KOHYA_IMG_PARENT="$KOHYA_DATASET_ROOT/img"
CONCEPT_DIR="1_ohwx_man"   # <repeats>_<instance_token>

mkdir -p "$KOHYA_IMG_PARENT/$CONCEPT_DIR"

echo "=== Inspect TRAIN_DIR (first 50 files) ==="
find "$TRAIN_DIR" -maxdepth 2 -type f | head -n 50 || true

echo "=== Link images/captions into Kohya structure ==="
# If your images are directly under TRAIN_DIR, link them.
# If they are inside subfolders already, you can instead link that folder.
# This simplest approach links files from TRAIN_DIR root only:
linked=0
for f in "$TRAIN_DIR"/*; do
  case "${f,,}" in
    *.png|*.jpg|*.jpeg|*.webp|*.bmp|*.gif|*.tif|*.tiff|*.txt)
      ln -sf "$f" "$KOHYA_IMG_PARENT/$CONCEPT_DIR/$(basename "$f")"
      linked=$((linked+1))
      ;;
  esac
done

echo "Linked files: $linked"
echo "Total images in concept dir:"
find "$KOHYA_IMG_PARENT/$CONCEPT_DIR" -maxdepth 1 \( -type f -o -type l \) \
  \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l

# This is what must go into TOML as train_data_dir:
export TRAIN_DATA_FOR_TOML="$KOHYA_IMG_PARENT"
echo "TRAIN_DATA_FOR_TOML=$TRAIN_DATA_FOR_TOML"

# Optional: hard fail early if no images
img_count="$(find "$KOHYA_IMG_PARENT/$CONCEPT_DIR" -maxdepth 1 \( -type f -o -type l \) \
  \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)"
if [ "$img_count" -eq 0 ]; then
  echo "ERROR: No images found for training. Check TRAIN_DIR contents."
  exit 2
fi

# -----------------------------
# 2) Render TOML: replace tokens with actual resolved paths
#    IMPORTANT: inputs.train_data should become TRAIN_DATA_FOR_TOML (not TRAIN_DIR)
# -----------------------------
python - << "PY"
import os
from pathlib import Path

flux = os.environ["FLUX_DIR"]
train_parent = os.environ["TRAIN_DATA_FOR_TOML"]
out = os.environ["OUT_DIR"]

tok_flux  = "${" + "{inputs.flux_weights}" + "}"
tok_train = "${" + "{inputs.train_data}" + "}"
tok_out   = "${" + "{outputs.ModelSaves}" + "}"

src = Path("runConfig.toml").read_text()

resolved = (
    src.replace(tok_flux, flux)
       .replace(tok_train, train_parent)   # <-- critical change
       .replace(tok_out, out)
)

Path("runConfig.resolved.toml").write_text(resolved)

print("Wrote runConfig.resolved.toml")
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
'
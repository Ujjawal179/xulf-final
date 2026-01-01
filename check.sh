echo "=== Make weights available where preprocessing expects them ==="

# We’ll create a local folder in the job working dir where the preprocessing code can find weights
YOLO_LOCAL="$PWD/yolo_weights"
mkdir -p "$YOLO_LOCAL"

python - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["YOLO_DIR"])          # AzureML mounted yolo weights input
dst = Path(os.environ["YOLO_LOCAL"])        # Local folder inside job working dir
dst.mkdir(parents=True, exist_ok=True)

linked = 0
copied = 0

# Link everything from src -> dst (fallback to copy if linking fails)
for p in src.rglob("*"):
    if p.is_dir():
        continue

    rel = p.relative_to(src)
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if out.exists():
            out.unlink()
        out.symlink_to(p)
        linked += 1
    except Exception:
        # If symlink is not allowed, do a copy
        import shutil
        shutil.copy2(p, out)
        copied += 1

print(f"YOLO weights: linked={linked}, copied={copied}")
print("YOLO_LOCAL =", dst)
print("Sample files:")
for i, f in enumerate(dst.rglob("*")):
    if f.is_file():
        print(" -", f)
        if i >= 10:
            break
PY

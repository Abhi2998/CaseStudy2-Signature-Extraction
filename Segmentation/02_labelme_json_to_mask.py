import os, json, glob
from PIL import Image, ImageDraw

ROOT   = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\seg_dataset"
ANN_DIR = os.path.join(ROOT, "annotations")
MSK_DIR = os.path.join(ROOT, "masks")
os.makedirs(MSK_DIR, exist_ok=True)

ALLOWED = {"signature"}  # label(s) to keep

def convert_one(json_path, out_mask_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w = int(data["imageWidth"])
    h = int(data["imageHeight"])

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shp in data.get("shapes", []):
        label = (shp.get("label") or "").strip().lower()
        if label not in ALLOWED:
            continue

        pts = shp.get("points", [])
        if len(pts) < 3:
            continue

        pts = [(float(x), float(y)) for x, y in pts]
        draw.polygon(pts, outline=255, fill=255)

    mask.save(out_mask_path)

json_files = glob.glob(os.path.join(ANN_DIR, "*.json"))
if not json_files:
    raise SystemExit(f"No .json files found in {ANN_DIR}")

for jp in json_files:
    base = os.path.splitext(os.path.basename(jp))[0]
    outp = os.path.join(MSK_DIR, base + ".png")
    convert_one(jp, outp)

print("Done. Masks written to:", MSK_DIR)

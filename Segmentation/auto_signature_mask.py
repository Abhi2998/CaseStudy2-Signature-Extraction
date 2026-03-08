import os, glob
import cv2
import numpy as np
from tqdm import tqdm

SRC_ROOT = r"C:\Users\Dell\Desktop\Case Study 2\data_gen_signet"
OUT_ROOT = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks"

OUT_MASKS  = os.path.join(OUT_ROOT, "masks")
OUT_CUTOUT = os.path.join(OUT_ROOT, "cutouts")
os.makedirs(OUT_MASKS, exist_ok=True)
os.makedirs(OUT_CUTOUT, exist_ok=True)

def sanitize_name(path):
    rel = os.path.relpath(path, SRC_ROOT)
    return os.path.splitext(rel.replace("\\", "__").replace("/", "__"))[0]

def make_mask(bgr):
    h, w = bgr.shape[:2]

    # 1) grayscale -> denoise
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2) binarize (try Otsu; fallback to adaptive if needed)
    
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Adaptive threshold is more robust to uneven lighting/shadows. [web:710]
    th_adpt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Combine: keep pixels detected by either method
    th = cv2.bitwise_or(th_otsu, th_adpt)

    # 3) morphology cleanup: remove specks, connect strokes
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k3, iterations=2)

    # 4) connected components: filter by size & shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Heuristics (tune if needed)
    min_area = max(30, (h * w) // 20000)   # very small noise removed
    max_area = (h * w) // 3                # avoid selecting huge regions
    min_w, min_h = 15, 10

    for i in range(1, num):  # skip background
        x, y, ww, hh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if ww < min_w or hh < min_h:
            continue
        # signatures tend to be wider than tall (not always, but often)
        if ww / max(1, hh) < 1.2:
            continue
        mask[labels == i] = 255

    # Final smoothing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    return mask

def cutout_rgba(bgr, mask):
    # put signature pixels on transparent background
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask  # alpha = mask
    return bgra

img_paths = glob.glob(os.path.join(SRC_ROOT, "**", "*.jpg"), recursive=True)

# If 4.2 lakh images, start with a limit to test quickly
LIMIT = 5000   # set None to run all (but it will take long)
if LIMIT is not None:
    img_paths = img_paths[:LIMIT]

for p in tqdm(img_paths, desc="Masking"):
    bgr = cv2.imread(p)
    if bgr is None:
        continue

    name = sanitize_name(p)
    mask = make_mask(bgr)
    out_mask = os.path.join(OUT_MASKS, name + "_mask.png")
    out_cut  = os.path.join(OUT_CUTOUT, name + "_cutout.png")

    cv2.imwrite(out_mask, mask)
    cv2.imwrite(out_cut, cutout_rgba(bgr, mask))

print("Done.")
print("Masks:", OUT_MASKS)
print("Cutouts:", OUT_CUTOUT)

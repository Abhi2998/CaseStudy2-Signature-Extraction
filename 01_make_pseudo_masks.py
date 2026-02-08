import os
import glob
from tqdm import tqdm
import cv2
import numpy as np

# CHANGE THIS to your dataset root
DATA_ROOT = r"C:\Users\Dell\Desktop\Case Study 2\data_gen_signet"
OUT_ROOT  = r"C:\Users\Dell\Desktop\Case Study 2\seg_dataset"

IMG_OUT = os.path.join(OUT_ROOT, "images")
MSK_OUT = os.path.join(OUT_ROOT, "masks")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MSK_OUT, exist_ok=True)

def make_mask(bgr):
    # grayscale + denoise
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # signature often darker than background -> inverse binary
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove small noise, connect strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # keep as 0/255 mask
    mask = (th > 0).astype(np.uint8) * 255
    return mask

img_paths = glob.glob(os.path.join(DATA_ROOT, "**", "*.jpg"), recursive=True)

for p in tqdm(img_paths, desc="Creating pseudo masks"):
    bgr = cv2.imread(p)
    if bgr is None:
        continue

    mask = make_mask(bgr)

    # create safe filename that stays unique
    rel = os.path.relpath(p, DATA_ROOT).replace("\\", "__").replace("/", "__")
    base = os.path.splitext(rel)[0]

    out_img = os.path.join(IMG_OUT, base + ".jpg")
    out_msk = os.path.join(MSK_OUT, base + ".png")

    cv2.imwrite(out_img, bgr)
    cv2.imwrite(out_msk, mask)

print("Done.")
print("Images:", IMG_OUT)
print("Masks :", MSK_OUT)

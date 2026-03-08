import cv2
import numpy as np
import os
import glob
import shutil
from tqdm import tqdm

def fix_unet_dataset(img_dir, msk_dir, out_img_dir, out_msk_dir):
    """
    Scans the separate U-Net dataset folders.
    - Hollows out >10% density blobs using Morphological Gradients.
    - Purges <0.1% empty black masks.
    - Keeps images and masks perfectly synced in the new output folders.
    """
    # Create new clean directories
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    # Grab all files and sort them to ensure they match up exactly
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    msk_paths = sorted(glob.glob(os.path.join(msk_dir, "*.png")))

    if len(img_paths) != len(msk_paths):
        print("⚠️ Warning: Number of images and masks do not match! Check your folders.")

    fixed_count = 0
    clean_count = 0
    removed_count = 0

    # The 3x3 kernel used to extract the boundary
    kernel = np.ones((3, 3), np.uint8)

    print(f"🛠️ Scanning {len(msk_paths)} U-Net mask pairs...")

    for img_path, msk_path in tqdm(zip(img_paths, msk_paths), total=len(msk_paths), desc="Processing"):
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        total_pixels = h * w
        white_pixels = np.sum(mask > 128)
        density = white_pixels / total_pixels

        filename = os.path.basename(msk_path)
        out_msk_path = os.path.join(out_msk_dir, filename)
        out_img_path = os.path.join(out_img_dir, os.path.basename(img_path))

        if density > 0.10: 
            # 🚨 Corrupted Blob Detected: Apply Morphological Gradient to hollow it out
            fixed_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            cv2.imwrite(out_msk_path, fixed_mask)
            shutil.copy(img_path, out_img_path) # Copy matching image
            fixed_count += 1
            
        elif density < 0.001: 
            # 🚨 Empty Black Mask Detected: Skip it completely
            removed_count += 1
            pass 
            
        else: 
            # ✅ Clean Mask: Just copy both files over untouched
            cv2.imwrite(out_msk_path, mask)
            shutil.copy(img_path, out_img_path)
            clean_count += 1

    print("\n✅ U-Net Dataset Recovery Complete!")
    print(f"  -> Hollowed out and fixed: {fixed_count} masks")
    print(f"  -> Clean masks retained: {clean_count} masks")
    print(f"  -> Empty/Blank masks purged: {removed_count} masks")
    print(f"  -> Total usable pairs ready for training: {fixed_count + clean_count}")
    print(f"\n🚀 Your fixed dataset is ready in:\n Images: {out_img_dir}\n Masks: {out_msk_dir}")

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Update these paths to match your desktop setup!
    # ---------------------------------------------------------
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Clean_Ground_Truth"
    
    # New folders where the synced, perfect data will go
    OUT_IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Fixed_Cutouts"
    OUT_MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Fixed_Ground_Truth"
    
    fix_unet_dataset(IMG_DIR, MSK_DIR, OUT_IMG_DIR, OUT_MSK_DIR)
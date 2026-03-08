import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def clean_ground_truth(img_dir, mask_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)
    
    img_paths = sorted(glob.glob(f"{img_dir}/*.png")) + sorted(glob.glob(f"{img_dir}/*.jpg"))
    mask_paths = sorted(glob.glob(f"{mask_dir}/*.png")) + sorted(glob.glob(f"{mask_dir}/*.jpg"))
    
    print(f"Found {len(img_paths)} images to clean...")
    
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        filename = os.path.basename(mask_path)
        
        # 1. Load original image and flawed mask
        img = cv2.imread(img_path)
        flawed_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or flawed_mask is None:
            continue
            
        # 2. Resize to match our training dimensions
        img = cv2.resize(img, (256, 256))
        clean_mask = cv2.resize(flawed_mask, (256, 256))
        
        # 3. Use HSV to find all RED stamps in the original image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 4. ERASE the red stamp pixels from the Ground Truth Mask!
        clean_mask[red_pixels > 0] = 0
        
        # 5. ERASE tiny printed text (Area < 120 pixels) from Ground Truth Mask
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(clean_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 120:  # Keep only big signature strokes
                cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
        # 6. Save the perfectly cleaned Ground Truth Mask
        save_path = os.path.join(output_mask_dir, filename)
        cv2.imwrite(save_path, final_mask)

if __name__ == "__main__":
    # --- IMPORTANT: UPDATE THESE THREE PATHS ---
    
    # 1. Path to your newly extracted cutouts
    IMGS = r"C:\Users\Dell\Desktop\Case Study 2\data_gen_signet\cutouts" 
    
    # 2. Path to the newly extracted FLAWED masks
    OLD_MASKS = r"C:\Users\Dell\Desktop\Case Study 2\data_gen_signet\masks" 
    
    # 3. Where you want the PERFECT masks to be saved
    NEW_MASKS = r"C:\Users\Dell\Desktop\Case Study 2\Cleaned_Ground_Truth"
    
    clean_ground_truth(IMGS, OLD_MASKS, NEW_MASKS)
    print("\n✅ Dataset Ground Truth successfully cleaned and updated!")
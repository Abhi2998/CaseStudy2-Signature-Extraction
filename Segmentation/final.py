import torch
import torch.nn as nn
import cv2
import numpy as np
import glob
import os

# ==========================================
# 1. The UNet Architecture
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512) 
        self.dropout = nn.Dropout2d(0.5)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        e1 = self.enc1(x)                                 
        e2 = self.enc2(nn.MaxPool2d(2)(e1))               
        e3 = self.enc3(nn.MaxPool2d(2)(e2))               
        e4 = self.enc4(nn.MaxPool2d(2)(e3))               
        e4 = self.dropout(e4)
        d1 = self.up1(e4)                                 
        d1 = torch.cat([d1, e3], dim=1)                   
        d1 = self.dec1(d1)                                
        d2 = self.up2(d1)                                 
        d2 = torch.cat([d2, e2], dim=1)                   
        d2 = self.dec2(d2)                                
        d3 = self.up3(d2)                                 
        d3 = torch.cat([d3, e1], dim=1)                   
        d3 = self.dec3(d3)                                
        return self.final(d3)

# ==========================================
# 2. AI + OpenCV Post-Processing
# ==========================================
def extract_perfect_signatures(model_path, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading AI Brain...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    
    # Process the first 20 images to see the magic
    for path in img_paths[:20]:
        filename = os.path.basename(path)
        
        # 1. Prepare Image
        original_img = cv2.imread(path)
        if original_img is None:
            continue
            
        original_img = cv2.resize(original_img, (256, 256))
        inp_tensor = torch.FloatTensor(original_img.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        # 2. AI Predicts (Grabs all ink, including stamps)
        with torch.no_grad():
            logits = model(inp_tensor)
            ai_mask = (torch.sigmoid(logits).cpu().numpy()[0][0] > 0.5).astype(np.uint8)
            
        # ---------------------------------------------------------
        # 3. NEW MAGIC: HSV Color Filtering (The Red Eraser)
        # ---------------------------------------------------------
        # Convert image to HSV color space to easily isolate red
        hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        
        # Red loops around the hue spectrum in OpenCV, so we need two ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Find all the red pixels
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Tell the AI Mask to FORGET any pixel that is red
        ai_mask[red_pixels > 0] = 0
        
        # ---------------------------------------------------------
        # 4. Size Filtering (To clean up tiny noise/printed text)
        # ---------------------------------------------------------
        contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(ai_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # If the blob is larger than 120 pixels, keep it (it's a signature stroke)
            # If it's smaller than 120 pixels, ignore it (it's leftover printed text)
            if area > 120:  
                cv2.drawContours(filtered_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        
        # ---------------------------------------------------------
        # 5. Create Final Clean Image (White paper, Black ink)
        # ---------------------------------------------------------
        final_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        final_image[filtered_mask == 1] = [0, 0, 0] 
        
        # Save it
        save_path = os.path.join(output_dir, f"perfect_{filename}")
        cv2.imwrite(save_path, final_image)
        print(f"✅ Filtered and saved: {save_path}")

if __name__ == "__main__":
    # Point this to your BEST model (Likely your focal loss one)
    MODEL = "best_signature_unet_focal.pth" 
    IMGS = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    OUTPUT = "perfect_signatures_output"
    
    extract_perfect_signatures(MODEL, IMGS, OUTPUT)
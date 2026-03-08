import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os


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


if __name__ == '__main__':
    
    MODEL_WEIGHTS = "best_signature_unet_focal.pth"
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Clean_Ground_Truth"
    SIZE = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 Loading U-Net weights from {MODEL_WEIGHTS} on {device}...")
    
    
    model = UNet().to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"❌ Could not find {MODEL_WEIGHTS}. Did the training script finish?")
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    
   
    all_imgs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    all_masks = sorted(glob.glob(f"{MSK_DIR}/*.png"))
    
    if not all_imgs:
        raise ValueError("❌ No images found in the specified IMG_DIR.")
        
    
    idx = random.randint(0, len(all_imgs) - 1)
    img_path = all_imgs[idx]
    mask_path = all_masks[idx]
    
    print(f"🖼️ Testing on: {os.path.basename(img_path)}")
    
    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) # For plotting later
    img_resized = cv2.resize(img_raw, (SIZE, SIZE)) / 255.0
    
    target_mask = cv2.imread(mask_path, 0)
    target_mask = cv2.resize(target_mask, (SIZE, SIZE))
    
    # Convert to tensor: shape (1, 3, H, W)
    input_tensor = torch.FloatTensor(img_resized.transpose(2,0,1)).unsqueeze(0).to(device)
    
    # --- Predict ---
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            raw_pred = model(input_tensor)
            # Apply sigmoid to convert logits to probabilities (0 to 1)
            prob_mask = torch.sigmoid(raw_pred)
            # Binary threshold at 0.5
            binary_mask = (prob_mask > 0.5).float()
            
    
    out_mask_np = binary_mask.squeeze().cpu().numpy()
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.resize(img_rgb, (SIZE, SIZE)))
    axes[0].set_title("1. Original Input")
    axes[0].axis("off")
    
    axes[1].imshow(out_mask_np, cmap='gray')
    axes[1].set_title("2. U-Net Predicted Mask")
    axes[1].axis("off")
    
    axes[2].imshow(target_mask, cmap='gray')
    axes[2].set_title("3. Ground Truth")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
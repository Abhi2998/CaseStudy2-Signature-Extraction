import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2, os, glob
import numpy as np
from tqdm import tqdm

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder upsampling layers
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Decoder conv blocks (after concatenation)
        self.dec1 = self.conv_block(256, 128)  # 128(up) + 128(skip) = 256
        self.dec2 = self.conv_block(128, 64)   # 64(up) + 64(skip) = 128
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder - save skip connections
        e1 = self.enc1(x)           # [B,64,256,256]
        e2 = self.enc2(nn.MaxPool2d(2)(e1))  # [B,128,128,128]
        e3 = self.enc3(nn.MaxPool2d(2)(e2))  # [B,256,64,64]
        
        # Decoder path
        d1 = self.up1(e3)           # [B,128,128,128]
        d1 = torch.cat([d1, e2], dim=1)  # [B,256,128,128]
        d1 = self.dec1(d1)          # [B,128,128,128]
        
        d2 = self.up2(d1)           # [B,64,256,256]
        d2 = torch.cat([d2, e1], dim=1)  # [B,128,256,256]
        d2 = self.dec2(d2)          # [B,64,256,256]
        
        return torch.sigmoid(self.final(d2))

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.imgs = sorted(glob.glob(f"{img_dir}/*.png"))
        self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.size = size
        print(f"Found {len(self.imgs)} images, {len(self.masks)} masks")
        assert len(self.imgs) == len(self.masks)
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        mask = cv2.imread(self.masks[idx], 0)
        
        img = cv2.resize(img, (self.size, self.size)) / 255.0
        mask = cv2.resize(mask, (self.size, self.size)) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        return torch.FloatTensor(img.transpose(2,0,1)), torch.FloatTensor(mask).unsqueeze(0)

if __name__ == '__main__':
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\masks"
    
    device = torch.device('cpu')
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    
    dataset = SegDataset(IMG_DIR, MSK_DIR)
    loader = DataLoader(dataset, 4, shuffle=True, num_workers=0, pin_memory=False)
    
    print(f"Training on {device}")
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = bce(pred, msks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} COMPLETE - Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"signature_unet_e{epoch+1:02d}.pth")
    
    print("✅ Training complete! Check signature_unet_e*.pth files")

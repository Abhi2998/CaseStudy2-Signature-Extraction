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
        
        # Decoder conv blocks 
        self.dec1 = self.conv_block(256, 128)  
        self.dec2 = self.conv_block(128, 64)   
        
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
        e1 = self.enc1(x)           
        e2 = self.enc2(nn.MaxPool2d(2)(e1))  
        e3 = self.enc3(nn.MaxPool2d(2)(e2))  
        
        # Decoder path
        d1 = self.up1(e3)           
        d1 = torch.cat([d1, e2], dim=1)  
        d1 = self.dec1(d1)          
    
        d2 = self.up2(d1)           
        d2 = torch.cat([d2, e1], dim=1)  
        d2 = self.dec2(d2)          
        
        return self.final(d2)

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
    
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # 1. Calculate Stable BCE 
        bce_loss = self.bce(inputs, targets)
        
        # 2. Apply Sigmoid purely for the Dice calculation
        inputs_soft = torch.sigmoid(inputs)
        
        # 3. Flatten label and prediction tensors for Dice
        inputs_flat = inputs_soft.view(-1)
        targets_flat = targets.view(-1)
        
        # 4. Calculate Dice Loss
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # 5. Combine them
        total_loss = (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)
        
        return total_loss
    
if __name__ == '__main__':
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\masks"
    
    # Force GPU usage if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss(bce_weight=0.5)
    
    
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = SegDataset(IMG_DIR, MSK_DIR)
    
    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training on {train_size} images, Validating on {val_size} images.")
    
    # Track the best validation metric
    best_val_iou = 0.0
    print(f"Training on {device}")
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, msks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} COMPLETE - Avg Loss: {avg_loss:.4f}")
        # ================= VALIDATION PHASE =================
        model.eval() 
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad(): 
            for val_imgs, val_msks in val_loader:
                val_imgs, val_msks = val_imgs.to(device), val_msks.to(device)
                
                with torch.amp.autocast('cuda'):
                    val_pred = model(val_imgs)
                    v_loss = criterion(val_pred, val_msks)
                    
                val_loss += v_loss.item()
                
                
                pred_binary = (val_pred > 0.5).float()
                intersection = (pred_binary * val_msks).sum()
                union = pred_binary.sum() + val_msks.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6) 
                val_iou += iou.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch+1} VAL - Loss: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f}")
        
        # Only save the model if it actually got better on unseen data!
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "best_signature_unet.pth")
            print(f"   --> Saved new BEST model! (IoU: {best_val_iou:.4f})")
    
    print("✅ Training complete! Check signature_unet_e*.pth files")

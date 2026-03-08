import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2, os, glob
import numpy as np
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure


import albumentations as A
from albumentations.pytorch import ToTensorV2


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
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


class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        # We now pass LISTS of paths instead of directories
        self.imgs = img_paths
        self.masks = mask_paths
        self.transform = transform
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1. Load Image and Mask
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Albumentations requires RGB
        mask = cv2.imread(self.masks[idx], 0)
        
        # 2. Apply Transforms (Augmentation)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # 3. Ensure mask is float and has channel dimension [1, H, W]
        mask = (mask > 0.5).float().unsqueeze(0)
        
        return img, mask


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_soft = torch.sigmoid(inputs)
        inputs_flat = inputs_soft.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)


if __name__ == '__main__':
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\masks"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
   
    # Training: Hard augmentations to force learning
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),           
        A.VerticalFlip(p=0.1),             
        A.Rotate(limit=20, p=0.5),         # Rotate +/- 20 degrees
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3), # Random lighting
        A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
        ToTensorV2(),
    ])
    
    # Validation: No augmentation, just resize
    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
        ToTensorV2(),
    ])
    
    # --- PREPARE DATA ---
    all_imgs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    all_masks = sorted(glob.glob(f"{MSK_DIR}/*.png"))
    print(f"Total found: {len(all_imgs)} images")
    
    # Manual Split to ensure Train gets Augmentation and Val gets None
    indices = np.arange(len(all_imgs))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(all_imgs))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Create distinct datasets
    train_dataset = SegDataset(
        [all_imgs[i] for i in train_idx], 
        [all_masks[i] for i in train_idx], 
        transform=train_transform
    )
    
    val_dataset = SegDataset(
        [all_imgs[i] for i in val_idx], 
        [all_masks[i] for i in val_idx], 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training on {device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Initialize Model components
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    best_val_iou = 0.0
    
    
    for epoch in range(15): 
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(imgs)
                loss = criterion(pred, msks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_ssim = 0
        
        with torch.no_grad():
            for val_imgs, val_msks in val_loader:
                val_imgs, val_msks = val_imgs.to(device), val_msks.to(device)
                
                with torch.amp.autocast('cuda'):
                    val_pred = model(val_imgs)
                    v_loss = criterion(val_pred, val_msks)
                
                val_loss += v_loss.item()
                pred_binary = (val_pred > 0.0).float()
                intersection = (pred_binary * val_msks).sum()
                union = pred_binary.sum() + val_msks.sum() - intersection
                val_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
                val_ssim += ssim_metric(torch.sigmoid(val_pred), val_msks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val IoU: {avg_val_iou:.4f} | SSIM: {avg_val_ssim:.4f}")
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "best_signature_unet_aug.pth")
            print(f"   --> 💾 Saved new BEST model! (IoU: {best_val_iou:.4f})")
            
    print("Training complete! Check best_signature_unet_aug.pth")
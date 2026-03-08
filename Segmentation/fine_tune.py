import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2, os, glob
import numpy as np
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure


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
        d1 = torch.cat([self.up1(e4), e3], dim=1)                   
        d1 = self.dec1(d1)                                
        d2 = torch.cat([self.up2(d1), e2], dim=1)                   
        d2 = self.dec2(d2)                                
        d3 = torch.cat([self.up3(d2), e1], dim=1)                   
        d3 = self.dec3(d3)                                
        return self.final(d3)


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        # Recursive search to ensure no files are missed
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True) + 
                           glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "**/*.png"), recursive=True))
        self.size = size
        print(f"Found {len(self.imgs)} images, {len(self.masks)} masks")
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        mask = cv2.imread(self.masks[idx], 0)
        img = cv2.resize(img, (self.size, self.size)) / 255.0
        mask = cv2.resize(mask, (self.size, self.size)) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        return torch.FloatTensor(img.transpose(2,0,1)), torch.FloatTensor(mask).unsqueeze(0)


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.gamma, self.alpha, self.dice_weight, self.smooth = gamma, alpha, dice_weight, smooth

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal = focal.mean() 
        
        inputs_soft = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_soft * targets_flat).sum()                            
        dice = 1 - (2. * intersection + self.smooth) / (inputs_soft.sum() + targets_flat.sum() + self.smooth)
        return ((1 - self.dice_weight) * focal) + (self.dice_weight * dice)


if __name__ == '__main__':
    
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Clean_Ground_Truth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalDiceLoss()
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    dataset = SegDataset(IMG_DIR, MSK_DIR)
    train_split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_split, len(dataset)-train_split])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, pin_memory=True)
    
    best_iou = 0.0

    # PHASE 1: Main Training (20 Epochs)
    for epoch in range(20):
        model.train()
        pbar = tqdm(train_loader, desc=f"Phase 1 - Ep {epoch+1}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        v_iou, v_ssim = 0, 0
        with torch.no_grad():
            for vi, vm in val_loader:
                vi, vm = vi.to(device), vm.to(device)
                vp = model(vi)
                pb = (vp > 0.0).float()
                inter = (pb * vm).sum()
                union = pb.sum() + vm.sum() - inter
                v_iou += ((inter + 1e-6) / (union + 1e-6)).item()
                v_ssim += ssim_metric(torch.sigmoid(vp), vm).item()
        
        avg_iou, avg_ssim = v_iou/len(val_loader), v_ssim/len(val_loader)
        print(f"Epoch {epoch+1} | IoU: {avg_iou:.4f} | SSIM: {avg_ssim:.4f}")
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_signature_unet_focal.pth")

    # PHASE 2: Fine-Tuning (5 Epochs)
    print("\n🌟 Starting Phase 2: Fine-Tuning...")
    model.load_state_dict(torch.load("best_signature_unet_focal.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Sharpening

    for epoch in range(5):
        model.train()
        pbar = tqdm(train_loader, desc=f"Phase 2 - Ep {epoch+1}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
       
    torch.save(model.state_dict(), "ultimate_signature_unet.pth")
    print("✅ Training complete. Ultimate model saved!")
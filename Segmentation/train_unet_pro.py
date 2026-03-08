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
        super(UNet, self).__init__()
        
        # Encoder Blocks
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck Dropout
        self.dropout = nn.Dropout2d(0.5)
        
        # Decoder Upsampling
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Decoder Convolutional Blocks
        self.dec1 = self._conv_block(512, 256)
        self.dec2 = self._conv_block(256, 128)
        self.dec3 = self._conv_block(128, 64)
        
        # Final Output Layer
        self.final = nn.Conv2d(64, 1, 1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)                                 
        e2 = self.enc2(F.max_pool2d(e1, 2))               
        e3 = self.enc3(F.max_pool2d(e2, 2))               
        e4 = self.enc4(F.max_pool2d(e3, 2))               
        
        # Bottleneck
        e4 = self.dropout(e4)
        
        # Decoder Path with Skip Connections
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
    def __init__(self, img_dir, mask_dir, size=256):
        # Search recursively for images and masks
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True) + 
                           glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "**/*.png"), recursive=True))
        self.size = size
        print(f"DEBUG: Found {len(self.imgs)} images and {len(self.masks)} masks.")
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        mask = cv2.imread(self.masks[idx], 0)
        
        # Resize and Normalize
        img = cv2.resize(img, (self.size, self.size)) / 255.0
        mask = cv2.resize(mask, (self.size, self.size)) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to Tensors
        return torch.FloatTensor(img.transpose(2,0,1)), torch.FloatTensor(mask).unsqueeze(0)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Standard Binary Cross Entropy
        bce_loss = self.bce_logits(inputs, targets)
        
        # Dice Loss for Overlap Maximization
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # Combined Weighted Loss
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)


if __name__ == '__main__':
   
    IMG_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\cutouts"
    MSK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\Clean_Ground_Truth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss(bce_weight=0.2) # Weighting BCE lower to focus on Dice
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Prepare Data
    dataset = SegDataset(IMG_DIR, MSK_DIR)
    train_split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_split, len(dataset)-train_split])
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, pin_memory=True)
    
    best_iou = 0.0

    #  Main Training (20 Epochs) 
    print(f"🌟 Starting Phase 1: General Learning...")
    for epoch in range(20):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/20")
        for imgs, msks in train_pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, msks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

     
        model.eval()
        v_iou, v_ssim = 0, 0
        with torch.no_grad():
            for vi, vm in val_loader:
                vi, vm = vi.to(device), vm.to(device)
                vp = model(vi)
                
                # IoU Metric Calculation
                pred_bin = (vp > 0.0).float()
                inter = (pred_bin * vm).sum()
                union = pred_bin.sum() + vm.sum() - inter
                v_iou += ((inter + 1e-6) / (union + 1e-6)).item()
                
                # SSIM Metric Calculation
                v_ssim += ssim_metric(torch.sigmoid(vp), vm).item()
        
        avg_iou = v_iou / len(val_loader)
        avg_ssim = v_ssim / len(val_loader)
        print(f"📊 Validation Ep {epoch+1} | IoU: {avg_iou:.4f} | SSIM: {avg_ssim:.4f}")
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_signature_unet_pro.pth")
            print(f"💾 Saved BEST Weights (IoU: {best_iou:.4f})")

   #Fine-Tuning Polish (5 Epochs) ---
    print("\n🌟 Starting Phase 2: Fine-Tuning for Stroke Integrity...")
    model.load_state_dict(torch.load("best_signature_unet_pro.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Lower Learning Rate

    for epoch in range(5):
        model.train()
        ft_pbar = tqdm(train_loader, desc=f"Fine-Tune {epoch+1}/5")
        for imgs, msks in ft_pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ft_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    torch.save(model.state_dict(), "ultimate_signature_unet_pro.pth")
    print("✅ Pro training sequence complete! Metrics ready for comparison.")
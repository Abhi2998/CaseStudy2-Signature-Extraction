import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class IndustrialCGANDataset(Dataset):
    def __init__(self, root_dir, cache_file="image_paths.pkl", subset_size=43200):
        self.root_dir = Path(root_dir)
        self.image_paths = []
        
        # 1. Faster Indexing with Cache
        if os.path.exists(cache_file):
            print(f"📂 Loading cached paths from {cache_file}...")
            with open(cache_file, "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            print(f"🔍 Indexing folders... (Wait for it)")
            for i in range(1, 4001):
                folder_path = self.root_dir / f"{i:04d}"
                if folder_path.is_dir():
                    for ext in ('*.png', '*.jpg', '*.jpeg'):
                        self.image_paths.extend(list(folder_path.glob(ext)))
            
            if len(self.image_paths) == 0:
                raise ValueError(f"❌ No images found! Check path: {self.root_dir}")
                
            with open(cache_file, "wb") as f:
                pickle.dump(self.image_paths, f)
        
        # 2. Sub-Sampling for Fast Iteration
        if len(self.image_paths) > subset_size:
            print(f"🎲 Shuffling and sub-sampling down to {subset_size} images...")
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:subset_size] 

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print(f"✅ Ready! {len(self.image_paths)} images loaded.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = img.size
        # Split stitched pair: Left = Cluttered Input, Right = Clean Target
        input_img = self.transform(img.crop((0, 0, w // 2, h)))
        target_img = self.transform(img.crop((w // 2, 0, w, h)))
        return input_img, target_img


def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'InstanceNorm' in classname or 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

class GBlock(nn.Module):
    def __init__(self, in_f, out_f, down=True, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_f, out_f, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_f),
            nn.ReLU() if not down else nn.LeakyReLU(0.2)
        )
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv(x))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.d2 = GBlock(64, 128); self.d3 = GBlock(128, 256); self.d4 = GBlock(256, 512)
        self.d5 = GBlock(512, 512); self.d6 = GBlock(512, 512); self.d7 = GBlock(512, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.u1 = GBlock(512, 512, down=False, use_dropout=True)
        self.u2 = GBlock(1024, 512, down=False, use_dropout=True)
        self.u3 = GBlock(1024, 512, down=False, use_dropout=True)
        self.u4 = GBlock(1024, 512, down=False)
        self.u5 = GBlock(1024, 256, down=False); self.u6 = GBlock(512, 128, down=False)
        self.u7 = GBlock(256, 64, down=False)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        s1 = self.d1(x); s2 = self.d2(s1); s3 = self.d3(s2); s4 = self.d4(s3)
        s5 = self.d5(s4); s6 = self.d6(s5); s7 = self.d7(s6)
        bn = self.bottleneck(s7)
        u1 = self.u1(bn); u2 = self.u2(torch.cat([u1, s7], 1))
        u3 = self.u3(torch.cat([u2, s6], 1)); u4 = self.u4(torch.cat([u3, s5], 1))
        u5 = self.u5(torch.cat([u4, s4], 1)); u6 = self.u6(torch.cat([u5, s3], 1))
        u7 = self.u7(torch.cat([u6, s2], 1))
        return self.final(torch.cat([u7, s1], 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def d_layer(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.LeakyReLU(0.2)
            )
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            d_layer(64, 128), d_layer(128, 256),
            d_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


if __name__ == "__main__":
    # --- Config ---
    DATA_ROOT = "extracted_dataset" 
    BATCH_SIZE = 64
    EPOCHS = 10
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = IndustrialCGANDataset(DATA_ROOT)
    
    # ULTIMATE FIX: num_workers=0 completely bypasses the /dev/shm container limit
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    gen = nn.DataParallel(Generator()).to(device).apply(init_weights)
    disc = nn.DataParallel(Discriminator()).to(device).apply(init_weights)

    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()
    
    # --- Deprecation Warning Fix ---
    scaler = torch.amp.GradScaler('cuda')

    # --- Loop ---
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            # Discriminator Step
            with torch.amp.autocast('cuda'):
                fake_y = gen(x)
                d_real = disc(x, y)
                d_fake = disc(x, fake_y.detach())
                loss_d = (BCE(d_real, torch.ones_like(d_real)) + BCE(d_fake, torch.zeros_like(d_fake))) / 2
            
            opt_disc.zero_grad(); scaler.scale(loss_d).backward(); scaler.step(opt_disc)

            # Generator Step
            with torch.amp.autocast('cuda'):
                g_fake = disc(x, fake_y)
                # Adversarial + heavily weighted L1 loss (100x)
                loss_g = BCE(g_fake, torch.ones_like(g_fake)) + (L1(fake_y, y) * 100)
            
            opt_gen.zero_grad(); scaler.scale(loss_g).backward(); scaler.step(opt_gen)
            scaler.update()

            loop.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())

        torch.save(gen.module.state_dict(), f"industrial_cgan_epoch_{epoch+1}.pth")

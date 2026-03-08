import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm


class GANStitchedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        cluttered = img.crop((0, 0, w // 2, h))
        clean = img.crop((w // 2, 0, w, h))
        return self.transform(cluttered), self.transform(clean)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size)) # FIXED: InstanceNorm
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size), # FIXED: InstanceNorm
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128), # FIXED: InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256), # FIXED: InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1)
            # FIXED: Removed Sigmoid for LSGAN stability
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

DATA_DIR = "gan_stitched" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Ultimate Pix2Pix (LSGAN) on: {device}")

gen = nn.DataParallel(GeneratorUNet()).to(device)
disc = nn.DataParallel(Discriminator()).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# FIXED: Using MSELoss (Least Squares GAN) instead of BCELoss
criterion_GAN = nn.MSELoss() 
criterion_pixelwise = nn.L1Loss() 

train_ds = GANStitchedDataset(DATA_DIR)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)

for epoch in range(50):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        
        # 1. Train Discriminator
        fake_y = gen(x)
        d_real = disc(x, y)
        d_fake = disc(x, fake_y.detach())
        
        loss_d_real = criterion_GAN(d_real, torch.ones_like(d_real))
        loss_d_fake = criterion_GAN(d_fake, torch.zeros_like(d_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
        
        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # 2. Train Generator
        g_fake = disc(x, fake_y)
        loss_g_gan = criterion_GAN(g_fake, torch.ones_like(g_fake))
        loss_g_pixel = criterion_pixelwise(fake_y, y) * 100 
        loss_g = loss_g_gan + loss_g_pixel
        
        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()
        
        loop.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())
        
    if (epoch + 1) % 10 == 0:
        torch.save(gen.module.state_dict(), f"ultimate_pix2pix_epoch_{epoch+1}.pth")
        
print("✅ Ultimate Pix2Pix Training Complete!")

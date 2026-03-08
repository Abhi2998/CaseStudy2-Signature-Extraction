import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load the Ultimate LSGAN Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ultimate_gen = GeneratorUNet().to(device) # Using the deep U-Net class

# Load the new LSGAN weights!
ultimate_gen.load_state_dict(torch.load("ultimate_pix2pix_epoch_50.pth", map_location=device))
ultimate_gen.eval()

# 2. Image Processing Pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def test_ultimate_gan(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = ultimate_gen(x)
        
    # Un-normalize the output (from [-1, 1] to [0, 1])
    generated = generated.squeeze(0).cpu()
    generated = generated * 0.5 + 0.5 
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Cluttered Input")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Ultimate LSGAN Cleaned Output")
    plt.imshow(generated.permute(1, 2, 0).clamp(0, 1)) 
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- RUN THE TEST ---
# Point to a messy cutout
TEST_IMAGE = "cutouts/0039__cf-039-51_cutout.png" 
test_ultimate_gan(TEST_IMAGE)

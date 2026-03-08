import cv2, os, glob
import matplotlib.pyplot as plt
from PIL import Image

MASK_DIR = r"C:\Users\Dell\Desktop\Case Study 2\Segmentation\auto_masks\masks"
files = glob.glob(os.path.join(MASK_DIR, "*.png"))[:12]

fig, axs = plt.subplots(3, 4, figsize=(15, 12))
for i, f in enumerate(files):
    row, col = divmod(i, 4)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    axs[row, col].imshow(img, cmap='gray')
    axs[row, col].set_title(os.path.basename(f))
    axs[row, col].axis('off')
plt.tight_layout()
plt.savefig("mask_preview.png", dpi=150, bbox_inches='tight')
plt.show()

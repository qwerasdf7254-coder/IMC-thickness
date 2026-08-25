import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar
from imc_tool.segment import local_std_map

path = sys.argv[1]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 3)
h, w = gray.shape

k = max(7, (min(h, w) // 100) | 1)
std_map = local_std_map(gray_dn, k=k)

K = max(21, (min(h, w) // 30) | 1)
density = cv2.blur(std_map, (K, K))

# row-wise mean profile to see the 3-band structure
row_profile = density.mean(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('gray')
im = axes[1].imshow(density, cmap='viridis')
axes[1].set_title(f'texture density (blur std, K={K})')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[2].plot(row_profile, np.arange(h))
axes[2].invert_yaxis()
axes[2].set_title('row-mean density profile')
plt.tight_layout()
out = path.split('\\')[-1] + ".density.png"
outpath = r"C:\Users\82109\IMC_Thickness_Tool\scratch\\" + out
plt.savefig(outpath, dpi=100)
print("saved", outpath)

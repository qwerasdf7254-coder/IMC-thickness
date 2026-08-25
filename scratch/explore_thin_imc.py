import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar
from imc_tool.segment import local_std_map, segment_interface

path = sys.argv[1]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 3)
h, w = gray.shape

seg = segment_interface(gray)
cu_top = seg["cu_top"]
print("cu_top range:", np.min(cu_top), np.max(cu_top), "mean", np.mean(cu_top))

# zoom into a band from well above to well below cu_top, centered per-column would
# be complex; just take a fixed crop around the median cu_top for visual inspection
cy = int(np.median(cu_top))
lo = max(0, cy - 250)
hi = min(h, cy + 100)
crop = gray[lo:hi, :]
crop_big = cv2.resize(crop, (crop.shape[1], crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\thin_imc_crop.png", crop_big)
print(f"saved crop rows [{lo},{hi}) -> thin_imc_crop.png (2x vertical stretch)")

fine = local_std_map(gray_dn, k=9)
coarse = local_std_map(gray_dn, k=25)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axes[0].imshow(gray[lo:hi, :], cmap='gray')
axes[0].set_title('gray (crop)')
axes[1].imshow(fine[lo:hi, :], cmap='viridis')
axes[1].set_title('fine std k=9')
axes[2].imshow(coarse[lo:hi, :], cmap='viridis')
axes[2].set_title('coarse std k=25')
for ax in axes:
    ax.plot(np.arange(w), cu_top - lo, color='red', linewidth=0.8)
plt.tight_layout()
plt.savefig(r"C:\Users\82109\IMC_Thickness_Tool\scratch\thin_imc_scales.png", dpi=100)
print("saved thin_imc_scales.png")

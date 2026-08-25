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
std_u8 = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

band_top = int(0.25 * h)  # search a generous band, wider than before
std_band = std_u8.copy()
std_band[:band_top, :] = 0

thresh, rough_u8 = cv2.threshold(std_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
rough_u8 = cv2.morphologyEx(rough_u8, cv2.MORPH_OPEN, kernel, iterations=1)
rough_u8 = cv2.morphologyEx(rough_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

n, labels, stats, _ = cv2.connectedComponentsWithStats(rough_u8, connectivity=8)
# pick the component with the largest width-coverage (most columns touched)
best_label, best_cov = 0, 0
for lbl in range(1, n):
    cols = np.unique(np.where(labels == lbl)[1])
    cov = len(cols) / w
    if cov > best_cov:
        best_cov = cov
        best_label = lbl
print("best component coverage:", best_cov, "thresh:", thresh)

imc_mask = labels == best_label
cu_top = np.full(w, np.nan)   # bottom edge of IMC band = Cu/IMC boundary
solder_bottom = np.full(w, np.nan)  # top edge of IMC band = solder/IMC boundary
for x in range(w):
    ys = np.where(imc_mask[:, x])[0]
    if ys.size:
        solder_bottom[x] = ys.min()
        cu_top[x] = ys.max()

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
fill = vis.copy()
fill[imc_mask] = (0, 140, 255)
vis = cv2.addWeighted(vis, 0.6, fill, 0.4, 0)
for x in range(w):
    if not np.isnan(cu_top[x]):
        cv2.circle(vis, (x, int(cu_top[x])), 1, (0, 255, 255), -1)
    if not np.isnan(solder_bottom[x]):
        cv2.circle(vis, (x, int(solder_bottom[x])), 1, (0, 0, 255), -1)

cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\band_result.png", vis)
print("saved band_result.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(std_band, cmap='viridis')
axes[0].set_title(f'std (band-limited) thresh={thresh}')
axes[1].imshow(rough_u8, cmap='gray')
axes[1].set_title(f'rough mask, chosen comp cov={best_cov:.2f}')
plt.tight_layout()
plt.savefig(r"C:\Users\82109\IMC_Thickness_Tool\scratch\band_debug.png", dpi=100)
print("saved band_debug.png")

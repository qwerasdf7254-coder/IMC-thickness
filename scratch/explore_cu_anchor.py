import cv2
import numpy as np
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
thresh, rough_u8 = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
smooth_u8 = 255 - rough_u8

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
smooth_open = cv2.morphologyEx(smooth_u8, cv2.MORPH_OPEN, kernel3, iterations=1)

n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth_open, connectivity=8)
touching = set(np.unique(labels[h - 1, :])) - {0}
cu_mask = np.isin(labels, list(touching)) if touching else np.zeros_like(smooth_open, dtype=bool)
coverage = float(np.mean(cu_mask.any(axis=0)))
print("cu bottom-connected coverage:", coverage, "thresh:", thresh)

cu_top = np.full(w, np.nan)
for x in range(w):
    ys = np.where(cu_mask[:, x])[0]
    if ys.size:
        cu_top[x] = ys.min()

# now find rough components touching the cu region from above (dilate cu mask by a couple px)
cu_u8 = (cu_mask * 255).astype(np.uint8)
cu_dilated = cv2.dilate(cu_u8, kernel3, iterations=2)

rough_open = cv2.morphologyEx(rough_u8, cv2.MORPH_OPEN, kernel3, iterations=1)
nr, rlabels, rstats, _ = cv2.connectedComponentsWithStats(rough_open, connectivity=8)
touch_labels = set(np.unique(rlabels[(cu_dilated > 0) & (rough_open > 0)])) - {0}
imc_mask = np.isin(rlabels, list(touch_labels)) if touch_labels else np.zeros_like(rough_open, dtype=bool)
imc_cov = float(np.mean(imc_mask.any(axis=0)))
print("imc (touching cu) coverage:", imc_cov)

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
fill = vis.copy()
fill[imc_mask] = (0, 140, 255)
fill[cu_mask] = (0, 255, 0)
vis = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\cu_anchor_result.png", vis)
print("saved cu_anchor_result.png")

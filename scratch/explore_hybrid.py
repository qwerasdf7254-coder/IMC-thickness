import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar
from imc_tool.segment import multi_otsu_2thresh


def local_std_map(gray, k=9):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sq_mean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sq_mean - mean * mean, 0, None)
    return np.sqrt(var)


def keep_touch(mask_u8, row):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    touching = set(np.unique(labels[row, :])) - {0}
    return np.isin(labels, list(touching)) if touching else np.zeros_like(mask_u8, dtype=bool)


path = sys.argv[1]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 3)
h, w = gray.shape

t1, t2 = multi_otsu_2thresh(gray_dn.ravel())
not_brightest = gray_dn <= t2  # exclude only the definitively-brightest class

k = max(7, (min(h, w) // 100) | 1)
std_map = local_std_map(gray_dn, k=k)
std_u8 = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
smooth_thresh, smooth_bin = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
smooth = smooth_bin > 0

cu_candidate = (not_brightest & smooth).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cu_open = cv2.morphologyEx(cu_candidate, cv2.MORPH_OPEN, kernel, iterations=1)
cu_mask = keep_touch(cu_open, h - 1)
coverage = float(np.mean(cu_mask.any(axis=0)))
print("thresholds:", t1, t2, "cu coverage:", coverage)

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
fill = vis.copy()
fill[cu_mask] = (0, 255, 0)
vis = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\hybrid_result.png", vis)
print("saved hybrid_result.png")

import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar
from imc_tool.segment import local_std_map, dp_ridge_path

path = sys.argv[1]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 3)
h, w = gray.shape

k = max(7, (min(h, w) // 100) | 1)
std_map = local_std_map(gray_dn, k=k)

band_top = int(0.35 * h)
std_band = std_map.copy()
std_band[:band_top, :] = -1e9
seed, scores = dp_ridge_path(std_band, max_jump=4, smooth_lambda=3.0)
print("seed range:", seed.min(), seed.max())

thresh = np.percentile(std_map, 40)  # rough global "smooth" reference level
print("smooth threshold (40th pct):", thresh)

run_len = 6
max_extent = int(0.35 * h)

cu_top = np.full(w, np.nan)     # bottom edge of rough band
solder_top = np.full(w, np.nan)  # top edge of rough band

for x in range(w):
    y0 = int(seed[x])
    col = std_map[:, x]
    # scan down
    y = y0
    while y < min(h - 1, y0 + max_extent):
        if np.all(col[y:min(h, y + run_len)] < thresh):
            break
        y += 1
    cu_top[x] = y
    # scan up
    y = y0
    while y > max(0, y0 - max_extent):
        if np.all(col[max(0, y - run_len):y] < thresh):
            break
        y -= 1
    solder_top[x] = y

# median filter first (robust to narrow spike outliers from stray dark veins
# reaching up into solder), then a light box smooth for visual continuity
from scipy.signal import medfilt

def smooth(a, mk=21, bk=7):
    a = medfilt(a, kernel_size=mk)
    kernel = np.ones(bk) / bk
    pad = bk // 2
    ap = np.pad(a, pad, mode='edge')
    return np.convolve(ap, kernel, mode='valid')

cu_top_s = smooth(cu_top)
solder_top_s = smooth(solder_top)

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for x in range(w):
    cv2.circle(vis, (x, int(cu_top_s[x])), 1, (0, 255, 255), -1)
    cv2.circle(vis, (x, int(solder_top_s[x])), 1, (0, 0, 255), -1)
    cv2.circle(vis, (x, int(seed[x])), 1, (255, 0, 255), -1)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\seed_expand_result.png", vis)
print("saved seed_expand_result.png")
print("mean IMC thickness (px):", np.mean(cu_top_s - solder_top_s))

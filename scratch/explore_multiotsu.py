import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar


def multi_otsu_2thresh(values, bins=256):
    hist, _ = np.histogram(values, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    idx = np.arange(bins)
    # cumulative sums for fast between-class variance eval
    best = (-1, 0, 0)
    csum = np.cumsum(hist)
    cmean = np.cumsum(hist * idx)
    global_mean = cmean[-1] / total
    for t1 in range(1, bins - 2):
        w0 = csum[t1]
        if w0 == 0:
            continue
        m0 = cmean[t1] / w0
        for t2 in range(t1 + 1, bins - 1):
            w1 = csum[t2] - csum[t1]
            if w1 == 0:
                continue
            w2 = total - csum[t2]
            if w2 == 0:
                break
            m1 = (cmean[t2] - cmean[t1]) / w1
            m2 = (cmean[-1] - cmean[t2]) / w2
            var_between = w0 * (m0 - global_mean) ** 2 + w1 * (m1 - global_mean) ** 2 + w2 * (m2 - global_mean) ** 2
            if var_between > best[0]:
                best = (var_between, t1, t2)
    return best[1], best[2]


path = sys.argv[1]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 3)
h, w = gray.shape

t1, t2 = multi_otsu_2thresh(gray_dn.ravel())
print("thresholds:", t1, t2)

dark = gray_dn <= t1
mid = (gray_dn > t1) & (gray_dn <= t2)
bright = gray_dn > t2

# bottom-connected filter on 'dark' class to find true Cu vs artifact black
dark_u8 = (dark * 255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dark_open = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, kernel, iterations=1)
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_open, connectivity=8)
touching = set(np.unique(labels[h - 1, :])) - {0}
true_cu = np.isin(labels, list(touching)) if touching else np.zeros_like(dark_open, dtype=bool)
artifact_black = dark & ~true_cu

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
fill = vis.copy()
fill[true_cu] = (0, 255, 0)       # green = true Cu
fill[mid] = (0, 140, 255)          # orange = IMC candidate
fill[bright] = (255, 80, 0)        # blue = solder (Sn/Bi bright)
fill[artifact_black] = (255, 80, 0)  # reassign artifact black -> solder too
vis = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\multiotsu_result.png", vis)
print("saved multiotsu_result.png")
print("class coverage: cu=%.3f imc_candidate=%.3f solder(bright+artifact)=%.3f" % (
    true_cu.mean(), mid.mean(), (bright.mean() + artifact_black.mean())))

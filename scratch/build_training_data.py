import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode, imwrite_unicode
from imc_tool.barcrop import crop_bar

LABEL_DIR = Path(r"C:\Users\82109\IMC_Thickness_Tool\라벨링")
ROOT = Path(r"E:\00_정명진\03_연구실 컴퓨터\03_SBSAC_Reliability")
OUT_DIR = Path(r"C:\Users\82109\IMC_Thickness_Tool\training_data")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)
(OUT_DIR / "masks").mkdir(exist_ok=True)
(OUT_DIR / "debug").mkdir(exist_ok=True)

NAMES = [
    "0_0_14", "0_0_20", "0_200_17", "0_500_17", "0_500_27", "0_1000_11",
    "50_0_22", "50_200_4", "50_200_13", "50_500_18", "50_500_26", "50_1000_11",
    "80_0_13", "80_0_19", "80_200_9", "80_200_17", "80_500_16", "80_1000_4", "80_1000_9",
    "100_0_17", "100_200_4", "100_200_8", "100_500_32", "100_1000_13",
]


def extract_two_curves(red_mask, w, h):
    """For each column, split red pixels into upper cluster (min-y group) and
    lower cluster (max-y group) -- the two annotated boundary lines rarely
    cross, so a simple gap-based split per column works."""
    upper = np.full(w, np.nan)
    lower = np.full(w, np.nan)
    for x in range(w):
        ys = np.where(red_mask[:, x])[0]
        if ys.size == 0:
            continue
        if ys.size == 1:
            upper[x] = lower[x] = ys[0]
            continue
        # split into contiguous-ish groups by biggest gap
        gaps = np.diff(ys)
        if gaps.size and gaps.max() > 3:
            split = np.argmax(gaps)
            g1 = ys[:split + 1]
            g2 = ys[split + 1:]
            upper[x] = g1.mean()
            lower[x] = g2.mean()
        else:
            upper[x] = lower[x] = ys.mean()
    return upper, lower


def fill_nan(a, w):
    idx = np.arange(w)
    valid = ~np.isnan(a)
    if not valid.any():
        return None
    return np.interp(idx, idx[valid], a[valid])


results = []
for i, name in enumerate(NAMES, 1):
    lab_path = LABEL_DIR / f"그림{i}.png"
    candidates = list(ROOT.rglob(f"{name}.jpg"))
    if not candidates:
        print(f"[{i}] {name}: ORIGINAL NOT FOUND, skip")
        continue
    orig_path = candidates[0]

    lab = imread_unicode(str(lab_path), cv2.IMREAD_COLOR)
    orig_full = imread_unicode(str(orig_path), cv2.IMREAD_COLOR)
    oh, ow = orig_full.shape[:2]

    lab_resized = cv2.resize(lab, (ow, oh), interpolation=cv2.INTER_LINEAR)
    b, g, r = lab_resized[:, :, 0].astype(int), lab_resized[:, :, 1].astype(int), lab_resized[:, :, 2].astype(int)
    red_mask = (r > 150) & (r - g > 60) & (r - b > 60)

    upper, lower = extract_two_curves(red_mask, ow, oh)
    coverage = np.mean(~np.isnan(upper))
    upper_f = fill_nan(upper, ow)
    lower_f = fill_nan(lower, ow)
    if upper_f is None:
        print(f"[{i}] {name}: NO RED FOUND, skip")
        continue

    gray_full = cv2.cvtColor(orig_full, cv2.COLOR_BGR2GRAY)
    gray, bar_top = crop_bar(gray_full)
    h, w = gray.shape

    mask = np.zeros((h, w), dtype=np.uint8)  # 0=solder,1=IMC,2=Cu
    for x in range(w):
        u = int(round(upper_f[x]))
        l = int(round(lower_f[x]))
        u = max(0, min(h, u))
        l = max(0, min(h, l))
        if u > l:
            u, l = l, u
        mask[:u, x] = 0
        mask[u:l, x] = 1
        mask[l:, x] = 2

    imwrite_unicode(str(OUT_DIR / "images" / f"{name}.png"), gray)
    imwrite_unicode(str(OUT_DIR / "masks" / f"{name}.png"), mask)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    overlay[mask == 1] = (0, 140, 255)
    overlay[mask == 2] = (0, 200, 0)
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    imwrite_unicode(str(OUT_DIR / "debug" / f"{name}_check.png"), vis)

    thick = np.mean(mask == 1)
    print(f"[{i}] {name}: ok, curve coverage={coverage:.2f}, IMC area frac={thick:.3f}, src={orig_path}")
    results.append(name)

print(f"\nDone: {len(results)}/{len(NAMES)} processed")

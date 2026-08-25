import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode, imwrite_unicode

crops = []
for i in range(1, 25):
    path = rf"C:\Users\82109\IMC_Thickness_Tool\라벨링\그림{i}.png"
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # bottom-left filename label region
    crop = img[int(h*0.94):h, 0:int(w*0.25)]
    crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2))
    label = np.zeros((crop.shape[0], 150, 3), dtype=np.uint8)
    cv2.putText(label, f"#{i}", (5, crop.shape[0]//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    combo = np.hstack([label, crop])
    crops.append(combo)

maxw = max(c.shape[1] for c in crops)
padded = []
for c in crops:
    if c.shape[1] < maxw:
        pad = np.zeros((c.shape[0], maxw - c.shape[1], 3), dtype=np.uint8)
        c = np.hstack([c, pad])
    padded.append(c)

full = np.vstack(padded)
imwrite_unicode(r"C:\Users\82109\IMC_Thickness_Tool\scratch\filename_montage.png", full)
print("saved, shape:", full.shape)

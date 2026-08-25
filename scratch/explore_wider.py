import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar

path = sys.argv[1]
lo, hi, x0, x1 = [int(a) for a in sys.argv[2:6]]
img = imread_unicode(path)
gray, bar_top = crop_bar(img)
crop = gray[lo:hi, x0:x1]
crop_big = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), interpolation=cv2.INTER_NEAREST)
out = r"C:\Users\82109\IMC_Thickness_Tool\scratch\wider_crop.png"
cv2.imwrite(out, crop_big)
print("saved", out, "orig shape", crop.shape)

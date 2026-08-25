import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode
from imc_tool.barcrop import crop_bar
from imc_tool.segment import multi_otsu_2thresh

img = imread_unicode(r"E:\00_정명진\03_연구실 컴퓨터\03_SBSAC_Reliability\IMC 측정\main\plus\0_200_11.jpg")
gray, _ = crop_bar(img)
gray_dn = cv2.medianBlur(gray, 15)
t1, t2 = multi_otsu_2thresh(gray_dn.ravel())
print("t1,t2:", t1, t2)

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
dark = gray_dn <= t1
mid = (gray_dn > t1) & (gray_dn <= t2)
bright = gray_dn > t2
fill = vis.copy()
fill[dark] = (0,255,0)
fill[mid] = (0,140,255)
fill[bright] = (255,80,0)
vis2 = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)
cv2.imwrite(r"C:\Users\82109\IMC_Thickness_Tool\scratch\0200_11_classes.png", vis2)
print("saved")

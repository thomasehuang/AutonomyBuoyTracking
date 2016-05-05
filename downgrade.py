import numpy as np
import cv2
from matplotlib import pyplot as plt

img1_orig = cv2.imread('/home/anthuang/Documents/AutonomyBuoyTracking/buoy_7.jpg', 1)
img1_trans = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
img1_res = cv2.resize(img1_trans, (1000, 750))

cv2.imwrite('/home/anthuang/Documents/AutonomyBuoyTracking/BuoyImages/buoy_7.jpg', img1_res)

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('buoy1.jpg', 0)
img2 = cv2.imread('buoy2.jpg', 0)

threshold = 10

coords1 = (0, 0)
coords2 = (0, 0)

# Initiate STAR detector
orb = cv2.ORB()

# setting number of features
orb.setInt("nFeatures", 1000)
# setting edge threshold (should roughly equal patch size)
orb.setInt("edgeThreshold", 50)
# setting patch size
orb.setInt("patchSize", 50)

# find the keypoints and compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

kp1_del = []
kp2_del = []

for idx in range(len(kp1)):
    if not (kp1[idx].x > coords1[0] and kp1[idx].y > coords1[1] and kp1[idx].x < coords2[0] and kp1[idx].y < coords2[1]):
        kp1_del.append(kp1[idx])
        kp2_del.append(kp2[idx])
    else if abs(kp1[idx].y - kp2[idx].y) > threshold + (coords2.y - coords1.y):
        kp1_del.append(kp1[idx])
        kp2_del.append(kp2[idx])

kp1 = list(set(kp1).difference(set(kp1_del)))
kp2 = list(set(kp2).difference(set(kp2_del)))

maxx = maxy = 0
minx = miny = 99999
for idx in range(len(kp2)):
    if kp2[idx].x < minx:
        minx = kp2[idx].x
    else if kp2[idx].x > maxx:
        maxx = kp2[idx].x
    else if kp1[idx].y < miny:
        miny = kp2[idx].y
    else if kp1[idx].y > maxy:
        maxy = kp2[idx].y

cv2.rectangle(img2, (minx - threshold, miny - threshold), (maxx - minx + threshold, maxy - miny + threshold), (255,0,0), 2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# draw only keypoints location,not size and orientation
img3 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)

img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)

plt.imshow(img3),plt.show()
plt.imshow(img4),plt.show()

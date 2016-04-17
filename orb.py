import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 5
    thickness = 1
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        # if not color:
        #     c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        c = (0, 255, 0)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    plt.imshow(new_img)
    plt.show()

img1_orig = cv2.imread('/home/anthuang/Documents/AutonomyBuoyTracking/buoy1.jpg', 1)
img2_orig = cv2.imread('/home/anthuang/Documents/AutonomyBuoyTracking/buoy2.jpg', 1)
img1_trans = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
img2_trans = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB)
img1_res = cv2.resize(img1_trans, (1000, 750))
img2_res = cv2.resize(img2_trans, (1000, 750))

threshold = 10

coords1 = (477, 363)
coords2 = (550, 450)

cut_x1 = coords1[0] - threshold * 5
cut_x2 = coords2[0] + threshold * 5
cut_y1 = coords1[1] - threshold * 5
cut_y2 = coords2[1] + threshold * 5
img1 = img1_res[cut_y1:cut_y2, cut_x1:cut_x2]
img2 = img2_res[cut_y1:cut_y2, cut_x1:cut_x2]

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

# Initiate ORG detector
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

print(len(kp1), len(kp2))
print(len(des1), len(des2))

# img1_key = cv2.drawKeypoints(img1, kp1, color=(0,255,0), flags=0)
# img2_key = cv2.drawKeypoints(img1, kp1, color=(0,255,0), flags=0)
# plt.imshow(img1_key), plt.show()
# plt.imshow(img2_key), plt.show()

# Deleting keypoints that are out of the bounding box
# Unnecessary since we are cutting the image based on the bounding box
# kp1_del = []
# kp2_del = []
# for idx in range(len(kp1)):
#     if not (kp1[idx].pt[0] > coords1[0] and kp1[idx].pt[1] > coords1[1] and kp1[idx].pt[0] < coords2[0] and kp1[idx].pt[1] < coords2[1]):
#         kp1_del.append(kp1[idx])
#         kp2_del.append(kp2[idx])
#     elif abs(kp1[idx].pt[1] - kp2[idx].pt[1]) > threshold + (coords2[1] - coords1[1]):
#         kp1_del.append(kp1[idx])
#         kp2_del.append(kp2[idx])
# kp1 = list(set(kp1).difference(set(kp1_del)))
# kp2 = list(set(kp2).difference(set(kp2_del)))

maxx = maxy = 0
minx = miny = 99999
for idx in range(len(kp2)):
    if kp2[idx].pt[0] < minx:
        minx = kp2[idx].pt[0]
    elif kp2[idx].pt[0] > maxx:
        maxx = kp2[idx].pt[0]
    elif kp1[idx].pt[1] < miny:
        miny = kp2[idx].pt[1]
    elif kp1[idx].pt[1] > maxy:
        maxy = kp2[idx].pt[1]

cv2.rectangle(img2_res, (int(minx - threshold + cut_x1), int(miny - threshold + cut_y1)), (int(maxx + threshold + cut_x1), int(maxy + threshold + cut_y1)), (255,0,0), 2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

draw_matches(img1, kp1, img2, kp2, matches)
plt.imshow(img2_res), plt.show()

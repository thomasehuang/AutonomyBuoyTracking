import cv2
import numpy as np
from matplotlib import pyplot as plt

img_orig = cv2.imread('/home/anthuang/Documents/AutonomyBuoyTracking/buoy1.jpg',1)
img_trans = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(img_trans, (1000, 750))
img2 = img.copy()
img_orig3 = cv2.imread('/home/anthuang/Documents/AutonomyBuoyTracking/buoy2.jpg',1)
img_trans3 = cv2.cvtColor(img_orig3, cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img_trans3, (1000, 750))
img4 = img3.copy()

plt.imshow(img), plt.show()

threshold = 10
coords1 = (477, 363)
coords2 = (550, 450)
cut_x1 = coords1[0] - threshold * 5
cut_x2 = coords2[0] + threshold * 5
cut_y1 = coords1[1] - threshold * 5
cut_y2 = coords2[1] + threshold * 5
template = img[cut_y1:cut_y2, cut_x1:cut_x2]
z, w, h = template.shape[::-1]


# All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ['cv2.TM_SQDIFF_NORMED']

for meth in methods:
   img = img3.copy()
   method = eval(meth)

   # Apply template Matching
   res = cv2.matchTemplate(img,template,method)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

   # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
   if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
       top_left = min_loc
   else:
       top_left = max_loc
   bottom_right = (top_left[0] + w, top_left[1] + h)

   cv2.rectangle(img,top_left, bottom_right, 255, 2)

   plt.subplot(121),plt.imshow(res,cmap = 'gray')
   plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
   plt.subplot(122),plt.imshow(img,cmap = 'gray')
   plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
   plt.suptitle(meth)

   plt.show()

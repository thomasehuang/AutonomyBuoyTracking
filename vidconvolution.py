import numpy as np
import cv2
from matplotlib import pyplot as plt

# list of resize sizes
resize_pixels = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]

# coordinates is a list of lists containing a track id and the corners of each bounding box [track_id, top_left, bottom_right]
# track id isn't really implemented, since I don't know how it is used
coordinates = [[0, (485, 380), (540, 435)], [1, (664, 420), (727, 486)]]

for x in range(1, 8):
    # code to read in images from a directory
    # can be replaced to read in image from camera
    filestart = "BuoyImages/buoy_"
    ext = ".jpg"
    filename_1 = filestart + str(x) + ext
    filename_2 = filestart + str(x + 1) + ext

    # resize the image down since image from iPhone's camera resolution is too high
    # can be deleted when reading image from boat camera
    img_orig = cv2.imread(filename_1,1)
    img_trans = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_trans, (1000, 750))
    img_orig2 = cv2.imread(filename_2,1)
    img_trans2 = cv2.cvtColor(img_orig2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img_trans2, (1000, 750))

    # for each bounding box
    for coordinate in range(len(coordinates)):

        box_id = coordinates[coordinate][0]
        top_left = coordinates[coordinate][1]
        bottom_right = coordinates[coordinate][2]

        # create a template image consisting only the buoy
        threshold = 0
        cut_x1 = top_left[0] - threshold * 5
        cut_x2 = bottom_right[0] + threshold * 5
        cut_y1 = top_left[1] - threshold * 5
        cut_y2 = bottom_right[1] + threshold * 5
        template = img[cut_y1:cut_y2, cut_x1:cut_x2]
        z, w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        # methods = ['cv2.TM_SQDIFF_NORMED']
        # methods = ['cv2.TM_CCOEFF']
        # methods = ['cv2.TM_CCOEFF_NORMED']
        # methods = ['cv2.TM_SQDIFF']
        method = 'cv2.TM_CCORR_NORMED'

        conv_values = []
        conv_coords = []

        # check each resize pixels to find the best bounding box size
        # this is in case the buoy moves closer or further away
        # the list of resize pixels needs to be changed according to the boat's speed
        # this loop takes a lot of time to run in the worst case, so it should be optimized
        for resize_pixel in resize_pixels:

            template_copy = template.copy()

            template_copy = cv2.resize(template, (w + resize_pixel, h + resize_pixel))

            # Apply template Matching
            res = cv2.matchTemplate(img2, template_copy, eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
               top_left_match = min_loc
            else:
               top_left_match = max_loc
            bottom_right_match = (top_left_match[0] + w + resize_pixel, top_left_match[1] + h + resize_pixel)

            conv_values.append(max_val)
            conv_coords.append([top_left_match, bottom_right_match])

            # if max_val is over a certain threshold, break from loop (no need to check other sizes since good enough)
            # this threshold needs to be tested
            if max_val > 0.95:
                conv_values = [max_val]
                conv_coords = [[top_left_match, bottom_right_match]]
                break

        max_index = conv_values.index(max(conv_values))

        # draw the new bounding box
        cv2.rectangle(img2, conv_coords[max_index][0], conv_coords[max_index][1], 255, 2)

        # update the bounding box
        coordinates[coordinate] = [box_id, top_left_match, bottom_right_match]

    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)

    plt.imshow(img2), plt.show()
    plt.show()

    # print(coordinates)

cv2.destroyAllWindows()

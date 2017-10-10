# AutonomyBuoyTracking

Buoys are tracked using template matching with each buoy as template images and matching
them with camera images.

The buoy tracking algorithm takes in (1) the current and next camera images and (2) a list of
bounding boxes of buoys that were detected with the buoy detection algorithm on the current
image.

The algorithm loops through these bounding boxes. For each bounding box, the buoy is cut out
of the current image and used as a template image. The template image is matched against the
next image using OpenCV’s template matching algorithm. This algorithm basically matches the
template image against all possible sub-images of the same size in the next image and scores
each sub-image depending on how closely they match with a score of 0 (complete mismatch) to
1 (complete match). The sub-image with the highest score will be returned and used as the new
bounding box for the buoy.

To account for the buoy getting closer or further away, the algorithm also runs OpenCV’s
template matching algorithm on rescaled versions of the template images. The template images
are rescaled by a few pixels only, since the buoy will not move far between camera images. The
match with the highest template matching score among the rescaled versions will be returned
and used as the new bounding box for the buoy.

Running the algorithm against many template images is slow and inefficient. Thus, there is a
threshold value for the template matching score. If the matching score passes this threshold
value, that match will be used as the new bounding box. This threshold value is experimentally
determined.

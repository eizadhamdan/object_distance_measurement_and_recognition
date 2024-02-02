import cv2
from realsense_camera import *
from mask_rcnn import *

# Load Realsense camera and Mask R-CNN
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    point_x, point_y = 250, 100
    distance_mm = depth_frame[point_y, point_x]

    cv2.circle(bgr_frame, (point_x, point_y), 8, (0, 0, 255), -1)
    cv2.putText(bgr_frame, "{} mm".format(distance_mm), (point_x, point_y - 10), 0, 1, (0, 0, 255), 2)

    cv2.imshow("Depth Frame", depth_frame)
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# point_x, point_y = 250, 100
# distance_mm = depth_frame[point_x, point_y]
#
# cv2.circle(bgr_frame, (point_x, point_y), 8, (0,0,255), -1)
#
# # Get object mask
# boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)
#
# # Draw object mask
# bgr_frame = mrcnn.draw_object_mask(bgr_frame)
#
# # Show depth info of the objects
# mrcnn.draw_object_info(bgr_frame, depth_frame)

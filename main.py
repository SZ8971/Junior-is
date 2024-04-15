# pip install opencv-contrib-python # opencv-python contains the main packages wheras the other
                                    # contains both main modules and contrib/extra modules
# pip install cvlib # for object detection
# coming from https://www.youtube.com/watch?v=V62M9d8QkYM&t=51s
# I change the code for line 19 since if I did not change, the program can not work

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

video = cv2.VideoCapture(0)

while True:
    status, frame = video.read()

    # Bounding box.
    # the cvlib library has learned some basic objects using object learning
    # usually it takes around 800 images for it to learn what a phone is.
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

    output_image = draw_bbox(frame, bbox, label, conf, write_conf=True)

    cv2.imshow("Object Detection", output_image)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break  

video.release()
cv2.destroyAllWindows
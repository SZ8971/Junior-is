#pip install ultralytics
# comes from https://www.youtube.com/watch?v=hg4oVgNq7Do&t=358s
# I did not use the numpy and I changed the videocapture from picture to the live camera.

import cv2
from ultralytics import YOLO

video = cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")

if not video.isOpened():
    print("Cannot open camera")
    exit()

while True:
     # Capture frame-by-frame
    success, frame = video.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break  

    else:
        print("Can't receive frame. Exiting ...")
        break

video.release()
cv2.destroyAllWindows
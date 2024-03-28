import tkinter
import cv2
from ultralytics import YOLO

# Create a window with a width of 400 and a height of 500
root = tkinter.Tk()
root.title('Object Detection')
window_width = 400
window_height = 500
root.geometry(f'{window_width}x{window_height}')

# Using the Live Camera
def live():
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

# Upload the pictures or videos from the inference
def inf():
    picture = cv2.VideoCapture("inference")

#create two buttons for users to choose: Live camera, Pictures or Videos
# Create the reset button
label_result = tkinter.Label(root, text='Choose')
label_result.pack()
button_Live = tkinter.Button(root, text='Live Camera', command=live)
button_Live.pack()
button_Picture = tkinter.Button(root, text='Pictures or Videos', command=inf)
button_Picture.pack()

root.mainloop()
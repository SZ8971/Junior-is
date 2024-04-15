# The creating window part is typed by meself
# Selecting folder and file part is typed by myself
# Detection part is comes from https://www.youtube.com/watch?v=hg4oVgNq7Do&t=358s & https://www.youtube.com/watch?v=V62M9d8QkYM&t=51s
# The difference is I did not use the numpy from the first one and I changed the videocapture from picture to the live camera.
import tkinter
from tkinter import filedialog
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
            if cv2.waitKey(1) & 0xf == ord('q'):
                break
            
        else:
            print("Can't receive frame. Exiting ...")
            break

# Upload the pictures or videos from the inference
def inf():
    #Opens a dialog box for folder selection
    Folderpath = filedialog.askdirectory()
    #Opens a dialog box for file selection
    Filepath = filedialog.askopenfilename()
    #change the index to the file which selected
    picture = cv2.VideoCapture(Filepath)
    model=YOLO("yolov8n.pt")

    if not picture.isOpened():
        print("There is no picture/video")
        exit()

    while True:
         # Capture frame-by-frame
        success, frame = picture.read()

        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xf == ord('q'):
                break

        else:
            print("Can't receive frame. Exiting ...")
            break



#create two buttons for users to choose: Live camera, Pictures or Videos
# Create the reset button
label_result = tkinter.Label(root, text='Choose')
label_result.pack()
button_Live = tkinter.Button(root, text='Live Camera', command=live)
button_Live.pack()
button_Picture = tkinter.Button(root, text='Pictures or Videos', command=inf)
button_Picture.pack()

root.mainloop()
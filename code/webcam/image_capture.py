# things to change:
# lighting, angle, background, floor, camera angle

import os
import cv2

stream = 'http://10.50.72.107:8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cam = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)

path = "/Users/collinwentzien/Documents/Projects/ASR Bike/tensorflow/images/"
model = "cone"
count = 0

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to open video camera...")
        exit(1)
    cv2.imshow("snapshot", frame)

    k = cv2.waitKey(1)

    if k%256 == 27: # esc: quit
        print("Escape hit, closing...")
        break
    elif k%256 == 32: # space: take frame
        img_name = path + model + "_" + str(count) + ".png"
        cv2.imwrite(img_name, frame)
        print("{} wrote image to ".format(img_name))
        count += 1

cam.release()
cv2.destroyAllWindows()
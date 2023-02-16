import numpy as np
import cv2
import os

stream = 'http://10.0.0.42:8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('[ERROR] Cannot open RTSP stream (is there a password?)')
    exit(-1)

while True:
    success, img = cap.read()
    cv2.imshow('bike output', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
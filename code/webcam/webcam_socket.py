import numpy as np
import cv2
import os
import socket
from datetime import datetime
import time

ip = "10.50.72.111"

stream = 'http://10.50.72.111:8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, 8080))
print("[LOCAL] Connected to bike.")

if not cap.isOpened():
    print('[ERROR:LOCAL] Cannot open stream (is there a password?)')
    exit(-1)

while True:
    success, img = cap.read()
    cv2.imshow('bike output', img)

    # tf

    now = datetime.now()
    current_time = now.strftime("%M%S")
    print(current_time)
    msg = current_time + " " + str(4)
    print("[SENT] " + msg)
    client.send(msg.encode())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
client.close()
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2

imageHub = imagezmq.ImageHub()

print("[LOCAL] Waiting for video stream.")

while True:
    (name, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    lastActive = {}
    lastActiveCheck = datetime.now()

    # print("received")

    cv2.imshow('stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Keep running until you press `q`
        name.release()
        break

cv2.destroyAllWindows()
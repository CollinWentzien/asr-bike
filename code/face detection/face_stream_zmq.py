from mtcnn import MTCNN
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2

detector = MTCNN()

width = 320
bb = (0, 0)

imageHub = imagezmq.ImageHub()
print("[LOCAL] Waiting for video stream.")

while True:
   (name, frame) = imageHub.recv_image()
   frame = cv2.merge([frame,frame,frame])
   imageHub.send_reply(b'OK')

   detections = detector.detect_faces(frame)

   cv2.putText(frame, "Faces detected: " + str(len(detections)), (8, frame.shape[0] - 8), 0, 0.5, (255, 255, 255), 1)

   for detection in detections:
      score = detection["confidence"]
      # print("[CLIENT] Face detected with confidence " + str(score))
      if score >= 0.70:
         x, y, w, h = detection["box"]
         detected_face = frame[int(y):int(y+h), int(x):int(x+w)]

         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
         cv2.putText(frame, "Face", (x, y - 40), 0, 0.5, (255, 255, 255), 2)
         cv2.putText(frame, "Probability: " + str(100 * round(score, 5)) + "%", (x, y - 10), 0, 0.5, (255, 255, 255), 1)

   cv2.imshow('video', frame)

   key = cv2.waitKey(1)
   if key == 27:
      break

name.release()
cv2.destroyAllWindows()
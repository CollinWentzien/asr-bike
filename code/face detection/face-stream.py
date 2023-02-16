from mtcnn import MTCNN
from datetime import datetime
import numpy as np
import cv2
import os


stream = 'http://10.50.72.111:8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)

detector = MTCNN()

width = 320
bb = (0, 0)

print("[LOCAL] Loading stream at " + stream)

if not cap.isOpened():
    print('[ERROR] Cannot open RTSP stream (is there a password?)')
    exit(-1)

while True:
   success, frame = cap.read()
   frame = cv2.resize(frame, (480, 320))

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

cap.release()
cv2.destroyAllWindows()
from mtcnn import MTCNN
import cv2
import socket

ip = "10.0.0.42"
 
print("[CLIENT] TF Loading...")
detector = MTCNN()
cam = cv2.VideoCapture(0)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, 8080))
print("[BIKE] Connection established")

while True:
   check, frame = cam.read()

   detections = detector.detect_faces(frame)

   cv2.putText(frame, "Faces detected: " + str(len(detections)), (8, frame.shape[0] - 8), 0, 1.5, (255, 255, 255), 2)
   client.send((str(len(detections))).encode())

   for detection in detections:
      score = detection["confidence"]
      # print("[CLIENT] Face detected with confidence " + str(score))
      if score >= 0.70:
         x, y, w, h = detection["box"]
         detected_face = frame[int(y):int(y+h), int(x):int(x+w)]

         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
         cv2.putText(frame, "Face", (x, y - 40), 0, 1, (255, 255, 255), 2)
         cv2.putText(frame, "Probability: " + str(100 * round(score, 5)) + "%", (x, y - 10), 0, 1, (255, 255, 255), 2)

   cv2.imshow('video', frame)

   key = cv2.waitKey(1)
   if key == 27:
      break

cam.release()
cv2.destroyAllWindows()
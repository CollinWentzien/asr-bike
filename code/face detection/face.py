from mtcnn import MTCNN
import cv2
 
detector = MTCNN()

img = cv2.imread("img.png")
detections = detector.detect_faces(img)

print("TF Loading...")

cv2.putText(img, "Faces detected: " + str(len(detections)), (8, img.shape[0] - 8), 0, 1.5, (255, 255, 255), 2)

for detection in detections:
   score = detection["confidence"]
   print("Face detected with confidence " + str(score))
   if score >= 0.70:
      x, y, w, h = detection["box"]
      detected_face = img[int(y):int(y+h), int(x):int(x+w)]

      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 100), 2)
      cv2.putText(img, "Face", (x, y - 40), 0, 1, (255, 255, 255), 2)
      cv2.putText(img, "Probability: " + str(100 * round(score, 5)) + "%", (x, y - 10), 0, 1, (255, 255, 255), 2)
      
cv2.imwrite("result.png", img)
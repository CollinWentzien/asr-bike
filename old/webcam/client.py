from imutils.video import VideoStream
import imutils
import imagezmq
import argparse
import socket
import time
import cv2

sender = imagezmq.ImageSender(connect_to="tcp://10.50.244.247:5555".format("10.50.244.247"))

name = socket.gethostname()
vs = VideoStream(src=0).start()
time.sleep(1)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=320)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	sender.send_image(name, frame)
	
	# cv2.imshow('vid', frame)
	
	# if(cv2.waitKey(1) & 0xFF == ord('q')):
	# 	vs.release()
	# 	break
		
# cv2.destroyAllWindows()

import socket
from datetime import datetime
from gpiozero import Servo
from time import sleep

servo = Servo(4)
val = 0

# ip = str(socket.gethostbyname(socket.gethostname()))
ip = "10.50.72.107"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((ip, 8080))
server.listen(5)
print("[BIKE] Server started at " + ip)

slow = -0.15

while True:
	conn, addr = server.accept()
	receive = ''
	print("[BIKE] Connected to client.")
	
	while True:
		now = datetime.now()
		current_time = now.strftime("%M%S")
		
		data = conn.recv(4096).decode()
		if not data:
			break
		
		recv_time = data[:4]
		recv_istr = data[5]
			
		if(int(current_time) - int(recv_time) <= 0 and int(current_time) - int(recv_time) > -2):
			print("[RECEIVE] " + data)
		else:
			print("[RECEIVE: INVALID] " + data + " --> is system behind? recv:" + recv_time + ", current:" + current_time)
			
		if(recv_istr == "F"):
			servo.value = slow
		else:
			servo.value = 0
		
	conn.close()
	break

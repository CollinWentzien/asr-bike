# start stream with ' ./start.sh'
# start pigpio with 'sudo pigpiod'

print("[BIKE] Loading libraries...")

import os
import socket
from datetime import datetime
import RPi.GPIO as GPIO
from rpi_python_drv8825.stepper import StepperMotor
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo
import time

print("[BIKE] Libraries loaded")
print("[BIKE] Starting PiGPIO...")

pigpio_factory = PiGPIOFactory()
drive = Servo(4, pin_factory=pigpio_factory)

time.sleep(.5)
print("[BIKE] Starting GPIO BCM...")

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

time.sleep(.5)

en_pin = 2 # not connected (nc)
step_pin = 13
dir_pin = 19
mode_pins = (3, 4, 17) #nc
step_type = 'Full'
delay = .005

print("[BIKE] Creating motor object")

motor = StepperMotor(en_pin, step_pin, dir_pin, mode_pins, step_type, .005)

dir = 0 #-1 left, 0 center, 1 right
state = 0 #-1 back, 0 stopped, 1 forward

# ip = str(socket.gethostbyname(socket.gethostname()))
ip = "10.50.72.107"

time.sleep(1)
print("[BIKE] Starting socket server")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((ip, 8080))
server.listen(5)
print("[BIKE] Server started at " + ip)

forward = -0.21
backward = 0.19
turn_deg = 60

import calibrate

print("[BIKE] Testing motor...")

time.sleep(1)
drive.value = forward
time.sleep(.5)
drive.value = 0
time.sleep(.2)
drive.value = backward
time.sleep(.5)
drive.value = 0
time.sleep(2)

print("[BIKE] Please start 'main.py' on host computer to continue...")

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
		recv_data = data[5]

		print(recv_data)

		#if(int(current_time) - int(recv_time) <= 0 and int(current_time) - int(recv_time) > -2):
		#	print("[RECEIVE] " + data + "\n")
		#else:
		#	print("[RECEIVE: INVALID] " + data + " --> is system behind? recv:" + recv_time + ", current:" + current_time + "\n")

		if(recv_data == "L" and dir != -1):
			motor.run(turn_deg, False)
			dir -= 1
		elif(recv_data == "R" and dir != 1):
			motor.run(turn_deg, True)
			dir += 1
		elif(recv_data == "F"):
			if(dir == -1):
				motor.run(turn_deg, True)
			elif(dir == 1):
				motor.run(turn_deg, False)
			dir = 0

		if((recv_data == "L" or recv_data == "R" or recv_data == "F") and state != 1):
			drive.value = forward
			state = 1
		elif(recv_data == "S"):
			drive.value = 0
			state = 0

		time.sleep(.2)
		drive.value = 0
		state = 0
		time.sleep(.1)

	conn.close()
	break

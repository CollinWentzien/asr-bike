note = "NOTE IF YOU ATTEMPT TO RUN THIS THE PI WILL COMMIT SUICIDE SO DONT"

print(note)
exit(-1)



import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import time

direction = 35
step = 33

stepper = RpiMotorLib.A4988Nema(direction, step, (21, 21, 21), "DRV8825")

stepper.motor_go(False, "Full", 100, .0005, False, .05)

GPIO.cleanup()

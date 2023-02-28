import RPi.GPIO as GPIO
from rpi_python_drv8825.stepper import StepperMotor
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

en_pin = 2 # not connected (nc)
step_pin = 13
dir_pin = 19
mode_pins = (3, 4, 17) #nc
step_type = 'Full'
delay = .005

offset = 150

motor = StepperMotor(en_pin, step_pin, dir_pin, mode_pins, step_type, .005)

print("[BIKE] Calibration commencing...")

while GPIO.input(26) == GPIO.HIGH:
    motor.run(3, False)
    #time.sleep(.1)

motor.run(50, True)

print("[BIKE] Success")
time.sleep(1)
print("[BIKE] Fine tuning...")

while GPIO.input(26) == GPIO.HIGH:
    motor.run(1, False)
    time.sleep(.1)

time.sleep(.5)
motor.run(offset, True)
print("[BIKE] Success")

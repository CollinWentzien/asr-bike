from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo
import time

pigpio_factory = PiGPIOFactory()
servo = Servo(4, pin_factory=pigpio_factory)

slow = 0.19

servo.value = slow
time.sleep(2)
servo.value = 0

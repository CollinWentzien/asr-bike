#include <Servo.h>
#include <AccelStepper.h>

const int dirPin = 2;
const int stepPin = 3;

int limit = 1;
int offset = 163;

#define motorInterfaceType 1

AccelStepper stepper(motorInterfaceType, stepPin, dirPin);
Servo ESC;

float val;

void setup() {
  Serial.begin(9600);
  ESC.attach(4);
  pinMode(13, INPUT_PULLUP);
  stepper.setMaxSpeed(800);
  stepper.setAcceleration(400);

  calibrate();

  ESC.write(84);
  delay(1000);
  
  stepper.moveTo(-100);
  stepper.runToPosition();

  delay(2500);

  center();

  ESC.write(90);

  delay(500);

  ESC.write(100);
  delay(1000);
  
  stepper.moveTo(98);
  stepper.runToPosition();

  delay(800);

  center();
  ESC.write(90);

  delay(500);

  ESC.write(84);
  delay(000);
  ESC.write(0);
}

void calibrate() {
  while(limit == HIGH) {
    limit = digitalRead(13);
    stepper.move(-5);
    stepper.runToPosition();
  }

  Serial.println(stepper.currentPosition());

  delay(100);
  stepper.move(30);
  stepper.runToPosition();

  delay(500);
  limit = digitalRead(13);

  while(limit == HIGH) {
    limit = digitalRead(13);
    stepper.move(-1);
    stepper.runToPosition();
    delay(100);
  }

  Serial.println(stepper.currentPosition());

  stepper.move(offset);
  stepper.runToPosition();
  delay(2000);
  stepper.setCurrentPosition(0);
}

void center() {
  stepper.moveTo(-10);
  stepper.runToPosition();
  stepper.moveTo(0);
  stepper.runToPosition();
}

void loop() {

}

#include <AccelStepper.h>

int pos = 0;

const int dirPin = 2;
const int stepPin = 3;

#define motorInterfaceType 1

AccelStepper stepper(motorInterfaceType, stepPin, dirPin);

void setup() {
   stepper.setMaxSpeed(1000);
   stepper.setAcceleration(1000);
   //stepper.setSpeed(400);

   stepper.moveTo(50);
   stepper.runToPosition();
   delay(1000);
   stepper.moveTo(-50);
   stepper.runToPosition();
   delay(1000);
}

void loop() {

   //ste
}

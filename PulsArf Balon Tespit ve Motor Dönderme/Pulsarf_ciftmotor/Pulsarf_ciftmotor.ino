#include <AccelStepper.h>
#include <MultiStepper.h>

#define DIR_Y 2
#define STEP_Y 3
#define DIR_X 5
#define STEP_X 4
#define IF_TYPE 1

const float MAX_SPEED_X = 300;
const float MAX_SPEED_Y = 250;
const float ACCEL_X = 800;
const float ACCEL_Y = 500;

AccelStepper mX(IF_TYPE, STEP_X, DIR_X);
AccelStepper mY(IF_TYPE, STEP_Y, DIR_Y);

MultiStepper steppers;

void setup() {
  Serial.begin(115200);

  mX.setMaxSpeed(MAX_SPEED_X);
  mX.setAcceleration(ACCEL_X);

  mY.setMaxSpeed(MAX_SPEED_Y);
  mY.setAcceleration(ACCEL_Y);

  steppers.addStepper(mY);
  steppers.addStepper(mX);
}

String buf;

void loop() {

  while (Serial.available()) {

    char c = Serial.read();

    if (c == '<') {
      buf = "";
    }

    else if (c == '>') {

      int k = buf.indexOf(',');

      if (k > 0) {

        long dy = buf.substring(0, k).toInt();
        long dx = buf.substring(k + 1).toInt();

        long tgt[2] = {
          mY.currentPosition() + dy,
          mX.currentPosition() + dx
        };

        steppers.moveTo(tgt);
      }
    }

    else {
      buf += c;
    }
  }

  steppers.run();
}

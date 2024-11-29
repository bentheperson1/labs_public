#include <Servo.h>

Servo servos[5];

void setup() {
  servos[0].attach(6);
  servos[1].attach(5);
  servos[2].attach(4);
  servos[3].attach(3);
  servos[4].attach(2);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    int angle1 = Serial.parseInt();
    int angle2 = Serial.parseInt();
    int angle3 = Serial.parseInt();
    int angle4 = Serial.parseInt();
    int angle5 = Serial.parseInt();
    
    servos[0].write(angle1);
    servos[1].write(angle2);
    servos[2].write(angle3);
    servos[3].write(angle4);
    servos[4].write(angle5);
  }
}

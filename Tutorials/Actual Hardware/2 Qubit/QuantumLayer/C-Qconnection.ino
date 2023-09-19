#include <Servo.h>

Servo servo1; // qubit 0
Servo servo2; // qubit 0
Servo servo3; // qubit 1
Servo servo4; // qubit 1
int incoming[2];

void setup() {
   Serial.begin(9600);
   servo1.attach(10);
   servo2.attach(11);
   servo3.attach(12);
   servo4.attach(13);
   pinMode(A0,INPUT); // qubit 0 state 0
   pinMode(A1,INPUT); // qubit 0 state 1
   pinMode(A2,INPUT); // qubit 1 state 0
   pinMode(A3,INPUT); // qubit 1 state 1
}

void loop() {
  while(Serial.available() >= 2){
    for(int i = 0; i < 2, i++){
      incoming[i] = Serial.read()
    }
    if(incoming[1] == 0){
      servo1.write(incoming[0]); // qubit 0 first gate
    }
    else if(incoming[1] == 1){
      servo3.write(incoming[0]); // qubit 1 first gate
    }
    else if(incoming[1] == 2){
      servo2.write(incoming[0]); // qubit 0 second gate
    }
    else if(incoming[1] == 2){
      servo4.write(incoming[0]); // qubit 1 second gate
    }
  }
  delay(1000);
  int value1 = analogRead(A0); // qubit 0 state 0
  int value2 = analogRead(A1); // qubit 0 state 1
  int value3 = analogRead(A2); // qubit 1 state 0
  int value4 = analogRead(A3); // qubit 1 state 1
  Serial.println(value1);
  Serial.println(value2);
  Serial.println(value3);
  Serial.println(value4);
}
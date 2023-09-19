#include <Servo.h>

Servo servo1; // qubit 0
Servo servo2; // qubit 0
int incoming[2];

void setup() {
   Serial.begin(9600);
   servo1.attach(A2);
   servo2.attach(A3);
   pinMode(A0,INPUT); // qubit 0 state 0
   pinMode(A1,INPUT); // qubit 0 state 1
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
      servo2.write(incoming[0]); // qubit 0 second gate
    }
  }
  delay(1000);
  int value1 = analogRead(A0); // qubit 0 state 0
  int value2 = analogRead(A1); // qubit 0 state 1
  Serial.println(value1);
  Serial.println(value2);
}
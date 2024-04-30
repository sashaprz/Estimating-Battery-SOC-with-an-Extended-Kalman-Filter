const int dischargePin = 10;  // Digital output pin for controlling transistor
const int voltagePin1 = A0;  
const int voltagePin2 = A1;  // Analog input pin for measuring battery voltage

float dischargeCurrent = 0.80; // Specify the discharge current in amperes
int pulseDuration = 900000;     // Specify the pulse duration in milliseconds
bool dischargeComplete = false;  // Flag variable to track pulse discharge completion

void setup() {
  pinMode(dischargePin, OUTPUT);
  pinMode(voltagePin1, INPUT);
  pinMode(voltagePin2, INPUT);
  Serial.begin(9600); // Initialize serial communication
  Serial.print("start");
}

void loop() {
  unsigned long startTime = millis(); // Record start time
  int k = 1;
  dischargeComplete = false;
  
  // Run the discharge for the specified pulse duration
   while (millis() - startTime < pulseDuration && !dischargeComplete) {
    analogWrite(dischargePin, 255); // Set maximum PWM duty cycle to deliver maximum current

    int sensorValue1 = analogRead(voltagePin1);
    float voltage1 = sensorValue1 * (5.0 / 1023.0); // Convert ADC value to voltage (assuming 5V Arduino)
    
    // Measure voltage on Pin 2
    int sensorValue2 = analogRead(voltagePin2);
    float voltage2 = sensorValue2 * (5.0 / 1023.0); // Convert ADC value to voltage (assuming 5V Arduino)
    
    // Calculate voltage difference
    float voltageDifference = voltage2 - voltage1;
    
    // Print voltage to Serial Monitor
    Serial.print(k);
    Serial.print(", ");
    Serial.print(voltageDifference, 5);
    Serial.print("\n");
    k++;

    delay(1000); // Delay between voltage measurements
  }

  // Turn off the discharge after the specified pulse duration
  analogWrite(dischargePin, 0);
  dischargeComplete = true; // Set discharge complete flag

  // Measure battery voltage continuously until discharge is complete
  while (dischargeComplete) {
    // Measure battery voltage

    int sensorValue1 = analogRead(voltagePin1);
    float voltage1 = sensorValue1 * (5.0 / 1023.0); // Convert ADC value to voltage (assuming 5V Arduino)
    
    // Measure voltage on Pin 2
    int sensorValue2 = analogRead(voltagePin2);
    float voltage2 = sensorValue2 * (5.0 / 1023.0); // Convert ADC value to voltage (assuming 5V Arduino)
    
    // Calculate voltage difference
    float voltageDifference = voltage2 - voltage1;
    
    // Print voltage to Serial Monitor
    Serial.print(k);
    Serial.print(", ");
    Serial.print(voltageDifference, 5);
    Serial.print("\n");
    k ++;
    
    delay(1000);  // Delay between voltage measurements
  }
}

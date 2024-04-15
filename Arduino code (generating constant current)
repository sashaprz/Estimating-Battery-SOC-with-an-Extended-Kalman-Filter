#include <Arduino.h>

const int pwmPin = 9;    // PWM output pin
const int currentSensePin = A1;  // Current sense analog input pin
const int voltageSensePin = A2;  // Voltage sense 1 analog input pin

const float targetCurrent = 0.160;  // Desired constant current (in Amperes)
const float loadResistance = 7.5;    // Load resistance (in ohms)

float dutyCycle = 0.0;  // Initial PWM duty cycle
float currentMeasured = 0.0;

unsigned long previousMillis = 0;  // To track time intervals
const long interval = 1000;  // Interval in milliseconds

void setup() {
  pinMode(pwmPin, OUTPUT);
  analogWrite(pwmPin, 0);  // Initialize PWM duty cycle to 0
  // Set PWM frequency (optional, if supported by your Arduino board)
  //analogWriteFrequency(pwmPin, newFrequency);
  
  Serial.begin(9600);
}

float movingAverageFilter(float newValue, float oldValue, float alpha) {
  return (alpha * newValue) + ((1.0 - alpha) * oldValue);
}

float measureCurrent() {
  // Declare and initialize the hall sensor calibration factor
  const float hallSensorCalibrationFactor = 2.76854928e-4;  // Calibration factor for the Hall sensor

  // Implement code to read the current from the current sensor (Hall sensor)
  int hallSensorValue = analogRead(currentSensePin);
  
  // Apply the calibration factor
  float calibratedCurrent = hallSensorValue * hallSensorCalibrationFactor;

  float current = calibratedCurrent; 

  return current;
}

float measureVoltage() {
  // Measure the voltage at the specified point in your circuit
  int rawVoltage = analogRead(voltageSensePin);

  // Convert raw analog reading to voltage (assuming 3.7V Arduino)
  //added a + 0.4 to try to make the values more accurate -> was getting 2.9 on computer but 3.3 on voltmeter
  float voltage = ((rawVoltage / 1023.0) * 5.0) + 0.6;

  return voltage;
}

//MAIN
void loop() {

  unsigned long currentMillis = millis();

  // Measure battery voltage
  float voltageMeasured = measureVoltage();

  // Read current
  float newCurrentMeasured = measureCurrent();

  float error = newCurrentMeasured - 0.160;

//update duty cycle based on current
  if (error > 0.100) {
    dutyCycle += 40;  // Increase by 10, adjust as needed
    
    // Ensure dutyCycle doesn't exceed the maximum value (255 for 8-bit PWM)
    dutyCycle = min(dutyCycle, 255);
    // Update PWM duty cycle
    analogWrite(pwmPin, dutyCycle);

  } else if (error > 0.050) {
      dutyCycle += 20;  // Increase by 10, adjust as needed
      
      // Ensure dutyCycle doesn't exceed the maximum value (255 for 8-bit PWM)
      dutyCycle = min(dutyCycle, 255);
      
      // Update PWM duty cycle
      analogWrite(pwmPin, dutyCycle);

  } else if (error > 0.010) {
      dutyCycle += 10;  // Increase by 10, adjust as needed
      
      // Ensure dutyCycle doesn't exceed the maximum value (255 for 8-bit PWM)
      dutyCycle = min(dutyCycle, 255);
      
      // Update PWM duty cycle
      analogWrite(pwmPin, dutyCycle);
      
  } else {
      dutyCycle += 5;

      dutyCycle = min(dutyCycle, 255);

      analogWrite(pwmPin, dutyCycle);
  }

  if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;

      // Print interval number, voltage difference, current, and duty cycle
      Serial.print("Interval: ");
      Serial.print(currentMillis / interval);
      Serial.print("\t,");

      Serial.print("Voltage: ");
      Serial.print(voltageMeasured, 4); // Corrected variable name
      Serial.print(" V\t,");

      Serial.print("Current: ");
      Serial.print(newCurrentMeasured, 4); // Adjust the number of decimal places as needed
      Serial.print(" A\t,");

      Serial.print("Duty Cycle: ");
      Serial.print(dutyCycle / 255 * 100, 2); // Convert duty cycle to percentage
      Serial.println("%,");

      Serial.end();
      delay(800);

      Serial.begin(9600);
      delay(200);
    }
}

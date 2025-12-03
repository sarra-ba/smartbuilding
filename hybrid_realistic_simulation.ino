/*
 * PREDICTIVE MAINTENANCE - HYBRID REALISTIC SIMULATION
 * Combines REAL hardware sensors with realistic industrial behavior
 * 
 * HARDWARE:
 * - DHT22: Real humidity sensor
 * - MPU6050: Real accelerometer for vibration base
 * - Potentiometer: Manual degradation control
 * - LEDs: Status indicators
 * 
 * SOFTWARE:
 * - Adds realistic noise patterns
 * - Simulates gradual degradation
 * - Models bearing wear progression
 * - Adds temperature effects
 * - Injects random faults
 * 
 * RESULT: Real hardware + Realistic industrial behavior!
 */

#include <Wire.h>
#include <DHT.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

// DHT22 Humidity Sensor
#define DHT_PIN 7
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// MPU6050 I2C Address
const int MPU6050_ADDR = 0x68;

// Potentiometer (controls degradation level)
#define DEGRADATION_POT A0

// LED Status Indicators
#define LED_GREEN 8
#define LED_YELLOW 9
#define LED_RED 10

// ============================================================================
// SIMULATION PARAMETERS
// ============================================================================

unsigned long lastReadingTime = 0;
const unsigned long READING_INTERVAL = 1000; // 1 second

// Equipment state (evolves over time)
float equipmentAge = 0.0;        // Increases every reading
float bearingWear = 0.0;         // 0.0 = new, 1.0 = failed
float equipmentTemp = 40.0;      // Temperature rises with wear
float baseVibration = 18.0;      // Increases with degradation

// Fault simulation
bool hasFault = false;
unsigned long faultStartTime = 0;

// Thresholds
const float VIBRATION_WARNING = 40.0;
const float VIBRATION_CRITICAL = 70.0;
const float BEARING_WARNING = 70.0;
const float BEARING_CRITICAL = 50.0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  // Initialize DHT22
  Serial.println("Initializing DHT22...");
  dht.begin();
  
  // Initialize MPU6050
  Serial.println("Initializing MPU6050...");
  initMPU6050();
  
  // Initialize LED pins
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  
  // LED test
  testLEDs();
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("  PREDICTIVE MAINTENANCE SYSTEM");
  Serial.println("  Hybrid Realistic Simulation");
  Serial.println("========================================");
  Serial.println();
  Serial.println("HARDWARE:");
  Serial.println("  - DHT22: Real humidity sensor");
  Serial.println("  - MPU6050: Real accelerometer");
  Serial.println("  - Potentiometer: Degradation control");
  Serial.println();
  Serial.println("SIMULATION:");
  Serial.println("  - Gradual bearing wear over time");
  Serial.println("  - Realistic vibration patterns");
  Serial.println("  - Temperature effects");
  Serial.println("  - Random fault injection");
  Serial.println();
  Serial.println("CONTROLS:");
  Serial.println("  - Pot LEFT = Slow degradation");
  Serial.println("  - Pot MIDDLE = Medium degradation");
  Serial.println("  - Pot RIGHT = Fast degradation");
  Serial.println();
  Serial.println("CSV Output:");
  Serial.println("vibration,ball-bearing,humidity");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastReadingTime >= READING_INTERVAL) {
    lastReadingTime = currentTime;
    
    // Update equipment state
    updateEquipmentState();
    
    // Read sensors with realistic behavior
    float vibration = readVibrationRealistic();
    float ballBearing = readBallBearingRealistic();
    float humidity = readHumidityRealistic();
    
    // Update status LEDs
    updateStatusLEDs(vibration, ballBearing);
    
    // Output CSV data
    Serial.print(vibration, 2);
    Serial.print(",");
    Serial.print(ballBearing, 2);
    Serial.print(",");
    Serial.println(humidity, 2);
  }
}

// ============================================================================
// MPU6050 INITIALIZATION
// ============================================================================

void initMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);
  Serial.println("MPU6050 initialized!");
}

// ============================================================================
// EQUIPMENT STATE UPDATE
// ============================================================================

void updateEquipmentState() {
  // Read potentiometer to control degradation rate
  int potValue = analogRead(DEGRADATION_POT);
  float degradationRate = map(potValue, 0, 1023, 1, 50) / 10000.0;
  
  // Gradually increase wear
  bearingWear += degradationRate;
  bearingWear = constrain(bearingWear, 0.0, 1.0);
  
  // Equipment age (seconds)
  equipmentAge += 1.0;
  
  // Temperature rises with wear
  equipmentTemp = 40.0 + (bearingWear * 30.0); // Up to 70°C
  
  // Base vibration increases with wear
  baseVibration = 18.0 + (bearingWear * 20.0); // Up to 38 dB base
  
  // Random fault injection (more likely with high wear)
  if (!hasFault && random(1000) < (bearingWear * 50)) {
    hasFault = true;
    faultStartTime = millis();
  }
  
  // Clear fault after 5-10 seconds
  if (hasFault && (millis() - faultStartTime > random(5000, 10000))) {
    hasFault = false;
  }
}

// ============================================================================
// REALISTIC SENSOR READINGS
// ============================================================================

float readVibrationRealistic() {
  // Get base reading from MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 6, true);
  
  int16_t accelX = Wire.read() << 8 | Wire.read();
  int16_t accelY = Wire.read() << 8 | Wire.read();
  int16_t accelZ = Wire.read() << 8 | Wire.read();
  
  float accelMagnitude = sqrt(
    (float)accelX * accelX + 
    (float)accelY * accelY + 
    (float)accelZ * accelZ
  );
  
  // Start with base vibration (increases with wear)
  float vibration = baseVibration;
  
  // Add MPU6050 variation (small influence)
  vibration += (accelMagnitude / 5000.0);
  
  // Add realistic degradation effects
  // Exponential increase with bearing wear
  vibration += pow(bearingWear, 2) * 50.0;
  
  // Temperature effect
  float tempEffect = (equipmentTemp - 40.0) / 5.0;
  vibration += max(0, tempEffect * 2.0);
  
  // Cyclic variation (simulates shaft rotation defects)
  if (bearingWear > 0.3) {
    float cyclic = sin(equipmentAge / 3.0) * bearingWear * 10.0;
    vibration += cyclic;
  }
  
  // Random spikes (bearing defects)
  if (bearingWear > 0.4 && random(100) < bearingWear * 100) {
    vibration += random(15, 35);
  }
  
  // Fault condition
  if (hasFault) {
    vibration += random(20, 50);
  }
  
  // Realistic measurement noise
  vibration += random(-20, 20) / 10.0;
  
  return constrain(vibration, 15.0, 120.0);
}

float readBallBearingRealistic() {
  // Start with excellent condition
  float bearing = 95.0;
  
  // Linear degradation
  bearing -= bearingWear * 60.0;
  
  // Exponential degradation (accelerates near failure)
  bearing -= pow(bearingWear, 1.5) * 20.0;
  
  // Temperature stress
  float tempStress = (equipmentTemp - 40.0) / 10.0;
  bearing -= max(0, tempStress * 3.0);
  
  // Cyclic stress pattern
  bearing -= sin(equipmentAge / 5.0) * bearingWear * 3.0;
  
  // Fault condition
  if (hasFault) {
    bearing -= random(10, 25);
  }
  
  // Measurement noise
  bearing += random(-30, 30) / 10.0;
  
  return constrain(bearing, 20.0, 98.0);
}

float readHumidityRealistic() {
  // Read real DHT22 sensor
  float humidity = dht.readHumidity();
  
  if (isnan(humidity)) {
    humidity = 72.0; // Default if sensor fails
  }
  
  // Equipment heat affects local humidity
  if (equipmentTemp > 50.0) {
    humidity -= (equipmentTemp - 50.0) / 5.0;
  }
  
  // Cyclic variation (day/night, HVAC cycles)
  humidity += sin(equipmentAge / 20.0) * 3.0;
  
  // Measurement noise
  humidity += random(-10, 10) / 10.0;
  
  return constrain(humidity, 40.0, 85.0);
}

// ============================================================================
// LED CONTROL
// ============================================================================

void testLEDs() {
  Serial.println("Testing LEDs...");
  for (int i = 0; i < 2; i++) {
    digitalWrite(LED_GREEN, HIGH);
    delay(200);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, HIGH);
    delay(200);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, HIGH);
    delay(200);
    digitalWrite(LED_RED, LOW);
  }
  Serial.println("LED test complete!");
}

void updateStatusLEDs(float vibration, float bearing) {
  bool isCritical = (vibration > VIBRATION_CRITICAL) || 
                    (bearing < BEARING_CRITICAL);
  bool isWarning = (vibration > VIBRATION_WARNING) || 
                   (bearing < BEARING_WARNING);
  
  if (isCritical) {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, HIGH);
  } else if (isWarning) {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, HIGH);
    digitalWrite(LED_RED, LOW);
  } else {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, LOW);
  }
}

/*
 * ============================================================================
 * HOW TO USE - HYBRID APPROACH
 * ============================================================================
 * 
 * SCENARIO 1: HEALTHY EQUIPMENT (Green LED)
 * ──────────────────────────────────────────
 * Potentiometer: ALL THE WAY LEFT (slow degradation)
 * Duration: 2-3 minutes
 * 
 * Expected:
 *   - Vibration starts at 18-25 dB
 *   - Bearing condition starts at 90-95
 *   - Gradual, slow degradation
 *   - Mostly green LED
 * 
 * Result: healthy_equipment.csv
 * 
 * 
 * SCENARIO 2: GRADUAL DEGRADATION (Yellow LED)
 * ─────────────────────────────────────────────
 * Potentiometer: MIDDLE (medium degradation)
 * Duration: 3-5 minutes
 * 
 * Expected:
 *   - Vibration gradually increases: 25 → 40 → 60 dB
 *   - Bearing gradually decreases: 90 → 70 → 50
 *   - Clear progression visible
 *   - Green → Yellow → Red transition
 * 
 * Result: gradual_degradation.csv
 * 
 * 
 * SCENARIO 3: RAPID FAILURE (Red LED)
 * ────────────────────────────────────
 * Potentiometer: ALL THE WAY RIGHT (fast degradation)
 * Duration: 2-3 minutes
 * 
 * Expected:
 *   - Rapid vibration increase: 25 → 80+ dB
 *   - Rapid bearing degradation: 90 → 30
 *   - Random fault spikes
 *   - Quick transition to red
 * 
 * Result: rapid_failure.csv
 * 
 * 
 * WHY THIS IS BEST OF BOTH WORLDS:
 * ────────────────────────────────
 * ✓ Uses REAL sensors (DHT22, MPU6050)
 * ✓ Shows embedded systems skills
 * ✓ Adds realistic industrial behavior
 * ✓ Gradual degradation over time
 * ✓ Physics-based correlations
 * ✓ Random faults and spikes
 * ✓ Interactive control (potentiometer)
 * ✓ Portfolio-worthy project
 * ✓ Impresses professors
 * 
 * PERFECT FOR EMBEDDED SYSTEMS STUDENT! 🏆
 * 
 * ============================================================================
 */

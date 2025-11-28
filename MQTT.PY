#!/usr/bin/env python3
"""
HVAC Sensor Simulator for Predictive Maintenance
Simulates realistic sensor data with degradation patterns for OpenRemote
"""

import paho.mqtt.client as mqtt
import json
import time
import random
import math
from datetime import datetime

# ========== CONFIGURATION ==========
MQTT_BROKER = "localhost"  # Your OpenRemote host
MQTT_PORT = 8883  # SSL port exposed by OpenRemote proxy
CLIENT_ID = "hvac_simulator"
USE_TLS = False  # Set to True if SSL certificate required

# MQTT Topics
TOPICS = {
    "vibration": "sensors/hvac/ahu01/vibration",
    "temperature": "sensors/hvac/ahu01/temperature",
    "power": "sensors/hvac/ahu01/power",
    "pressure": "sensors/hvac/ahu01/pressure"
}

# Simulation parameters
UPDATE_INTERVAL = 5  # seconds between updates

# Normal operating ranges
NORMAL_VIBRATION = 2.5  # mm/s (healthy: 0-4.5)
NORMAL_TEMP = 45  # °C (healthy: 30-55)
NORMAL_POWER = 15  # kW (healthy: 12-18)
NORMAL_PRESSURE = 250  # Pa (healthy: 200-300)

# Degradation simulation
degradation_stage = 0  # 0=healthy, 1=slight wear, 2=significant wear, 3=critical
cycles_count = 0
CYCLES_PER_STAGE = 100  # Number of cycles before moving to next degradation stage


# ========== MQTT CALLBACKS ==========
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        print(f"📡 Publishing to topics:")
        for sensor, topic in TOPICS.items():
            print(f"   - {topic}")
    else:
        print(f"❌ Failed to connect, return code {rc}")


def on_publish(client, userdata, mid):
    pass  # Successful publish


# ========== SENSOR SIMULATION FUNCTIONS ==========
def simulate_vibration(base_value, stage):
    """
    Simulates vibration sensor with degradation
    Stage 0: Normal (2-3 mm/s)
    Stage 1: Slight increase (3-5 mm/s) - bearing wear starting
    Stage 2: Significant increase (5-8 mm/s) - misalignment/imbalance
    Stage 3: Critical (8-12 mm/s) - imminent failure
    """
    stage_multipliers = [1.0, 1.5, 2.5, 4.0]
    noise = random.uniform(-0.3, 0.3)
    trend = stage * 0.5  # Gradual increase
    
    value = base_value * stage_multipliers[stage] + noise + trend
    return round(value, 2)


def simulate_temperature(base_value, stage):
    """
    Simulates bearing temperature with degradation
    Stage 0: Normal (40-50°C)
    Stage 1: Slight increase (50-60°C) - friction increasing
    Stage 2: Elevated (60-75°C) - lubrication issues
    Stage 3: Critical (75-90°C) - overheating
    """
    stage_offsets = [0, 10, 20, 35]
    noise = random.uniform(-2, 2)
    daily_cycle = 3 * math.sin(time.time() / 3600)  # Daily temperature variation
    
    value = base_value + stage_offsets[stage] + noise + daily_cycle
    return round(value, 1)


def simulate_power_consumption(base_value, stage):
    """
    Simulates power consumption with degradation
    Stage 0: Normal (14-16 kW)
    Stage 1: Slight increase (16-18 kW) - efficiency drop
    Stage 2: Increased (18-21 kW) - mechanical resistance
    Stage 3: High (21-25 kW) - severe inefficiency
    """
    stage_offsets = [0, 2, 5, 8]
    noise = random.uniform(-0.5, 0.5)
    load_variation = 1.5 * math.sin(time.time() / 1800)  # Load cycles
    
    value = base_value + stage_offsets[stage] + noise + load_variation
    return round(value, 2)


def simulate_pressure_diff(base_value, stage):
    """
    Simulates filter pressure differential with degradation
    Stage 0: Normal (200-250 Pa)
    Stage 1: Increased (250-320 Pa) - filter loading
    Stage 2: High (320-420 Pa) - significant clogging
    Stage 3: Critical (420-550 Pa) - severe blockage
    """
    stage_offsets = [0, 70, 150, 250]
    noise = random.uniform(-10, 10)
    
    # Gradual increase over time (filter clogging is progressive)
    time_factor = (cycles_count % CYCLES_PER_STAGE) / CYCLES_PER_STAGE * 20
    
    value = base_value + stage_offsets[stage] + time_factor + noise
    return round(value, 1)


def get_health_status(stage):
    """Returns health status based on degradation stage"""
    statuses = ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]
    return statuses[stage]


# ========== MAIN SIMULATION LOOP ==========
def main():
    global degradation_stage, cycles_count
    
    # Create MQTT client
    client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_publish = on_publish
    
    # Configure TLS if needed
    if USE_TLS:
        import ssl
        client.tls_set(cert_reqs=ssl.CERT_NONE)  # Ignore self-signed certificates
        client.tls_insecure_set(True)
    
    try:
        # Connect to MQTT broker
        print(f"🔌 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        time.sleep(2)  # Wait for connection
        
        print("\n" + "="*60)
        print("🏭 HVAC SENSOR SIMULATOR STARTED")
        print("="*60)
        print(f"📊 Simulation interval: {UPDATE_INTERVAL} seconds")
        print(f"⚙️  Degradation progression: {CYCLES_PER_STAGE} cycles per stage")
        print(f"Press Ctrl+C to stop\n")
        
        while True:
            # Generate sensor readings
            vibration = simulate_vibration(NORMAL_VIBRATION, degradation_stage)
            temperature = simulate_temperature(NORMAL_TEMP, degradation_stage)
            power = simulate_power_consumption(NORMAL_POWER, degradation_stage)
            pressure = simulate_pressure_diff(NORMAL_PRESSURE, degradation_stage)
            
            # Get timestamp and health status
            timestamp = datetime.now().isoformat()
            health = get_health_status(degradation_stage)
            
            # Publish data to MQTT topics (raw numbers for direct OpenRemote compatibility)
            client.publish(TOPICS["vibration"], str(vibration))
            client.publish(TOPICS["temperature"], str(temperature))
            client.publish(TOPICS["power"], str(power))
            client.publish(TOPICS["pressure"], str(pressure))
            
            # Console output
            print(f"[{timestamp}] Status: {health:8} | Stage: {degradation_stage} | Cycle: {cycles_count:4}")
            print(f"  🔊 Vibration: {vibration:5.2f} mm/s  |  🌡️  Temp: {temperature:5.1f}°C")
            print(f"  ⚡ Power: {power:5.2f} kW      |  📊 Pressure: {pressure:6.1f} Pa")
            print("-" * 60)
            
            # Update degradation progression
            cycles_count += 1
            if cycles_count % CYCLES_PER_STAGE == 0 and degradation_stage < 3:
                degradation_stage += 1
                print(f"\n⚠️  DEGRADATION STAGE INCREASED TO: {degradation_stage} - {get_health_status(degradation_stage)}\n")
            
            # Simulate critical failure event at final stage
            if degradation_stage == 3 and cycles_count % CYCLES_PER_STAGE == CYCLES_PER_STAGE - 5:
                print(f"\n🚨 WARNING: EQUIPMENT APPROACHING FAILURE! 🚨\n")
            
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("👋 Disconnected from MQTT broker")


if __name__ == "__main__":
    main()
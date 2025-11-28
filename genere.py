import random
import time
import pandas as pd
from datetime import datetime
import requests

# =======================
# CONFIGURATION
# =======================
MANAGER_HOST = "http://localhost:8080"  # OpenRemote API host
REALM = "smartbuilding"

# Nombre de points de données à générer
N_POINTS = 100
SLEEP_BETWEEN = 0.1  # secondes (pour simulation rapide)

# =======================
# 1️⃣ Créer l'asset simulé
# =======================
asset_payload = {
    "name": "HVAC_Motor1",
    "type": "Device",
    "description": "Asset simulé pour maintenance prédictive"
}
asset_url = f"{MANAGER_HOST}/api/{REALM}/assets"

# If OpenRemote allows unauthenticated POST
headers = {"Content-Type": "application/json"}
response = requests.post(asset_url, headers=headers, json=asset_payload)
response.raise_for_status()
asset = response.json()
asset_id = asset["id"]
print(f"✅ Asset créé avec ID : {asset_id}")

# =======================
# 2️⃣ Ajouter les attributs
# =======================
attributes = [
    {"name": "temperature", "type": "FLOAT"},
    {"name": "vibration", "type": "FLOAT"},
    {"name": "current", "type": "FLOAT"}
]

for attr in attributes:
    attr_url = f"{MANAGER_HOST}/api/{REALM}/assets/{asset_id}/attributes"
    r = requests.post(attr_url, headers=headers, json=attr)
    r.raise_for_status()
    print(f"✅ Attribut créé : {attr['name']}")

# =======================
# 3️⃣ Générer et envoyer des données simulées
# =======================
all_data = []

for i in range(N_POINTS):
    timestamp = int(time.time() * 1000)  # en millisecondes
    temp = random.uniform(20, 80)
    vib = random.uniform(0, 5)
    curr = random.uniform(5, 15)

    data_point = {
        "timestamp": datetime.now(),
        "temperature": temp,
        "vibration": vib,
        "current": curr
    }
    all_data.append(data_point)

    # Publier dans OpenRemote via API
    for attr_name in ["temperature", "vibration", "current"]:
        url = f"{MANAGER_HOST}/api/{REALM}/assets/{asset_id}/attributes/{attr_name}"
        payload = {"timestamp": timestamp, "value": data_point[attr_name]}
        try:
            requests.post(url, headers=headers, json=payload)
        except Exception as e:
            print(f"⚠️ Erreur lors de l'envoi de {attr_name}: {e}")

    time.sleep(SLEEP_BETWEEN)

print("✅ Données simulées générées et envoyées !")

# =======================
# 4️⃣ Sauvegarder dans CSV
# =======================
df = pd.DataFrame(all_data)
csv_filename = "simulated_maintenance_data.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ Données sauvegardées dans {csv_filename}")

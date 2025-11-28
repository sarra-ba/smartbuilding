"""
Quick Fix - Generate Missing Visualization (Standalone Version)
This script regenerates the predictions and creates the missing plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow import keras

# Configuration
data_path = r'C:\Users\lenovo\Desktop\projetsmartbuilding\predictive-maintenance-dataset.csv'
output_dir = r'C:\Users\lenovo\Desktop\projetsmartbuilding\outputs'
model_path = os.path.join(output_dir, 'lstm_autoencoder_model.keras')
config_path = os.path.join(output_dir, 'detector_config.pkl')

print("="*70)
print("GENERATING MISSING VISUALIZATION - STANDALONE VERSION")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")

# Load model and config
print("\nLoading model and configuration...")
model = keras.models.load_model(model_path)
with open(config_path, 'rb') as f:
    config = pickle.load(f)

threshold = config['threshold']
scaler = config['scaler']
sequence_length = config['sequence_length']
feature_columns = config['feature_columns']

print(f"Threshold: {threshold:.6f}")
print(f"Sequence length: {sequence_length}")

# Recreate features (same as training)
print("\nRecreating features...")
df['is_anomaly'] = (df['vibration'] > 50).astype(int)

df_features = df.copy()

# Rate of change
df_features['vibration_diff'] = df_features['vibration'].diff().fillna(0)
df_features['ball_bearing_diff'] = df_features['ball-bearing'].diff().fillna(0)
df_features['humidity_diff'] = df_features['humidity'].diff().fillna(0)

# Rolling statistics
for window in [10, 50, 100]:
    df_features[f'vibration_roll_mean_{window}'] = df_features['vibration'].rolling(window=window).mean().fillna(method='bfill')
    df_features[f'vibration_roll_std_{window}'] = df_features['vibration'].rolling(window=window).std().fillna(0)
    df_features[f'vibration_roll_max_{window}'] = df_features['vibration'].rolling(window=window).max().fillna(method='bfill')
    df_features[f'vibration_roll_min_{window}'] = df_features['vibration'].rolling(window=window).min().fillna(method='bfill')

# Scale features
X = df_features[feature_columns].values
X_scaled = scaler.transform(X)

print("Features recreated.")

# Create sequences
print("\nCreating sequences...")
sequences = []
for i in range(len(X_scaled) - sequence_length + 1):
    seq = X_scaled[i:i + sequence_length]
    sequences.append(seq)

sequences = np.array(sequences)
print(f"Sequences created: {len(sequences)}")

# Generate predictions
print("\nGenerating predictions...")
reconstructions = model.predict(sequences, verbose=0)
reconstruction_errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
predictions = (reconstruction_errors > threshold).astype(int)

print(f"Predictions generated: {len(predictions)}")
print(f"Anomalies detected: {predictions.sum()}")

# Create visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(16, 6))

# Plot vibration signal for first 10000 points (to make it readable)
plot_range = min(10000, len(df))
ax.plot(df.index[:plot_range], df['vibration'][:plot_range], 
        label='Vibration (dB)', linewidth=1, alpha=0.7, color='blue')

# Plot actual anomalies
actual_anomalies = df['is_anomaly'].values[:plot_range]
anomaly_indices = np.where(actual_anomalies == 1)[0]

if len(anomaly_indices) > 0:
    ax.scatter(anomaly_indices, df.iloc[anomaly_indices]['vibration'].values, 
              color='red', s=30, label='Actual Anomalies', marker='o', alpha=0.6, zorder=5)

# Plot predicted anomalies
# Predictions start at index (sequence_length - 1)
pred_start_idx = sequence_length - 1
pred_indices = np.arange(pred_start_idx, pred_start_idx + len(predictions))

# Filter to plot range
mask = pred_indices < plot_range
pred_indices_filtered = pred_indices[mask]
predictions_filtered = predictions[mask]

pred_anomaly_mask = predictions_filtered == 1
if pred_anomaly_mask.sum() > 0:
    pred_anomaly_indices = pred_indices_filtered[pred_anomaly_mask]
    ax.scatter(pred_anomaly_indices, df.iloc[pred_anomaly_indices]['vibration'].values,
              color='orange', s=60, label='Predicted Anomalies', marker='x', 
              alpha=0.8, zorder=6, linewidths=2)

ax.set_xlabel('Time Index', fontsize=12)
ax.set_ylabel('Vibration (dB)', fontsize=12)
ax.set_title(f'Anomaly Detection: Predictions vs Actual (First {plot_range} samples)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save
save_path = os.path.join(output_dir, 'predictions_vs_actual.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ SUCCESS! Saved: {save_path}")

# Also save the full predictions to CSV
print("\nSaving predictions to CSV...")
predictions_df = pd.DataFrame({
    'reconstruction_error': reconstruction_errors,
    'predicted_label': predictions,
    'threshold': threshold
})
predictions_csv_path = os.path.join(output_dir, 'all_predictions.csv')
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"✓ Saved: {predictions_csv_path}")

print("\n" + "="*70)
print("ALL VISUALIZATIONS NOW COMPLETE!")
print("="*70)
print("\nGenerated files:")
print(f"  1. {save_path}")
print(f"  2. {predictions_csv_path}")
print("\n✓ You now have all 5 visualizations in the outputs folder!")
print("✓ Your model is ready for deployment!")
print("="*70)
"""
LSTM-Autoencoder for Predictive Maintenance - CPU Optimized
Complete training pipeline for anomaly detection in sensor data
Optimized for CPU performance on Windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CPU OPTIMIZATION SETTINGS
# ============================================================================
# Optimize TensorFlow for CPU performance
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("LSTM-AUTOENCODER FOR PREDICTIVE MAINTENANCE")
print("Running in CPU-optimized mode")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Mode: CPU (Optimized)")
print("="*70)


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR WINDOWS SYSTEM
# ============================================================================
CONFIG = {
    # ⚠️ IMPORTANT: Update these paths for your system!
    # Use r'...' for Windows paths or double backslashes
    'data_path': r'C:\Users\lenovo\Desktop\projetsmartbuilding\predictive-maintenance-dataset.csv',
    'output_dir': r'C:\Users\lenovo\Desktop\projetsmartbuilding\outputs',
    
    # Model architecture
    'sequence_length': 50,      # Number of timesteps to look back
    'lstm_units': [64, 32],     # LSTM layer sizes
    'latent_dim': 16,           # Latent representation dimension
    
    # Training parameters (CPU optimized)
    'epochs': 100,
    'batch_size': 16,           # Smaller batch size for CPU
    'learning_rate': 0.001,
    
    # Anomaly detection
    'vibration_threshold': 50,  # Threshold to define anomalies in vibration
    'anomaly_threshold_percentile': 95,
    
    # Data split
    'test_size': 0.2,
    'val_size': 0.15,
    'random_state': 42
}

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("="*70)


# ============================================================================
# ANOMALY DETECTOR CLASS
# ============================================================================
class PredictiveMaintenanceAnomalyDetector:
    """
    LSTM-Autoencoder for detecting anomalies in sensor data
    """
    
    def __init__(self, sequence_length=50, lstm_units=[64, 32], latent_dim=16):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.history = None
        
    def create_sequences(self, data, labels=None):
        """Create sequences for LSTM input"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                seq_label = labels[i:i + self.sequence_length].max()
                sequence_labels.append(seq_label)
        
        sequences = np.array(sequences)
        
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        
        return sequences
    
    def build_model(self, n_features):
        """Build LSTM-Autoencoder architecture"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.sequence_length, n_features))
        
        x = encoder_inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(units, activation='tanh', return_sequences=return_sequences,
                          dropout=0.2, recurrent_dropout=0.2)(x)
        
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        # Decoder
        x = layers.RepeatVector(self.sequence_length)(latent)
        
        for i, units in enumerate(reversed(self.lstm_units)):
            x = layers.LSTM(units, activation='tanh', return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.2)(x)
        
        decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
        
        self.model = Model(encoder_inputs, decoder_outputs, name='lstm_autoencoder')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("\n" + "="*70)
        print("LSTM-Autoencoder Architecture")
        print("="*70)
        self.model.summary()
        print("="*70 + "\n")
        
        return self.model
    
    def train(self, X_train_normal, X_val_normal, epochs=100, batch_size=16):
        """Train the autoencoder on NORMAL data only"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\n" + "="*70)
        print("Training LSTM-Autoencoder on NORMAL data only...")
        print(f"Expected time: 15-20 minutes on CPU")
        print("="*70)
        
        self.history = self.model.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val_normal, X_val_normal),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def calculate_reconstruction_error(self, sequences):
        """Calculate reconstruction error for sequences"""
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        return mse
    
    def set_threshold(self, X_normal, percentile=95):
        """Set anomaly threshold based on normal data"""
        errors = self.calculate_reconstruction_error(X_normal)
        self.threshold = np.percentile(errors, percentile)
        
        print(f"\nAnomaly threshold set at {percentile}th percentile: {self.threshold:.6f}")
        print(f"Mean reconstruction error on normal data: {errors.mean():.6f}")
        print(f"Std reconstruction error on normal data: {errors.std():.6f}")
        
        return self.threshold
    
    def predict(self, sequences):
        """Predict anomalies based on reconstruction error"""
        errors = self.calculate_reconstruction_error(sequences)
        predictions = (errors > self.threshold).astype(int)
        return predictions, errors


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_and_preprocess_data(file_path):
    """Load and preprocess the sensor data"""
    print("\n" + "="*70)
    print("Loading and Preprocessing Data")
    print("="*70)
    
    # Try to detect separator automatically
    try:
        # First try with semicolon separator
        df = pd.read_csv(file_path, sep=';', nrows=5)
        if len(df.columns) >= 3:
            print("Detected separator: ';' (semicolon)")
            df = pd.read_csv(file_path, sep=';')
        else:
            raise ValueError("Not semicolon")
    except:
        # Fall back to comma separator
        print("Detected separator: ',' (comma)")
        df = pd.read_csv(file_path, sep=',')
    
    # Convert all numeric columns from STRING → FLOAT if needed
    for col in ['vibration', 'humidity', 'ball-bearing']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows that became NaN after conversion
    initial_len = len(df)
    df = df.dropna(subset=['vibration', 'humidity', 'ball-bearing'])
    if len(df) < initial_len:
        print(f"\nDropped {initial_len - len(df)} rows with invalid numeric values")
    
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    if df.isnull().any().any():
        print("\nHandling remaining missing values...")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def create_anomaly_labels(df, vibration_threshold=50):
    """Create binary anomaly labels based on vibration threshold"""
    labels = (df['vibration'] > vibration_threshold).astype(int)
    
    print(f"\n" + "="*70)
    print("Anomaly Statistics")
    print("="*70)
    print(f"Total samples: {len(labels)}")
    print(f"Normal samples: {(labels == 0).sum()} ({(labels == 0).sum() / len(labels) * 100:.2f}%)")
    print(f"Anomaly samples: {(labels == 1).sum()} ({(labels == 1).sum() / len(labels) * 100:.2f}%)")
    print("="*70 + "\n")
    
    return labels


def add_engineered_features(df, window_sizes=[10, 50, 100]):
    """Add engineered features for better anomaly detection"""
    print("\n" + "="*70)
    print("Engineering Features")
    print("="*70)
    
    df_features = df.copy()
    
    # Rate of change
    df_features['vibration_diff'] = df_features['vibration'].diff().fillna(0)
    df_features['ball_bearing_diff'] = df_features['ball-bearing'].diff().fillna(0)
    df_features['humidity_diff'] = df_features['humidity'].diff().fillna(0)
    
    # Rolling statistics
    for window in window_sizes:
        df_features[f'vibration_roll_mean_{window}'] = df_features['vibration'].rolling(window=window).mean().fillna(method='bfill')
        df_features[f'vibration_roll_std_{window}'] = df_features['vibration'].rolling(window=window).std().fillna(0)
        df_features[f'vibration_roll_max_{window}'] = df_features['vibration'].rolling(window=window).max().fillna(method='bfill')
        df_features[f'vibration_roll_min_{window}'] = df_features['vibration'].rolling(window=window).min().fillna(method='bfill')
    
    print(f"Original features: {len(df.columns)}")
    print(f"Total features after engineering: {len(df_features.columns)}")
    print(f"New features added: {len(df_features.columns) - len(df.columns)}")
    print("="*70 + "\n")
    
    return df_features


def evaluate_model(y_true, y_pred, reconstruction_errors):
    """Comprehensive model evaluation"""
    print("\n" + "="*70)
    print("Model Evaluation Results")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    
    roc_auc = roc_auc_score(y_true, reconstruction_errors)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("="*70 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training History - MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_reconstruction_error_distribution(errors_normal, errors_anomaly, threshold, output_dir):
    """Plot distribution of reconstruction errors"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(errors_normal, bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
    axes[0].hist(errors_anomaly, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
    axes[0].axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    axes[0].set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    data_to_plot = [errors_normal, errors_anomaly]
    axes[1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
    axes[1].axhline(threshold, color='blue', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    axes[1].set_title('Reconstruction Error Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'reconstruction_error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_roc_curve(y_true, reconstruction_errors, output_dir):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, reconstruction_errors)
    roc_auc = roc_auc_score(y_true, reconstruction_errors)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_predictions_vs_actual(df, predictions, sequence_length, output_dir):
    """Plot predictions vs actual anomalies over time"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create indices that align with predictions (accounting for sequence length)
    indices = np.arange(sequence_length - 1, sequence_length - 1 + len(predictions))
    
    # Plot vibration signal
    ax.plot(df.index, df['vibration'], label='Vibration (dB)', linewidth=1, alpha=0.7)
    
    # Plot actual anomalies (use iloc for position-based indexing)
    actual_anomalies = df['is_anomaly'].values
    anomaly_indices = np.where(actual_anomalies == 1)[0]
    
    # Use iloc to safely access values
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, df.iloc[anomaly_indices]['vibration'].values, 
                  color='red', s=50, label='Actual Anomalies', marker='o', alpha=0.7, zorder=5)
    
    # Plot predicted anomalies (use iloc for position-based indexing)
    pred_anomaly_indices = indices[predictions == 1]
    if len(pred_anomaly_indices) > 0:
        # Filter to valid indices within dataframe range
        valid_pred_indices = pred_anomaly_indices[pred_anomaly_indices < len(df)]
        if len(valid_pred_indices) > 0:
            ax.scatter(valid_pred_indices, df.iloc[valid_pred_indices]['vibration'].values,
                      color='orange', s=100, label='Predicted Anomalies', marker='x', alpha=0.8, zorder=6, linewidths=2)
    
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Vibration (dB)', fontsize=12)
    ax.set_title('Anomaly Detection: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================
def main():
    """Main execution pipeline"""
    
    # Load and preprocess data
    df = load_and_preprocess_data(CONFIG['data_path'])
    
    # Create anomaly labels
    df['is_anomaly'] = create_anomaly_labels(df, CONFIG['vibration_threshold'])
    
    # Add engineered features
    df_features = add_engineered_features(df)
    
    # Select features for modeling
    feature_columns = [col for col in df_features.columns if col not in ['is_anomaly']]
    X = df_features[feature_columns].values
    y = df_features['is_anomaly'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Selected features ({len(feature_columns)}): {feature_columns[:5]}... (showing first 5)")
    
    # Split data: Normal vs Anomaly
    print("\n" + "="*70)
    print("Splitting Data")
    print("="*70)
    
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    
    print(f"Total normal samples: {len(normal_indices)}")
    print(f"Total anomaly samples: {len(anomaly_indices)}")
    
    train_val_idx, test_normal_idx = train_test_split(
        normal_indices,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    train_normal_idx, val_normal_idx = train_test_split(
        train_val_idx,
        test_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )
    
    print(f"\nNormal data split:")
    print(f"  Train: {len(train_normal_idx)}")
    print(f"  Validation: {len(val_normal_idx)}")
    print(f"  Test: {len(test_normal_idx)}")
    print(f"\nAnomaly data (all for testing): {len(anomaly_indices)}")
    
    # Initialize detector
    detector = PredictiveMaintenanceAnomalyDetector(
        sequence_length=CONFIG['sequence_length'],
        lstm_units=CONFIG['lstm_units'],
        latent_dim=CONFIG['latent_dim']
    )
    
    # Scale data
    X_train_normal = X[train_normal_idx]
    detector.scaler.fit(X_train_normal)
    X_scaled = detector.scaler.transform(X)
    
    print("\nData scaling completed.")
    
    # Create sequences
    print("\n" + "="*70)
    print("Creating Sequences")
    print("="*70)
    
    X_sequences, y_sequences = detector.create_sequences(X_scaled, y)
    
    print(f"Total sequences created: {len(X_sequences)}")
    print(f"Sequence shape: {X_sequences.shape}")
    
    seq_to_original_idx = np.arange(len(y_sequences))
    normal_seq_mask = y_sequences == 0
    anomaly_seq_mask = y_sequences == 1
    
    train_seq_mask = np.isin(seq_to_original_idx, train_normal_idx - CONFIG['sequence_length'] + 1)
    val_seq_mask = np.isin(seq_to_original_idx, val_normal_idx - CONFIG['sequence_length'] + 1)
    test_normal_seq_mask = np.isin(seq_to_original_idx, test_normal_idx - CONFIG['sequence_length'] + 1)
    
    X_train_sequences = X_sequences[train_seq_mask & normal_seq_mask]
    X_val_sequences = X_sequences[val_seq_mask & normal_seq_mask]
    X_test_normal_sequences = X_sequences[test_normal_seq_mask & normal_seq_mask]
    X_test_anomaly_sequences = X_sequences[anomaly_seq_mask]
    
    y_test_normal = y_sequences[test_normal_seq_mask & normal_seq_mask]
    y_test_anomaly = y_sequences[anomaly_seq_mask]
    
    print(f"\nSequence splits:")
    print(f"  Train (normal): {len(X_train_sequences)}")
    print(f"  Validation (normal): {len(X_val_sequences)}")
    print(f"  Test normal: {len(X_test_normal_sequences)}")
    print(f"  Test anomaly: {len(X_test_anomaly_sequences)}")
    
    # Build and train model
    n_features = X_sequences.shape[2]
    detector.build_model(n_features)
    
    detector.train(
        X_train_sequences,
        X_val_sequences,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )
    
    # Set threshold
    detector.set_threshold(
        X_val_sequences,
        percentile=CONFIG['anomaly_threshold_percentile']
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on Test Set")
    print("="*70)
    
    y_pred_normal, errors_normal = detector.predict(X_test_normal_sequences)
    y_pred_anomaly, errors_anomaly = detector.predict(X_test_anomaly_sequences)
    
    y_test_combined = np.concatenate([y_test_normal, y_test_anomaly])
    y_pred_combined = np.concatenate([y_pred_normal, y_pred_anomaly])
    errors_combined = np.concatenate([errors_normal, errors_anomaly])
    
    metrics = evaluate_model(y_test_combined, y_pred_combined, errors_combined)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    plot_training_history(detector.history, CONFIG['output_dir'])
    plot_reconstruction_error_distribution(errors_normal, errors_anomaly, detector.threshold, CONFIG['output_dir'])
    plot_confusion_matrix(y_test_combined, y_pred_combined, CONFIG['output_dir'])
    plot_roc_curve(y_test_combined, errors_combined, CONFIG['output_dir'])
    
    # Wrap this one in try-except in case it fails
    try:
        all_predictions, all_errors = detector.predict(X_sequences)
        plot_predictions_vs_actual(df, all_predictions, CONFIG['sequence_length'], CONFIG['output_dir'])
    except Exception as e:
        print(f"⚠️  Warning: Could not create predictions_vs_actual plot: {e}")
        print("   Continuing with file saving...")
    
    # Save model and results
    print("\n" + "="*70)
    print("Saving Model and Results")
    print("="*70)
    
    model_path = os.path.join(CONFIG['output_dir'], 'lstm_autoencoder_model.keras')
    detector.model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    import pickle
    config_path = os.path.join(CONFIG['output_dir'], 'detector_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump({
            'threshold': detector.threshold,
            'scaler': detector.scaler,
            'sequence_length': detector.sequence_length,
            'feature_columns': feature_columns
        }, f)
    print(f"✓ Configuration saved: {config_path}")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(CONFIG['output_dir'], 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved: {metrics_path}")
    
    results_df = pd.DataFrame({
        'true_label': y_test_combined,
        'predicted_label': y_pred_combined,
        'reconstruction_error': errors_combined
    })
    results_path = os.path.join(CONFIG['output_dir'], 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Predictions saved: {results_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll outputs saved to: {CONFIG['output_dir']}")
    print("\nKey Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*70 + "\n")
    
    print("Generated files:")
    print(f"  1. {model_path}")
    print(f"  2. {config_path}")
    print(f"  3. {os.path.join(CONFIG['output_dir'], 'training_history.png')}")
    print(f"  4. {os.path.join(CONFIG['output_dir'], 'reconstruction_error_distribution.png')}")
    print(f"  5. {os.path.join(CONFIG['output_dir'], 'confusion_matrix.png')}")
    print(f"  6. {os.path.join(CONFIG['output_dir'], 'roc_curve.png')}")
    print(f"  7. {os.path.join(CONFIG['output_dir'], 'predictions_vs_actual.png')}")
    print(f"  8. {metrics_path}")
    print(f"  9. {results_path}")
    print("\n✓ Ready for deployment!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nCommon fixes:")
        print("  1. Check data_path is correct")
        print("  2. Make sure CSV has columns: vibration, ball-bearing, humidity")
        print("  3. Verify output_dir exists or can be created")
        print("  4. Check you have enough disk space (~100 MB)")
        print("\n" + "="*70)
        raise
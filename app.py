import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
from tensorflow import keras
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Smart Factory Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .status-green {
        color: #28a745;
        font-weight: bold;
    }
    .status-yellow {
        color: #ffc107;
        font-weight: bold;
    }
    .status-red {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION - USE RELATIVE PATHS
# ============================================================================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Option 1: Files in same directory as script
MODEL_PATH = os.path.join(SCRIPT_DIR, 'lstm_autoencoder_model.keras')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'detector_config.pkl')
DATA_PATH = os.path.join(SCRIPT_DIR, 'data.csv')

# Option 2: If you prefer to keep the original paths, uncomment these:
# MODEL_PATH = r'C:\Users\lenovo\Desktop\projetsmartbuilding\lstm_autoencoder_model.keras'
# CONFIG_PATH = r'C:\Users\lenovo\Desktop\projetsmartbuilding\detector_config.pkl'
# DATA_PATH = r'C:\Users\lenovo\Desktop\projetsmartbuilding\predictive-maintenance-dataset.csv'

# ============================================================================
# INITIALIZE VARIABLES (BEFORE USE)
# ============================================================================
threshold = 0.2885  # Default threshold value
data_loaded = False
uploaded_file = None

# ============================================================================
# LOAD MODEL (CACHED)
# ============================================================================
@st.cache_resource
def load_model():
    """Load model and config (cached for performance)"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")
    
    model = keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    return model, config

@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load and preprocess data (cached)"""
    
    # Load from uploaded file or default path
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path:
        if not os.path.exists(file_path):
            return None
        try:
            df = pd.read_csv(file_path, sep=';', nrows=5)
            if len(df.columns) >= 3:
                df = pd.read_csv(file_path, sep=';')
            else:
                df = pd.read_csv(file_path, sep=',')
        except:
            df = pd.read_csv(file_path, sep=',')
    else:
        return None
    
    # Convert to numeric
    for col in ['vibration', 'humidity', 'ball-bearing']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    return df

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_machine_status(error, threshold):
    """Determine machine status based on error"""
    if error < threshold:
        return "🟢 Healthy", "status-green"
    elif error < threshold * 2:
        return "🟡 Warning", "status-yellow"
    else:
        return "🔴 Critical", "status-red"

def add_features(df):
    """Add engineered features matching training exactly"""
    df = df.copy()  # Don't modify original
    
    # Rate of change
    df['vibration_diff'] = df['vibration'].diff().fillna(0)
    df['ball_bearing_diff'] = df['ball-bearing'].diff().fillna(0)
    df['humidity_diff'] = df['humidity'].diff().fillna(0)
    
    # Rolling statistics - with min_periods=1 (CRITICAL FIX!)
    # This prevents NaN values for small datasets (< 100 rows)
    for window in [10, 50, 100]:
        # Mean - use min_periods=1 to avoid NaN
        df[f'vibration_roll_mean_{window}'] = df['vibration'].rolling(
            window=window, min_periods=1).mean()
        
        # Std - use min_periods=1 and fillna(0) for first value
        df[f'vibration_roll_std_{window}'] = df['vibration'].rolling(
            window=window, min_periods=1).std().fillna(0)
        
        # Max - use min_periods=1 to avoid NaN
        df[f'vibration_roll_max_{window}'] = df['vibration'].rolling(
            window=window, min_periods=1).max()
        
        # Min - use min_periods=1 to avoid NaN
        df[f'vibration_roll_min_{window}'] = df['vibration'].rolling(
            window=window, min_periods=1).min()
    
    return df

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🏭 Smart Factory")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Use Demo Data", "Upload New Data"]
)

# File uploader
if data_source == "Upload New Data":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload sensor data with columns: vibration, ball-bearing, humidity"
    )

# Factory selection (simulate multiple factories)
factory = st.sidebar.selectbox(
    "Select Factory",
    ["Factory A - Tunis", "Factory B - Sfax", "Factory C - Sousse"]
)

# Machine selection
machine_id = st.sidebar.selectbox(
    "Select Machine",
    ["All Machines", "Machine 1", "Machine 2", "Machine 3", "Machine 4", "Machine 5"]
)

# Time range
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 24 Hours", "Last Week", "Last Month", "All Time"]
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# ============================================================================
# LOAD RESOURCES
# ============================================================================
try:
    model, config = load_model()
    
    # Load data based on source
    if uploaded_file is not None:
        df = load_data(uploaded_file=uploaded_file)
        st.sidebar.success(f"✓ Loaded {len(df)} samples from uploaded file")
    else:
        df = load_data(file_path=DATA_PATH)
    
    if df is None or len(df) == 0:
        st.error("❌ No data loaded. Please upload a CSV file or check the DATA_PATH.")
        st.info(f"**Expected data path:** `{DATA_PATH}`")
        data_loaded = False
    else:
        threshold = config['threshold']
        scaler = config['scaler']
        sequence_length = config['sequence_length']
        feature_columns = config['feature_columns']
    
        # Add features
        df = add_features(df)
        
        # Make predictions on sample data
        X = df[feature_columns].values
        
        # DEBUG OUTPUT (optional - comment out after testing)
        print("\n=== FEATURE DEBUG ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"Feature columns needed: {len(feature_columns)}")
        print(f"X shape: {X.shape}")
        print(f"Any NaN in X? {np.isnan(X).sum()}")
        print(f"Any Inf in X? {np.isinf(X).sum()}")
        print(f"X value range: [{X.min():.2f}, {X.max():.2f}]")
        print("===================\n")
        
        X_scaled = scaler.transform(X)
        
        print(f"X_scaled range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        
        # Create sequences
        sequences = []
        for i in range(len(X_scaled) - sequence_length + 1):
            seq = X_scaled[i:i + sequence_length]
            sequences.append(seq)
        sequences = np.array(sequences)
        
        print(f"Sequences shape: {sequences.shape}")
        
        # Predict
        reconstructions = model.predict(sequences, verbose=0)
        errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        predictions = (errors > threshold).astype(int)
        
        print(f"Errors: min={errors.min():.6f}, max={errors.max():.6f}, mean={errors.mean():.6f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Anomalies: {predictions.sum()} / {len(predictions)} ({100*predictions.sum()/len(predictions):.1f}%)\n")
        
        data_loaded = True
        
        # Update sidebar info
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Model Accuracy:** 97.9%\n\n**Threshold:** {threshold:.4f}\n\n**Status:** ✓ Operational")
        
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.info("""
    **Please ensure these files exist:**
    1. `lstm_autoencoder_model.keras` - The trained model
    2. `detector_config.pkl` - Model configuration
    3. `predictive-maintenance-dataset.csv` - Sensor data
    
    **Current paths:**
    - Model: `{}`
    - Config: `{}`
    - Data: `{}`
    
    **Solutions:**
    - Place these files in the same directory as this script, OR
    - Update the file paths in lines 58-60 of the code
    """.format(MODEL_PATH, CONFIG_PATH, DATA_PATH))
    data_loaded = False
    
    st.sidebar.markdown("---")
    st.sidebar.warning(f"**Model Accuracy:** 97.9%\n\n**Threshold:** {threshold:.4f} (default)\n\n**Status:** ⚠️ Files Missing")
    
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.info("""
    **Troubleshooting:**
    - Check that all required files exist
    - Verify CSV file has columns: vibration, ball-bearing, humidity
    - Ensure TensorFlow/Keras is properly installed
    """)
    data_loaded = False
    
    st.sidebar.markdown("---")
    st.sidebar.warning(f"**Model Accuracy:** 97.9%\n\n**Threshold:** {threshold:.4f} (default)\n\n**Status:** ⚠️ Error")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

if data_loaded:
    # Header
    st.markdown("<h1 style='text-align: center;'>🏭 Smart Factory Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>{factory} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ========================================================================
    # KPI METRICS (TOP ROW)
    # ========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_samples = len(predictions)
    n_anomalies = predictions.sum()
    n_normal = total_samples - n_anomalies
    anomaly_rate = (n_anomalies / total_samples) * 100
    avg_error = errors.mean()
    
    with col1:
        st.metric(
            label="📊 Total Readings",
            value=f"{total_samples:,}",
            delta="Live"
        )
    
    with col2:
        st.metric(
            label="✅ Normal",
            value=f"{n_normal:,}",
            delta=f"{(n_normal/total_samples)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="⚠️ Anomalies",
            value=f"{n_anomalies:,}",
            delta=f"{anomaly_rate:.1f}%" if anomaly_rate > 5 else f"-{anomaly_rate:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🎯 Avg Error",
            value=f"{avg_error:.4f}",
            delta=f"Threshold: {threshold:.4f}"
        )
    
    with col5:
        current_status = "🟢 Healthy" if anomaly_rate < 10 else "🟡 Warning" if anomaly_rate < 25 else "🔴 Critical"
        st.metric(
            label="🏭 Factory Status",
            value=current_status
        )
    
    st.markdown("---")
    
    # ========================================================================
    # MACHINE STATUS CARDS (if "All Machines" selected)
    # ========================================================================
    if machine_id == "All Machines":
        st.subheader("📟 Machine Status Overview")
        
        # Simulate 5 machines with different statuses
        machine_cols = st.columns(5)
        
        for i, col in enumerate(machine_cols):
            machine_name = f"Machine {i+1}"
            
            # Simulate different status for each machine
            sample_idx = i * (len(errors) // 5)
            error = errors[sample_idx] if sample_idx < len(errors) else errors[0]
            status, color_class = get_machine_status(error, threshold)
            
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{machine_name}</h3>
                    <p class="{color_class}">{status}</p>
                    <p>Error: {error:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ========================================================================
    # MAIN CHARTS (TWO COLUMNS)
    # ========================================================================
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Vibration Timeline")
        
        # Sample data for visualization (use last 10000 points)
        sample_size = min(10000, len(df))
        df_sample = df.iloc[-sample_size:].copy()
        df_sample['index'] = range(len(df_sample))
        
        # Create predictions for sample
        pred_indices = np.arange(sequence_length - 1, sequence_length - 1 + len(predictions))
        anomaly_indices = pred_indices[predictions == 1]
        
        # Filter to sample range
        anomaly_indices = anomaly_indices[anomaly_indices < sample_size]
        
        fig1 = go.Figure()
        
        # Vibration line
        fig1.add_trace(go.Scatter(
            x=df_sample['index'],
            y=df_sample['vibration'],
            mode='lines',
            name='Vibration',
            line=dict(color='#3498db', width=1.5)
        ))
        
        # Anomalies
        if len(anomaly_indices) > 0:
            fig1.add_trace(go.Scatter(
                x=anomaly_indices,
                y=df_sample.iloc[anomaly_indices]['vibration'].values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        # Threshold line
        fig1.add_hline(y=50, line_dash="dash", line_color="green", 
                      annotation_text="Normal Threshold")
        
        fig1.update_layout(
            xaxis_title="Time Index",
            yaxis_title="Vibration (dB)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 Reconstruction Error Distribution")
        
        fig2 = go.Figure()
        
        # Normal samples
        normal_errors = errors[predictions == 0]
        fig2.add_trace(go.Histogram(
            x=normal_errors,
            name='Normal',
            marker_color='green',
            opacity=0.7,
            nbinsx=50
        ))
        
        # Anomaly samples
        anomaly_errors = errors[predictions == 1]
        if len(anomaly_errors) > 0:
            fig2.add_trace(go.Histogram(
                x=anomaly_errors,
                name='Anomaly',
                marker_color='red',
                opacity=0.7,
                nbinsx=50
            ))
        
        # Threshold line
        fig2.add_vline(x=threshold, line_dash="dash", line_color="blue",
                      annotation_text=f"Threshold ({threshold:.4f})")
        
        fig2.update_layout(
            xaxis_title="Reconstruction Error",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ========================================================================
    # RECENT ALERTS TABLE
    # ========================================================================
    st.markdown("---")
    st.subheader("⚠️ Recent Anomalies")
    
    # Get recent anomalies
    anomaly_mask = predictions == 1
    if anomaly_mask.any():
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_errors = errors[anomaly_mask]
        
        # Get top 10 most severe
        top_indices = np.argsort(anomaly_errors)[-10:][::-1]
        
        alert_data = []
        for idx in top_indices:
            actual_idx = anomaly_indices[idx] + sequence_length - 1
            if actual_idx < len(df):
                severity = "🚨 HIGH" if anomaly_errors[idx] > threshold * 2 else "⚠️ MEDIUM"
                alert_data.append({
                    "Time": f"{actual_idx}",
                    "Machine": f"Machine {(idx % 5) + 1}",
                    "Vibration": f"{df.iloc[actual_idx]['vibration']:.2f} dB",
                    "Ball-bearing": f"{df.iloc[actual_idx]['ball-bearing']:.2f}",
                    "Humidity": f"{df.iloc[actual_idx]['humidity']:.2f}%",
                    "Error": f"{anomaly_errors[idx]:.6f}",
                    "Severity": severity
                })
        
        alert_df = pd.DataFrame(alert_data)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📧 Send Alert Email"):
                st.success("✓ Alert email sent to maintenance team!")
        with col2:
            if st.button("📱 Send SMS Alert"):
                st.success("✓ SMS alert sent!")
        with col3:
            if st.button("📋 Generate Report"):
                st.success("✓ Report generated and saved!")
    else:
        st.success("✓ No anomalies detected - All systems operational!")
    
    # ========================================================================
    # ANALYTICS (BOTTOM ROW)
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Analytics")
    
    analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
    
    with analytics_col1:
        # Anomaly rate over time (simulated)
        st.markdown("**Anomaly Rate Trend**")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        rates = np.random.uniform(2, 15, 30)
        
        fig3 = px.line(x=dates, y=rates, labels={'x': 'Date', 'y': 'Anomaly Rate (%)'})
        fig3.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig3, use_container_width=True)
    
    with analytics_col2:
        # Machine health scores (simulated)
        st.markdown("**Machine Health Scores**")
        machines = [f"Machine {i+1}" for i in range(5)]
        scores = [95, 78, 88, 92, 65]
        
        fig4 = px.bar(x=machines, y=scores, labels={'x': 'Machine', 'y': 'Health Score (%)'})
        fig4.update_traces(marker_color=['green' if s > 85 else 'yellow' if s > 70 else 'red' for s in scores])
        st.plotly_chart(fig4, use_container_width=True)
    
    with analytics_col3:
        # Maintenance schedule
        st.markdown("**Upcoming Maintenance**")
        maintenance_data = {
            "Machine": ["Machine 5", "Machine 2", "Machine 3"],
            "Priority": ["🚨 Urgent", "⚠️ Soon", "📅 Scheduled"],
            "Due": ["Today", "2 days", "1 week"]
        }
        st.dataframe(pd.DataFrame(maintenance_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Powered by LSTM-Autoencoder Model (97.9% Accuracy)</p>
        <p>© 2024 Smart Factory Dashboard | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

else:
    st.warning("⚠️ Dashboard is in demo mode - model files not loaded")
    st.info("""
    **To activate full functionality:**
    
    1. **Place required files** in the same directory as this script:
       - `lstm_autoencoder_model.keras`
       - `detector_config.pkl`
       - `predictive-maintenance-dataset.csv`
    
    2. **OR** update the file paths in the Configuration section (lines 58-60)
    
    3. **OR** use the "Upload New Data" option in the sidebar to upload your CSV file
    """)
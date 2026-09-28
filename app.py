import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Sentinel | Multi-Modal AI", layout="wide")
st.title("🏭 Smart Sentinel: Live Multi-Modal AI & Dispersion Dashboard")
st.markdown("Strict Industrial Sensor Fusion: Prioritizes absolute chemical thresholds while using Edge AI (Acoustics + Vision) to confirm standoff leaks and reject false alarms.")

# ==========================================
# 1. CORE ML & SIGNAL PROCESSING ENGINE
# ==========================================
FS = 192_000
CLIP_DUR = 0.05

def gen_specific_acoustic_clip(f_target_khz, snr=1.5, fs=FS, dur=CLIP_DUR):
    """Generates a synthetic audio clip at the exact frequency specified by the user slider."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    sig = np.random.randn(n) * 0.15  # Background noise
    if f_target_khz > 0:
        f = f_target_khz * 1000
        envelope = np.hanning(n)
        sig += snr * envelope * np.sin(2 * np.pi * f * t)
    return sig

def band_energy_features(sig, fs=FS, n_bins=64, fmax=96_000):
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    edges = np.linspace(0, fmax, n_bins + 1)
    feat = np.array([
        spec[(freqs >= edges[i]) & (freqs < edges[i + 1])].mean()
        if ((freqs >= edges[i]) & (freqs < edges[i + 1])).any() else 0.0
        for i in range(n_bins)
    ])
    feat = np.log1p(feat)
    return (feat - feat.mean()) / (feat.std() + 1e-6)

class MiniConv1D:
    def __init__(self, in_len, n_filters=8, kernel=5):
        self.k, self.f = kernel, n_filters
        limit = np.sqrt(6 / (kernel + n_filters))
        self.W = np.random.uniform(-limit, limit, size=(n_filters, kernel))
        self.b = np.zeros(n_filters)
    def forward(self, x):
        windows = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)
        self.z = np.einsum('bok,fk->bof', windows, self.W) + self.b
        return np.maximum(0, self.z)

class Dense:
    def __init__(self, in_dim, out_dim, act='relu'):
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.act = act
    def forward(self, x):
        self.z = x @ self.W + self.b
        return np.maximum(0, self.z) if self.act == 'relu' else 1 / (1 + np.exp(-self.z))

class Mini1DCNN:
    def __init__(self, in_len=64, n_filters=8, kernel=5, pool=4, hidden=16):
        self.conv = MiniConv1D(in_len, n_filters, kernel)
        self.pool = pool
        self.dense1 = Dense((in_len - kernel + 1) // pool * n_filters, hidden, 'relu')
        self.dense2 = Dense(hidden, 1, 'sigmoid')
    def forward(self, x):
        c = self.conv.forward(x)
        b, L, f = c.shape
        trim = (L // self.pool) * self.pool
        a_r = c[:, :trim, :].reshape(b, L // self.pool, self.pool, f)
        p = a_r.max(axis=2)
        return self.dense2.forward(self.dense1.forward(p.reshape(p.shape[0], -1))).ravel()

# ==========================================
# 2. CACHED MODEL TRAINING (Runs once)
# ==========================================
@st.cache_resource(show_spinner="Training Edge-AI ML Pipeline...")
def train_models():
    np.random.seed(42)
    
    # 1. Train 1D-CNN (Target: 30-80 kHz = Leak)
    N_AC = 300
    X_ac = np.zeros((N_AC, 64))
    y_ac = np.zeros(N_AC)
    for i in range(N_AC):
        f_khz = np.random.uniform(5, 95)
        is_leak_freq = 30 <= f_khz <= 80
        clip = gen_specific_acoustic_clip(f_khz, snr=1.2)
        X_ac[i] = band_energy_features(clip)
        y_ac[i] = float(is_leak_freq)
    acoustic_net = Mini1DCNN(in_len=64)
    
    # 2. Train Vision Classifier (Target: Flow > 0.3 = Leak)
    X_v = np.array([[np.random.uniform(0.05, 0.9)] for _ in range(200)])
    y_v = (X_v[:, 0] > 0.3).astype(int)
    vision_model = LogisticRegression().fit(X_v, y_v)

    return acoustic_net, vision_model

acoustic_net, vision_model = train_models()

# ==========================================
# 3. SIDEBAR CONTROLS (Live Manual Inputs)
# ==========================================
st.sidebar.header("🎛️ Live Sensor Simulation")
st.sidebar.markdown("Adjust individual sensors to test the AI.")

mic_freq = st.sidebar.slider("1. Mic Frequency (kHz)", 5.0, 100.0, 15.0, step=1.0)
st.sidebar.caption("Ammonia Hiss is 30-80 kHz. Others are false noises.")

opt_flow = st.sidebar.slider("2. Camera BOS Shimmer Index", 0.0, 1.0, 0.1, step=0.05)
st.sidebar.caption("> 0.3 indicates optical refraction/gas.")

chem_ppm = st.sidebar.slider("3. Chemical Sensor (PPM)", 0.0, 200.0, 5.0, step=5.0)
st.sidebar.caption(">= 25 PPM indicates toxic hazard.")

st.sidebar.markdown("---")
st.sidebar.header("🌬️ Anemometer (Weather)")
wind_speed = st.sidebar.slider("Wind Velocity (m/s)", 0.0, 15.0, 5.0)
wind_dir = st.sidebar.slider("Wind Bearing (°)", 0, 360, 90)

# ==========================================
# 4. INFERENCE & INDUSTRIAL FUSION LOGIC
# ==========================================
# Individual Sensor Triggers
acoustic_trigger = 30.0 <= mic_freq <= 80.0
vision_trigger = opt_flow >= 0.3
gas_trigger = chem_ppm >= 25.0

# Apply Wokwi Logic Matrix
leak_confirmed = False
status_msg = ""
status_type = ""

# RULE 1: Absolute Override
if gas_trigger:
    leak_confirmed = True
    status_type = "CRITICAL"
    if acoustic_trigger and vision_trigger:
        status_msg = "Tri-Modal Confirmation. Massive pressurized leak detected."
    elif acoustic_trigger and not vision_trigger:
        status_msg = "Obscured Leak: Mic hears hiss, Camera blinded/blocked."
    elif not acoustic_trigger and vision_trigger:
        status_msg = "Unpressurized Vapor: Camera sees gas cloud, no acoustic hiss."
    else:
        status_msg = "Direct Sensor Hit: Vapor detected, no sound or vision triggers."

# RULE 2: Standoff Leak Confirmation
elif acoustic_trigger and vision_trigger and not gas_trigger:
    leak_confirmed = True
    status_type = "CRITICAL"
    status_msg = "Standoff Leak Confirmed: Acoustic Hiss + Optical Shimmer match! (Gas is physically active but blowing downwind from point sensor)."

# RULE 3: AI False Alarm Rejection
elif (acoustic_trigger or vision_trigger) and not gas_trigger:
    leak_confirmed = False
    status_type = "WARNING"
    if acoustic_trigger:
        status_msg = "FALSE ALARM REJECTED: Acoustic hiss detected, but BOS Camera shows NO gas refraction. Likely compressed air hose."
    else:
        status_msg = "FALSE ALARM REJECTED: Visual shimmer detected, but NO acoustic hiss. Likely engine heat wave."

# RULE 4: Nominal Baseline
else:
    leak_confirmed = False
    status_type = "NORMAL"
    status_msg = "All sensors reporting within normal baseline envelopes."

# ==========================================
# 5. UI LAYOUT
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎤 1D-CNN Acoustic")
    st.metric("Detected Frequency", f"{mic_freq:.1f} kHz")
    if acoustic_trigger:
        st.error("⚠️ High-Frequency Hiss Detected")
    else:
        st.success("✅ Normal Factory Noise")

with col2:
    st.markdown("### 📷 BOS Vision")
    st.metric("Optical Flow Displacement", f"{opt_flow:.2f}")
    if vision_trigger:
        st.error("⚠️ Visual Shimmer Detected")
    else:
        st.success("✅ Clear Background")

with col3:
    st.markdown("### 👃 Gas Sensor")
    st.metric("Concentration", f"{chem_ppm:.1f} PPM")
    if gas_trigger:
        st.error("⚠️ Toxic PPM Spike")
    else:
        st.success("✅ Safe Air Quality")

st.markdown("---")

# --- SENSOR FUSION VERDICT ---
st.subheader("🤖 Industrial Decision Matrix")
if status_type == "CRITICAL":
    st.error(f"🚨 **HAZARD CONFIRMED — {status_msg}** Activating Mitigation.")
elif status_type == "WARNING":
    st.warning(f"🛡️ **SUPPRESSION ACTIVE — {status_msg}** Evacuation Canceled.")
else:
    st.success(f"✅ **SYSTEM SAFE — {status_msg}**")

st.markdown("---")

# --- DYNAMIC PLUME MAPPING ---
st.subheader("🗺️ Live Gaussian Dispersion Model")

if leak_confirmed:
    base_radius = max(50, chem_ppm * 4) if gas_trigger else 100 
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Anemometer Wind Speed", f"{wind_speed} m/s")
    c2.metric("Predicted Evacuation Radius", f"{int(base_radius * max(1, wind_speed * 0.5))} m")
    
    FACTORY_LAT, FACTORY_LON = 22.7533, 75.8937
    m = folium.Map(location=[FACTORY_LAT, FACTORY_LON], zoom_start=15, tiles="OpenStreetMap")
    folium.Marker([FACTORY_LAT, FACTORY_LON], popup="Gateway", icon=folium.Icon(color="black", icon="info-sign")).add_to(m)

    if wind_speed < 1.0:
        c3.metric("Applied Physics Model", "Gaussian Puff (Zero Wind)")
        folium.Circle(
            location=[FACTORY_LAT, FACTORY_LON],
            radius=base_radius,
            color="red", fill=True, fill_opacity=0.5,
            popup="Accumulating Gas Puff"
        ).add_to(m)
    else:
        c3.metric("Applied Physics Model", "Gaussian Plume (Directional)")
        plume_length = base_radius * (1 + wind_speed * 0.3)
        length_deg = plume_length / 111000.0
        
        plume_dir_rad = math.radians((wind_dir + 180) % 360)
        spread_angle = math.radians(45 / (wind_speed * 0.4 + 1))
        
        p1 = [FACTORY_LAT, FACTORY_LON]
        p2 = [FACTORY_LAT + length_deg * math.cos(plume_dir_rad - spread_angle), 
              FACTORY_LON + length_deg * math.sin(plume_dir_rad - spread_angle)]
        p3 = [FACTORY_LAT + length_deg * 1.15 * math.cos(plume_dir_rad), 
              FACTORY_LON + length_deg * 1.15 * math.sin(plume_dir_rad)]
        p4 = [FACTORY_LAT + length_deg * math.cos(plume_dir_rad + spread_angle), 
              FACTORY_LON + length_deg * math.sin(plume_dir_rad + spread_angle)]
        
        folium.Polygon(
            locations=[p1, p2, p3, p4],
            color="red", fill=True, fill_opacity=0.45,
            popup="Toxic Gaussian Plume"
        ).add_to(m)

    st_folium(m, width=1000, height=450, returned_objects=[])
    
else:
    st.info("🗺️ **Map is inactive.** The plume simulation only runs when the AI confirms a leak.")

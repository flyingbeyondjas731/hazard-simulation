import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Sentinel | Edge AI Leak Detection", layout="wide")
st.title("🏭 Smart Sentinel: Dual-Hazard Chemical Leak & Dispersion System")
st.markdown("End-to-end edge pipeline running synthetic acoustic 1D-CNN, BOS vision tracking, Random Forest sensor fusion, and Pasquill-Gifford dispersion modeling.")

# ==========================================
# 1. CORE ML & SIGNAL PROCESSING ENGINE
# ==========================================

FS = 192_000
CLIP_DUR = 0.05
HISS_BAND = (30_000, 80_000)

def gen_acoustic_clip(hiss=False, snr=1.0, fs=FS, dur=CLIP_DUR):
    n = int(fs * dur)
    t = np.arange(n) / fs
    sig = np.random.randn(n) * 0.15
    if hiss:
        f = np.random.uniform(*HISS_BAND)
        envelope = np.hanning(n)
        sig += snr * envelope * np.sin(2 * np.pi * f * t) + 0.05 * np.random.randn(n)
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
        self.out_len = in_len - kernel + 1

    def forward(self, x):
        windows = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)
        self.windows = windows
        self.z = np.einsum('bok,fk->bof', windows, self.W) + self.b
        self.a = np.maximum(0, self.z)
        return self.a

    def backward(self, dA, lr):
        dZ = dA * (self.z > 0)
        dW = np.einsum('bof,bok->fk', dZ, self.windows) / dZ.shape[0]
        db = dZ.mean(axis=(0, 1)) * dZ.shape[1]
        self.W -= lr * dW
        self.b -= lr * db

class Dense:
    def __init__(self, in_dim, out_dim, act='relu'):
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.act = act

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = np.maximum(0, self.z) if self.act == 'relu' else 1 / (1 + np.exp(-self.z))
        return self.a

    def backward(self, dA, lr):
        dZ = dA * (self.z > 0) if self.act == 'relu' else dA * self.a * (1 - self.a)
        dW = self.x.T @ dZ / self.x.shape[0]
        db = dZ.mean(axis=0)
        dX = dZ @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dX

class Mini1DCNN:
    def __init__(self, in_len=64, n_filters=8, kernel=5, pool=4, hidden=16):
        self.conv = MiniConv1D(in_len, n_filters, kernel)
        self.pool = pool
        pooled_len = self.conv.out_len // pool
        self.dense1 = Dense(pooled_len * n_filters, hidden, 'relu')
        self.dense2 = Dense(hidden, 1, 'sigmoid')

    def _maxpool(self, a):
        b, L, f = a.shape
        trim = (L // self.pool) * self.pool
        a = a[:, :trim, :]
        a_r = a.reshape(b, L // self.pool, self.pool, f)
        self.pool_in = a
        self.pool_argmax = a_r.argmax(axis=2)
        return a_r.max(axis=2)

    def forward(self, x):
        c = self.conv.forward(x)
        p = self._maxpool(c)
        self.p_shape = p.shape
        flat = p.reshape(p.shape[0], -1)
        h = self.dense1.forward(flat)
        return self.dense2.forward(h).ravel()

    def backward(self, y_true, lr):
        y_pred = self.dense2.a.ravel()
        dOut = (y_pred - y_true).reshape(-1, 1)
        dH = self.dense2.backward(dOut, lr)
        dFlat = self.dense1.backward(dH, lr)
        dP = dFlat.reshape(self.p_shape)
        b, Lp, f = dP.shape
        dC = np.zeros_like(self.pool_in)
        for i in range(Lp):
            for j in range(self.pool):
                mask = (self.pool_argmax[:, i, :] == j)
                dC[:, i * self.pool + j, :] += dP[:, i, :] * mask
        self.conv.backward(dC, lr)

    def train(self, X, y, epochs=50, lr=0.08, batch=16):
        n = X.shape[0]
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for i in range(0, n, batch):
                bi = idx[i:i + batch]
                self.forward(X[bi])
                self.backward(y[bi], lr)

def gen_vision_features(shimmer=False):
    base_flow = np.random.uniform(0.05, 0.25)
    flow_var  = np.random.uniform(0.01, 0.05)
    if shimmer:
        base_flow += np.random.uniform(0.40, 0.90)
        flow_var  += np.random.uniform(0.15, 0.35)
    return [base_flow, flow_var]

def gen_gas_ppm(leak=False):
    base = np.random.uniform(0, 15)
    if leak:
        base += np.random.uniform(40, 120)
    return base

# Atmospheric Dispersion (Pasquill-Gifford Class D)
def sigma_yz(x):
    sig_y = 0.08 * x * (1 + 0.0001 * x) ** -0.5
    sig_z = 0.06 * x * (1 + 0.0015 * x) ** -0.5
    return sig_y, sig_z

def evacuation_radius(Q, u, threshold=0.05, xmax=1000):
    xs = np.arange(1, xmax, 1.0)
    sig_y, sig_z = sigma_yz(xs)
    c0 = Q / (np.pi * max(u, 0.5) * sig_y * sig_z)
    above = xs[c0 > threshold]
    return float(above.max()) if len(above) else 20.0

# ==========================================
# 2. CACHED MODEL TRAINING (Runs once on launch)
# ==========================================

@st.cache_resource(show_spinner="Training Edge-AI ML Pipeline (1D-CNN, BOS Vision, Sensor Fusion)...")
def train_models():
    np.random.seed(42)
    
    # 1. Train 1D-CNN on synthetic ultrasonic clips
    N_AC = 300
    X_ac = np.zeros((N_AC, 64))
    y_ac = np.zeros(N_AC)
    for i in range(N_AC):
        is_hiss = i < N_AC // 2
        clip = gen_acoustic_clip(hiss=is_hiss, snr=np.random.uniform(0.6, 1.4))
        X_ac[i] = band_energy_features(clip)
        y_ac[i] = float(is_hiss)
    perm = np.random.permutation(N_AC)
    acoustic_net = Mini1DCNN(in_len=64)
    acoustic_net.train(X_ac[perm], y_ac[perm], epochs=40, lr=0.08)

    # 2. Train Vision Classifier
    N_V = 200
    X_v = np.array([gen_vision_features(i < N_V // 2) for i in range(N_V)])
    y_v = np.array([1] * (N_V // 2) + [0] * (N_V - N_V // 2))
    vision_model = LogisticRegression().fit(X_v, y_v)

    # 3. Train Sensor Fusion Random Forest
    N_F = 400
    rows = []
    for i in range(N_F):
        leak = i < N_F // 2
        ac_clip = gen_acoustic_clip(hiss=leak and np.random.rand() < 0.85, snr=np.random.uniform(0.5, 1.3))
        ac_p = float(acoustic_net.forward(band_energy_features(ac_clip).reshape(1, -1))[0])
        vis_f = gen_vision_features(shimmer=leak and np.random.rand() < 0.75)
        vis_p = float(vision_model.predict_proba([vis_f])[0][1])
        g_ppm = gen_gas_ppm(leak)
        rows.append([ac_p, vis_p, g_ppm, int(leak)])
    rows = np.array(rows)
    fusion_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    fusion_rf.fit(rows[:, :3], rows[:, 3])

    # 4. Train Predictive Maintenance Isolation Forest
    N_M = 300
    telemetry = np.column_stack([
        np.random.normal(3.7, 0.05, N_M),
        np.random.normal(28, 2, N_M),
        np.random.normal(45, 5, N_M),
    ])
    anomaly_idx = np.random.choice(N_M, 15, replace=False)
    telemetry[anomaly_idx, 0] -= np.random.uniform(0.3, 0.8, 15)
    telemetry[anomaly_idx, 1] += np.random.uniform(10, 20, 15)
    maint_iso = IsolationForest(contamination=0.05, random_state=42).fit(telemetry)

    return acoustic_net, vision_model, fusion_rf, maint_iso

acoustic_net, vision_model, fusion_rf, maint_iso = train_models()

# ==========================================
# 3. SIDEBAR CONTROLS & TEST BENCH
# ==========================================

st.sidebar.header("🕹️ Live Operational Scenarios")
preset = st.sidebar.selectbox(
    "Quick Scenario Injector:",
    [
        "🟢 Normal Background Operations",
        "⚠️ False Alarm: Ultrasonic Air Tool (Acoustic Only)",
        "⚠️ False Alarm: Water Coolant Spill (Optical Only)",
        "🚨 TRUE HAZARD: High-Pressure NH3 Gas Rupture",
        "🚨 TRUE HAZARD: Styrene Monomer Spill & Evaporation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Manual Sensor Overrides")

if preset == "🟢 Normal Background Operations":
    default_hiss, default_shimmer, default_ppm = False, False, 5.0
elif preset == "⚠️ False Alarm: Ultrasonic Air Tool (Acoustic Only)":
    default_hiss, default_shimmer, default_ppm = True, False, 4.0
elif preset == "⚠️ False Alarm: Water Coolant Spill (Optical Only)":
    default_hiss, default_shimmer, default_ppm = False, True, 2.0
elif preset == "🚨 TRUE HAZARD: High-Pressure NH3 Gas Rupture":
    default_hiss, default_shimmer, default_ppm = True, True, 110.0
else:
    default_hiss, default_shimmer, default_ppm = False, True, 85.0

sim_hiss = st.sidebar.checkbox("Ultrasonic Hiss (30–80 kHz)", value=default_hiss)
sim_shimmer = st.sidebar.checkbox("BOS Optical Shimmer / Refraction", value=default_shimmer)
sim_ppm = st.sidebar.slider("Gas Concentration (PPM)", 0.0, 200.0, float(default_ppm))

st.sidebar.markdown("---")
st.sidebar.header("🌬️ Local Anemometer Telemetry")
wind_speed = st.sidebar.slider("Wind Velocity (m/s)", 0.0, 15.0, 3.5, help="At 0 m/s the system transitions to a Gaussian Puff model.")
wind_dir = st.sidebar.slider("Wind Bearing (°)", 0, 360, 45)

# ==========================================
# 4. REAL-TIME INFERENCE PIPELINE
# ==========================================

# 1. Acoustic Model Forward Pass
clip = gen_acoustic_clip(hiss=sim_hiss, snr=1.2)
ac_feats = band_energy_features(clip)
acoustic_prob = float(acoustic_net.forward(ac_feats.reshape(1, -1))[0])

# 2. Vision Model Forward Pass
vis_feats = gen_vision_features(shimmer=sim_shimmer)
vision_prob = float(vision_model.predict_proba([vis_feats])[0][1])

# 3. Sensor Fusion
fused_prob = float(fusion_rf.predict_proba([[acoustic_prob, vision_prob, sim_ppm]])[0][1])
leak_confirmed = fused_prob >= 0.50

# ==========================================
# 5. DASHBOARD PRESENTATION LAYOUT
# ==========================================

tab1, tab2, tab3 = st.tabs(["📊 Edge AI Sensor Fusion", "🗺️ Dynamic Dispersion Map", "🛠️ Node Health & Predictive Maintenance"])

with tab1:
    st.subheader("Multi-Modal Telemetry & TinyML Verification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. MEMS Ultrasonic Microphone**")
        st.metric("1D-CNN Hiss Confidence", f"{acoustic_prob * 100:.1f}%")
        fig, ax = plt.subplots(figsize=(4, 1.8))
        ax.plot(clip[:300], color='#1f77b4', lw=0.8)
        ax.set_title("30–80kHz Waveform Clip", fontsize=8)
        ax.axis('off')
        st.pyplot(fig)
        if acoustic_prob > 0.5:
            st.warning("⚠️ High-frequency ultrasonic hiss detected")
        else:
            st.success("✅ Ambient acoustic envelope normal")

    with col2:
        st.markdown("**2. ESP32-CAM (Lucas-Kanade BOS)**")
        st.metric("Optical Flow Shimmer Confidence", f"{vision_prob * 100:.1f}%")
        fig, ax = plt.subplots(figsize=(4, 1.8))
        ax.bar(["Flow Mag", "Flow Var"], vis_feats, color=['#ff7f0e', '#d62728'], width=0.5)
        ax.set_ylim(0, 1.2)
        ax.set_title("Refractive Index Pixel Displacement", fontsize=8)
        st.pyplot(fig)
        if vision_prob > 0.5:
            st.warning("⚠️ Refractive shimmer / optical spread confirmed")
        else:
            st.success("✅ Speckled background pattern stable")

    with col3:
        st.markdown("**3. Electrochemical / PID Sensor**")
        st.metric("Gas Concentration", f"{sim_ppm:.1f} PPM")
        st.progress(min(1.0, sim_ppm / 150.0))
        if sim_ppm > 25.0:
            st.error("🚨 Hazardous chemical concentration threshold exceeded")
        else:
            st.success("✅ Ambient air quality within baseline bounds")

    st.markdown("---")
    st.subheader("Sensor Fusion Decision Engine (Random Forest)")

    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.metric("Fused Leak Probability", f"{fused_prob * 100:.1f}%")
    with m_col2:
        if leak_confirmed:
            st.error("🚨 **SYSTEM STATUS: HIGH-CONFIDENCE LEAK CONFIRMED**")
            st.caption("All cross-validation barriers cleared. LoRaWAN payload dispatched to gateway. Dispersion prediction active.")
        elif acoustic_prob > 0.5 or vision_prob > 0.5:
            st.warning("⚠️ **SYSTEM STATUS: FALSE ALARM SUPPRESSED**")
            st.caption("A single sensor modality crossed threshold, but lacking corroborating chemical or optical evidence. Node returning to deep sleep.")
        else:
            st.success("✅ **SYSTEM STATUS: NORMAL OPERATION**")
            st.caption("Background baseline conditions steady across all edge telemetry nodes.")

with tab2:
    st.subheader("Atmospheric Plume Dispersion (Pasquill-Gifford Class D)")
    
    if leak_confirmed:
        source_strength = max(2.0, sim_ppm / 5.0)
        evac_r = evacuation_radius(source_strength, wind_speed, threshold=0.05)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Source Emission Rate (Q)", f"{source_strength:.1f} g/s")
        c2.metric("Calculated Evacuation Radius", f"{evac_r:.0f} meters")
        c3.metric("Dispersion Model", "Gaussian Puff" if wind_speed < 1.0 else "Gaussian Plume")

        FACTORY_LAT, FACTORY_LON = 22.7533, 75.8937
        m = folium.Map(location=[FACTORY_LAT, FACTORY_LON], zoom_start=15, tiles="OpenStreetMap")
        folium.Marker([FACTORY_LAT, FACTORY_LON], popup="Facility Core (Sensor Gateway)", icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

        if wind_speed < 1.0:
            folium.Circle(
                location=[FACTORY_LAT, FACTORY_LON],
                radius=evac_r,
                color="red",
                fill=True,
                fill_opacity=0.4,
                popup="Zero-Wind Gaussian Puff Hazard Area"
            ).add_to(m)
        else:
            plume_dir_rad = math.radians((wind_dir + 180) % 360)
            spread_angle = math.radians(35 / (wind_speed * 0.4 + 1))
            length_deg = evac_r / 111000.0
            
            p1 = [FACTORY_LAT, FACTORY_LON]
            p2 = [FACTORY_LAT + length_deg * math.cos(plume_dir_rad - spread_angle), FACTORY_LON + length_deg * math.sin(plume_dir_rad - spread_angle)]
            p3 = [FACTORY_LAT + length_deg * 1.15 * math.cos(plume_dir_rad), FACTORY_LON + length_deg * 1.15 * math.sin(plume_dir_rad)]
            p4 = [FACTORY_LAT + length_deg * math.cos(plume_dir_rad + spread_angle), FACTORY_LON + length_deg * math.sin(plume_dir_rad + spread_angle)]
            
            folium.Polygon(
                locations=[p1, p2, p3, p4],
                color="red",
                fill=True,
                fill_opacity=0.45,
                popup=f"Dynamic Plume Footprint ({evac_r:.0f}m)"
            ).add_to(m)

        st_folium(m, width=1000, height=450, returned_objects=[])

        st.subheader("Autonomous Actuation Triggers")
        a1, a2, a3 = st.columns(3)
        a1.error("🔊 High-Decibel Solar Sirens: ACTIVATED")
        a2.error("🚧 Perimeter Road Barriers: DEPLOYED")
        a3.error("📲 State DMA Webhook: DISPATCHED")
    else:
        st.info("ℹ️ Standby mode. Plume simulation and automated perimeter interventions execute only upon confirmed leak validation.")

with tab3:
    st.subheader("Predictive Maintenance Telemetry (Isolation Forest)")
    
    st.markdown("Monitoring continuous hardware health parameters to alert operators to sensor degradation prior to node failure.")
    
    node_v = st.slider("Node Battery Voltage (V)", 2.8, 4.2, 3.7)
    node_t = st.slider("Enclosure Temp (°C)", 20.0, 65.0, 28.0)
    node_h = st.slider("Internal Relative Humidity (%)", 20.0, 95.0, 45.0)
    
    sample_point = np.array([[node_v, node_t, node_h]])
    is_anomaly = maint_iso.predict(sample_point)[0] == -1
    
    if is_anomaly:
        st.error("🚨 Predictive Maintenance Warning: Node telemetry exhibits hardware degradation patterns (voltage sag / internal overheating).")
    else:
        st.success("✅ Node hardware telemetry operating within nominal tolerances.")

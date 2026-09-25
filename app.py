import streamlit as st
import folium
from streamlit_folium import st_folium
import math

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Edge AI Sensor Fusion Dashboard", layout="wide")
st.title("🏭 Smart Sentinel: Edge AI & Sensor Fusion Dashboard")
st.markdown("Real-time telemetry showing multi-modal AI (Acoustics + Vision + Chemical) preventing false alarms.")

# --- SIDEBAR: SCENARIO CONTROLS ---
st.sidebar.header("🧪 Simulation Controls")
scenario = st.sidebar.radio(
    "Select Live Scenario:",
    [
        "🟢 Normal Factory Operations",
        "⚠️ False Alarm: Air Hose (Ammonia)",
        "🚨 TRUE LEAK: Ammonia Pipe Crack",
        "⚠️ False Alarm: Water Spill (Styrene)",
        "🚨 TRUE LEAK: Styrene Liquid Spill"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🌤️ Weather & Dispersion")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 5.0, help="0 m/s triggers Puff Model. >1 triggers Plume Model.")
wind_direction = st.sidebar.slider("Wind Direction (°)", 0, 360, 90)

# --- SETTING SIMULATED SENSOR VALUES BASED ON SCENARIO ---
if scenario == "🟢 Normal Factory Operations":
    mic_freq = 15  # Low engine hum
    bos_shimmer = False
    liquid_spill = False
    ppm_level = 0
    chem_type = "None"
elif scenario == "⚠️ False Alarm: Air Hose (Ammonia)":
    mic_freq = 45  # High pitch hiss from harmless air hose
    bos_shimmer = False # No gas to bend light
    liquid_spill = False
    ppm_level = 0
    chem_type = "Ammonia (NH3)"
elif scenario == "🚨 TRUE LEAK: Ammonia Pipe Crack":
    mic_freq = 45  # Toxic gas hiss
    bos_shimmer = True # Gas refracting light
    liquid_spill = False
    ppm_level = 150 # High toxicity
    chem_type = "Ammonia (NH3)"
elif scenario == "⚠️ False Alarm: Water Spill (Styrene)":
    mic_freq = 12
    bos_shimmer = False # Water doesn't emit heavy vapor
    liquid_spill = True # Camera sees puddle
    ppm_level = 0 # No VOCs
    chem_type = "Styrene (VOC)"
else: # TRUE LEAK: Styrene
    mic_freq = 12
    bos_shimmer = True # VOC vapors shimmering
    liquid_spill = True # Camera sees puddle
    ppm_level = 45 # High VOCs
    chem_type = "Styrene (VOC)"

# --- AI SENSOR FUSION LOGIC (The "Random Forest" simulation) ---
# Check Ammonia Logic
if chem_type == "Ammonia (NH3)" or scenario == "🟢 Normal Factory Operations":
    # 1D-CNN Acoustic check (Hiss is 30-60 kHz)
    acoustic_trigger = True if 30 <= mic_freq <= 60 else False
    # Vision check
    vision_trigger = bos_shimmer
    # Chemical check
    chem_trigger = True if ppm_level > 25 else False
    
    # FUSION
    if acoustic_trigger and vision_trigger and chem_trigger:
        fusion_status = "CONFIRMED LEAK"
        fusion_color = "red"
    elif acoustic_trigger:
        fusion_status = "FALSE ALARM (Acoustic Only - Ignored)"
        fusion_color = "orange"
    else:
        fusion_status = "SYSTEM NORMAL"
        fusion_color = "green"

# Check Styrene Logic
else:
    acoustic_trigger = False # Not used for Styrene
    vision_trigger_spill = liquid_spill
    vision_trigger_vapor = bos_shimmer
    chem_trigger = True if ppm_level > 5 else False
    
    # FUSION
    if vision_trigger_spill and vision_trigger_vapor and chem_trigger:
        fusion_status = "CONFIRMED LEAK"
        fusion_color = "red"
    elif vision_trigger_spill:
        fusion_status = "FALSE ALARM (Water Spill - Ignored)"
        fusion_color = "orange"
    else:
        fusion_status = "SYSTEM NORMAL"
        fusion_color = "green"


# --- DASHBOARD UI ---
st.subheader("1. Edge Sensor Telemetry (Data Acquisition)")
col1, col2, col3 = st.columns(3)

# Mic
col1.metric("🎤 MEMS Microphone", f"{mic_freq} kHz", "1D-CNN Processing", delta_color="off")
if acoustic_trigger:
    col1.error("Anomalous Hiss Detected!")
else:
    col1.success("Normal Factory Noise")

# Camera
if chem_type == "Styrene (VOC)":
    cam_display = "Liquid Spill" if liquid_spill else "Clear Floor"
else:
    cam_display = "Shimmer Detected" if bos_shimmer else "No Shimmer"
col2.metric("📷 ESP32-CAM (Lucas-Kanade BOS)", cam_display, "Optical Flow Tracking", delta_color="off")
if bos_shimmer or (chem_type=="Styrene (VOC)" and liquid_spill):
    col2.warning("Visual Anomaly Triggered")
else:
    col2.success("Background Stable")

# Gas
col3.metric(f"👃 Electrochemical / PID", f"{ppm_level} PPM", "Toxicity Threshold", delta_color="off")
if chem_trigger:
    col3.error("Hazardous Spike!")
else:
    col3.success("Air Quality Safe")

st.markdown("---")

# --- SENSOR FUSION DECISION ---
st.subheader("2. AI Sensor Fusion (Random Forest Edge Decision)")
st.markdown(f"<h2 style='text-align: center; color: {fusion_color};'>[{fusion_status}]</h2>", unsafe_allow_html=True)

if fusion_color == "orange":
    st.info("💡 **Edge AI Action:** Cross-validation failed. The ML model identified this as a harmless factory event (e.g., compressed air hose or spilled water). It logged the event and returned the node to deep sleep, preventing a costly false evacuation.")
elif fusion_color == "green":
    st.success("💡 **Edge AI Action:** System in deep-sleep polling mode. No anomalies detected.")

st.markdown("---")

# --- PLUME MODEL & MAPPING ---
st.subheader("3. Dynamic Hazard Mapping (Gateway Plume Model)")

if fusion_status == "CONFIRMED LEAK":
    # Plume Math
    def calculate_plume_polygon(lat, lon, wind_dir, wind_spd):
        if wind_spd < 1.0:
            return None, 300 # Puff model radius
        
        plume_length_meters = (wind_spd * 150) + 200
        length_deg = plume_length_meters / 111000.0
        plume_dir_rad = math.radians((wind_dir + 180) % 360)
        spread_angle = math.radians(45 / (wind_spd * 0.5 + 1)) 
        
        p1 = [lat, lon]
        p2 = [lat + length_deg * math.cos(plume_dir_rad - spread_angle), lon + length_deg * math.sin(plume_dir_rad - spread_angle)]
        p3 = [lat + length_deg * 1.1 * math.cos(plume_dir_rad), lon + length_deg * 1.1 * math.sin(plume_dir_rad)]
        p4 = [lat + length_deg * math.cos(plume_dir_rad + spread_angle), lon + length_deg * math.sin(plume_dir_rad + spread_angle)]
        return [p1, p2, p3, p4], None

    FACTORY_LAT, FACTORY_LON = 22.7533, 75.8937
    m = folium.Map(location=[FACTORY_LAT, FACTORY_LON], zoom_start=15)
    
    folium.Marker([FACTORY_LAT, FACTORY_LON], popup="Sensor Gateway", icon=folium.Icon(color="black", icon="info-sign")).add_to(m)
    
    polygon_coords, puff_radius = calculate_plume_polygon(FACTORY_LAT, FACTORY_LON, wind_direction, wind_speed)
    
    if puff_radius:
        folium.Circle(location=[FACTORY_LAT, FACTORY_LON], radius=puff_radius, color="red", fill=True, fill_opacity=0.4, popup="Gaussian Puff").add_to(m)
        st.caption("🌬️ Wind is zero. Running **Gaussian Puff Model** (Circular Expansion).")
    else:
        folium.Polygon(locations=polygon_coords, color="red", fill=True, fill_opacity=0.4, popup="Gaussian Plume").add_to(m)
        st.caption(f"🌬️ Wind is {wind_speed} m/s. Running **Gaussian Plume Model** (Directional Cone).")
    
    st_data = st_folium(m, width=1000, height=400)
    
    # Mitigation Alerts
    st.error("🚨 **AUTOMATED RESPONSES TRIGGERED VIA LORA MESH:**")
    c1, c2, c3 = st.columns(3)
    c1.error("🔊 Solar Sirens: ACTIVE")
    c2.error("🚧 Highway Barriers: CLOSED")
    c3.error("📲 SDMA SMS: DISPATCHED")

else:
    st.write("💤 Gateway in standby mode. No confirmed hazards to map.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/1x1.png/120px-1x1.png", width=1) # spacer

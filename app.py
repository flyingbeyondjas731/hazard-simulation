import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import numpy as np

# --- UI SETTINGS ---
st.set_page_config(layout="wide")
st.title("Dual-Hazard System: Live Dispersion Simulation")
st.markdown("Dynamic hazard zone prediction based on edge-sensor telemetry.")

# --- SIDEBAR CONTROLS (Simulating Sensor Inputs) ---
st.sidebar.header("Live Sensor Telemetry")
chemical_type = st.sidebar.selectbox("Chemical Detected", ["Ammonia (NH3)", "Styrene (VOC)"])
leak_rate = st.sidebar.slider("Leak Severity (kg/s)", min_value=1.0, max_value=50.0, value=10.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=5.0)
wind_direction = st.sidebar.slider("Wind Direction (Degrees)", min_value=0, max_value=360, value=90)

# Factory Coordinates (Example: An industrial area in India)
FACTORY_LAT = 22.7533
FACTORY_LON = 75.8937

# --- PLUME CALCULATION MATH ---
# In a real scenario, this uses the Pasquill-Gifford Gaussian equation. 
# For this UI prototype, we calculate a dynamic geometric footprint.

def calculate_plume_polygon(lat, lon, wind_dir, wind_spd, leak_rt):
    # If wind is zero, it's a puff (circle)
    if wind_spd < 1.0:
        radius = leak_rt * 10  # roughly scale radius by leak severity
        return None, radius
    
    # Calculate cone length based on wind speed and leak rate
    plume_length_meters = (leak_rt * 50) + (wind_spd * 100) 
    
    # Convert meters to degrees (roughly 1 degree = 111,000 meters)
    length_deg = plume_length_meters / 111000.0
    
    # Wind direction means wind blows FROM that direction, so plume goes OPPOSITE
    plume_dir_rad = math.radians((wind_dir + 180) % 360)
    
    # Plume spread angle (faster wind = narrower cone)
    spread_angle = math.radians(45 / (wind_spd * 0.5 + 1)) 
    
    # Calculate polygon vertices
    point1 = [lat, lon]
    point2 = [
        lat + length_deg * math.cos(plume_dir_rad - spread_angle),
        lon + length_deg * math.sin(plume_dir_rad - spread_angle)
    ]
    point3 = [
        lat + length_deg * 1.1 * math.cos(plume_dir_rad), # Rounded tip
        lon + length_deg * 1.1 * math.sin(plume_dir_rad)
    ]
    point4 = [
        lat + length_deg * math.cos(plume_dir_rad + spread_angle),
        lon + length_deg * math.sin(plume_dir_rad + spread_angle)
    ]
    
    return [point1, point2, point3, point4], None

# --- MAPPING ---
# Initialize the map centered on the factory
m = folium.Map(location=[FACTORY_LAT, FACTORY_LON], zoom_start=14)

# Add Factory Marker
folium.Marker(
    [FACTORY_LAT, FACTORY_LON], 
    popup="Industrial Plant (Sensor Gateway)",
    icon=folium.Icon(color="black", icon="info-sign")
).add_to(m)

# Generate and add the plume shape
polygon_coords, puff_radius = calculate_plume_polygon(
    FACTORY_LAT, FACTORY_LON, wind_direction, wind_speed, leak_rate
)

# Render Plume (Cone) or Puff (Circle) based on wind
plume_color = "red" if chemical_type == "Ammonia (NH3)" else "purple"

if puff_radius:
    # Zero wind scenario
    folium.Circle(
        location=[FACTORY_LAT, FACTORY_LON],
        radius=puff_radius,
        color=plume_color,
        fill=True,
        fill_opacity=0.4,
        popup="Zero-Wind Plume Accumulation"
    ).add_to(m)
else:
    # Steady wind scenario
    folium.Polygon(
        locations=polygon_coords,
        color=plume_color,
        fill=True,
        fill_opacity=0.4,
        popup="Predicted Hazard Zone"
    ).add_to(m)

# Display the map in the Streamlit app
st_data = st_folium(m, width=900, height=500)

# --- ACTIONABLE OUTPUTS UI ---
st.markdown("### Automated Emergency Actions Triggred:")
col1, col2, col3 = st.columns(3)
if leak_rate > 20:
    col1.error("🚨 Evacuation Sirens: ACTIVE")
    col2.error("🚧 Highway Barriers: CLOSED")
    col3.error("📲 SDMA SMS Dispatch: SENT")
else:
    col1.warning("⚠️ Evacuation Sirens: STANDBY")
    col2.warning("🚧 Highway Barriers: OPEN")
    col3.warning("📲 SDMA SMS Dispatch: ADVISORY LOGGED")

# 🏭 Smart Sentinel: Multi-Modal Edge AI Chemical Leak Detection

An industrial early-warning safety system that combines multi-modal Edge AI (acoustics, optical refraction tracking, and chemical sensing) with real-time atmospheric dispersion modeling to detect toxic chemical leaks (Ammonia and Styrene) while eliminating costly false alarms.

**WORKING**
* **Dual-Trigger Wake-Up:** For pressurized Ammonia, ultrasonic microphones listen for high-frequency hisses to wake the system. For volatile Styrene, duty-cycled cameras periodically wake to scan the floor for liquid spills. 
* **Optical Shimmer Verification:** Standard cameras utilize Background-Oriented Schlieren (BOS) AI to detect the invisible "shimmer" (light refraction) of escaping gas or evaporating vapors.
* **Strict Sensor Fusion:** An edge-deployed Random Forest classifier acts as a strict AND-gate. It only triggers a confirmed leak if the acoustic/spill trigger, the optical gas shimmer, and the chemical sensor (PPM spike) all corroborate the event. 
* **Dynamic Hazard Mapping:** Once a leak is confirmed, the system pulls live anemometer data to dynamically plot a Gaussian Plume (directional cone) or Gaussian Puff (expanding circle) hazard map, automatically triggering plant mitigation systems.

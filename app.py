import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Optional interactive 3D-ish visuals
import plotly.graph_objects as go

# --- 0. LOAD CUSTOM STYLING (NO CHANGE TO NAME/INFO) ---
def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("assets/style.css")

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="RF Backscatter Demo", layout="wide")
st.title("📡 Ambient RF Backscatter: Live Sensor Data Demo")
st.markdown("Simulating a battery-less sensor sending real-world data by reflecting Wi-Fi signals.")

# --- 2. TEXT TO BINARY CONVERTERS ---
def text_to_binary(text):
    """Converts a string of text into a binary string (0s and 1s)."""
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary_str):
    """Converts a binary string back into readable text."""
    try:
        chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
        return ''.join(chars)
    except:
        return "[Decoding Error]"

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Controls")
noise_level = st.sidebar.slider("Ambient Wi-Fi Noise Level", 0.1, 2.0, 0.5)
signal_strength = st.sidebar.slider("Backscatter Reflection Strength", 0.05, 0.5, 0.2)
sensor_data = st.sidebar.text_input("Simulated Sensor Reading", "24C") # Default text is 24C

# --- 4. MATHEMATICAL SIMULATION ---
@st.cache_data
def generate_rf_signal(bits, noise, strength):
    fs = 100000  # Sample rate (Hz)
    t_bit = 0.01 # Duration of one bit

    t = np.array([])
    rf_signal = np.array([])

    for i, bit in enumerate(bits):
        time_array = np.linspace(i*t_bit, (i+1)*t_bit, int(fs*t_bit), endpoint=False)
        t = np.append(t, time_array)

        # Simulated Ambient Wi-Fi Carrier + Noise
        ambient_wave = np.sin(2 * np.pi * 5000 * time_array) + np.random.normal(0, noise, len(time_array))

        # Backscatter modulation
        if bit == '1':
            backscatter = strength * np.sin(2 * np.pi * 10000 * time_array)
            rf_signal = np.append(rf_signal, ambient_wave + backscatter)
        else:
            rf_signal = np.append(rf_signal, ambient_wave)

    return t, rf_signal, fs, t_bit

# --- 5. DIGITAL SIGNAL PROCESSING (DSP) ---
def decode_signal(rf_signal, fs):
    # Bandpass filter
    nyq = 0.5 * fs
    low = 9000 / nyq
    high = 11000 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, rf_signal)

    # Envelope detection
    envelope = np.abs(filtered_signal)
    b_low, a_low = butter(4, 500 / nyq, btype='low')
    smooth_envelope = filtfilt(b_low, a_low, envelope)

    return smooth_envelope

def extract_bits(envelope, fs, t_bit, threshold):
    """Reads the DSP envelope and decides if each bit is a 1 or 0."""
    samples_per_bit = int(fs * t_bit)
    decoded_bits = ""

    # Check the average signal height for each bit's time slot
    for i in range(len(envelope) // samples_per_bit):
        chunk = envelope[i * samples_per_bit : (i + 1) * samples_per_bit]
        if np.mean(chunk) > threshold:
            decoded_bits += "1"
        else:
            decoded_bits += "0"
    return decoded_bits

# --- 6. RUNNING THE PIPELINE ---
if len(sensor_data) > 8:
    st.warning("Keep the text short (under 8 characters) so the simulation runs fast!")
elif len(sensor_data) == 0:
    st.error("Please enter some sensor data.")
else:
    # 1. Convert text to binary
    bits_to_send = text_to_binary(sensor_data)
    st.write(f"**Transmitting Binary:** `{bits_to_send}` (Length: {len(bits_to_send)} bits)")

    # 2. Generate and process the data
    time_axis, raw_rf, fs, t_bit = generate_rf_signal(bits_to_send, noise_level, signal_strength)
    envelope = decode_signal(raw_rf, fs)

    # (Styling-only) quick metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sample Rate (Hz)", f"{fs}")
    m2.metric("Bit Duration (s)", f"{t_bit:.3f}")
    m3.metric("Noise Level", f"{noise_level:.2f}")
    m4.metric("Reflection Strength", f"{signal_strength:.2f}")

    # 3. Visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Raw Antenna Feed (Noisy Wi-Fi)")
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(time_axis, raw_rf, color='grey', alpha=0.7)
        ax1.set_xlabel("Time (s)")
        ax1.grid(alpha=0.2)
        st.pyplot(fig1, use_container_width=True)

    with col2:
        st.subheader("2. DSP Output (Envelope Detection)")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(time_axis, envelope, color='green', linewidth=2)
        threshold = np.mean(envelope)
        ax2.axhline(threshold, color='red', linestyle='--', label="Detection Threshold")
        ax2.set_xlabel("Time (s)")
        ax2.grid(alpha=0.2)
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

    # 3D / interactive “effect”
    with st.expander("✨ 3D / Interactive View (Stylish)", expanded=True):
        stride = max(1, len(time_axis) // 2500)
        x = time_axis[::stride]
        y1 = raw_rf[::stride]
        y2 = envelope[::stride]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y1, z=np.zeros_like(x),
            mode="lines", name="Raw RF",
            line=dict(width=4, color="rgba(200,200,200,0.85)")
        ))
        fig.add_trace(go.Scatter3d(
            x=x, y=y2, z=np.ones_like(x) * 0.5,
            mode="lines", name="Envelope",
            line=dict(width=6, color="rgba(0,255,160,0.9)")
        ))
        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                zaxis_title="Layer",
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 4. Extract bits and convert back to text
    received_bits = extract_bits(envelope, fs, t_bit, threshold)
    decoded_text = binary_to_text(received_bits)

    # --- 7. FINAL DECODED OUTPUT ---
    st.subheader("3. Final Receiver Output")

    # Check for bit errors
    if sensor_data == decoded_text:
        st.success(f"✅ Successfully Decoded Sensor Data: **{decoded_text}**")
    else:
        st.error(f"❌ Bit Error! Too much noise. Decoded text looks corrupted: **{decoded_text}**")
        st.write(f"Received Binary: `{received_bits}`")

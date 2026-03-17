import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Optional interactive 3D-ish visuals
import plotly.graph_objects as go

# --- 0. LOAD SURREAL STYLING & JS ---
def load_surreal_ui():
    st.markdown(
        """
        <style>
        /* 1. TYPOGRAPHY IMPORTS */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;500;600&family=JetBrains+Mono:wght@400;700&family=Syne:wght@500;700;800&display=swap');

        /* 2. THEME VARIABLES (CSS VARS) */
        :root, [data-theme="aurora"] {
            --bg-base: #050d1a;
            --bg-panel: rgba(13, 18, 36, 0.65);
            --accent-1: #00c896;
            --accent-2: #00e5ff;
            --accent-3: #7c3aed;
            --text-main: #f0fdfa;
            --text-muted: #94a3b8;
            --glow-1: rgba(0, 200, 150, 0.25);
            --glow-2: rgba(0, 229, 255, 0.2);
            --glass-border: rgba(255,255,255,0.06);
            --glass-border-hi: rgba(0, 200, 150, 0.3);
        }
        
        [data-theme="dark-void"] {
            --bg-base: #0a0a0f;
            --bg-panel: rgba(15, 15, 22, 0.7);
            --accent-1: #00f5ff;
            --accent-2: #7c3aed;
            --accent-3: #ff007f;
            --text-main: #ffffff;
            --text-muted: #a1a1aa;
            --glow-1: rgba(0, 245, 255, 0.25);
            --glow-2: rgba(124, 58, 237, 0.2);
            --glass-border: rgba(255,255,255,0.08);
            --glass-border-hi: rgba(0, 245, 255, 0.3);
        }

        [data-theme="knight"] {
            --bg-base: #0d0d0d;
            --bg-panel: rgba(20, 20, 20, 0.8);
            --accent-1: #c9a84c;
            --accent-2: #8b0000;
            --accent-3: #ff4500;
            --text-main: #f5f5f5;
            --text-muted: #888888;
            --glow-1: rgba(201, 168, 76, 0.25);
            --glow-2: rgba(139, 0, 0, 0.2);
            --glass-border: rgba(201, 168, 76, 0.1);
            --glass-border-hi: rgba(201, 168, 76, 0.4);
        }

        [data-theme="light-editorial"] {
            --bg-base: #faf8f5;
            --bg-panel: rgba(255, 255, 255, 0.7);
            --accent-1: #d47c0f;
            --accent-2: #1a1a1a;
            --accent-3: #8b4513;
            --text-main: #1a1a1a;
            --text-muted: #555555;
            --glow-1: rgba(212, 124, 15, 0.15);
            --glow-2: rgba(26, 26, 26, 0.1);
            --glass-border: rgba(0,0,0,0.05);
            --glass-border-hi: rgba(212, 124, 15, 0.3);
        }

        [data-theme="ember"] {
            --bg-base: #1a1008;
            --bg-panel: rgba(36, 22, 12, 0.75);
            --accent-1: #ff6b1a;
            --accent-2: #cc2200;
            --accent-3: #ffb703;
            --text-main: #fff0e6;
            --text-muted: #bda18e;
            --glow-1: rgba(255, 107, 26, 0.25);
            --glow-2: rgba(204, 34, 0, 0.2);
            --glass-border: rgba(255, 107, 26, 0.15);
            --glass-border-hi: rgba(255, 107, 26, 0.4);
        }

        /* Smooth crossfade on root */
        body {
            transition: background 0.3s ease, color 0.3s ease;
            /* Default cursor remains visible, augmented by JS effects */
        }

        /* 3. CORE GLOBAL OVERRIDES */
        .stApp {
            background-color: var(--bg-base);
            font-family: 'DM Sans', sans-serif;
            color: var(--text-main);
        }
        
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Syne', sans-serif !important;
            letter-spacing: -0.02em;
            font-weight: 700;
        }

        /* 4. BACKGROUND ATMOSPHERE (BLOBS & GRAIN) */
        .stApp::before {
            content: "";
            position: fixed;
            inset: -50%;
            background: 
                radial-gradient(circle at 20% 30%, var(--glow-1) 0%, transparent 40%),
                radial-gradient(circle at 80% 70%, var(--glow-2) 0%, transparent 45%),
                radial-gradient(circle at 50% 10%, rgba(124, 58, 237, 0.08) 0%, transparent 50%);
            filter: blur(90px);
            z-index: -2;
            animation: driftBlobs 40s infinite alternate ease-in-out;
            pointer-events: none;
            transition: background 0.5s ease;
        }
        
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            opacity: 0.04;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            pointer-events: none;
            z-index: -1;
        }

        @keyframes driftBlobs {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(4%, 6%) scale(1.05); }
            100% { transform: translate(-3%, 2%) scale(0.95); }
        }

        /* 5. SIDEBAR */
        section[data-testid="stSidebar"] {
            background: var(--bg-panel);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
            transition: background 0.3s ease, border 0.3s ease;
        }

        /* 6. METRICS & CARDS WITH 3D TILT HOVER */
        div[data-testid="stVerticalBlockBorderWrapper"], 
        div[data-testid="metric-container"] {
            background: var(--bg-panel);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,255,255,0.02);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease, border-color 0.3s ease;
            padding: 1rem;
            transform-style: preserve-3d;
            perspective: 800px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover,
        div[data-testid="metric-container"]:hover {
            transform: translateY(-6px) translateZ(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 40px var(--glow-1);
            border-color: var(--glass-border-hi);
        }

        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* 7. BUTTONS (Magnetic targets via JS class later) */
        .stButton > button, 
        [data-testid="stDownloadButton"] button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
            color: #fff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 8px 20px var(--glow-1);
            transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:hover, 
        [data-testid="stDownloadButton"] button:hover {
            transform: scale(1.04);
            box-shadow: 0 12px 30px var(--glow-1), 0 0 20px var(--accent-1);
        }
        
        .stButton > button:active, 
        [data-testid="stDownloadButton"] button:active {
            transform: scale(0.97);
        }

        /* 8. INPUTS */
        .stTextInput input, .stSelectbox > div > div {
            background: rgba(0,0,0,0.2) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 10px;
            color: var(--text-main) !important;
            transition: all 0.25s ease !important;
        }
        
        .stTextInput input:focus, .stSelectbox > div > div:focus {
            border-color: var(--accent-1) !important;
            box-shadow: 0 0 0 2px var(--glow-1) !important;
        }

        /* 9. PAGE LOAD STAGGERED ENTRANCE */
        .main [data-testid="stVerticalBlock"] > * {
            animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .main [data-testid="stVerticalBlock"] > *:nth-child(1) { animation-delay: 0.1s; }
        .main [data-testid="stVerticalBlock"] > *:nth-child(2) { animation-delay: 0.2s; }
        .main [data-testid="stVerticalBlock"] > *:nth-child(3) { animation-delay: 0.3s; }
        .main [data-testid="stVerticalBlock"] > *:nth-child(4) { animation-delay: 0.4s; }
        .main [data-testid="stVerticalBlock"] > *:nth-child(5) { animation-delay: 0.5s; }

        @keyframes slideUpFade {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* 10. CUSTOM CURSOR DOM ELEMENTS STYLING (Injected via JS) */
        #custom-cursor-dot {
            position: fixed;
            top: 0; left: 0;
            width: 12px; height: 12px;
            background: var(--accent-1);
            border-radius: 50%;
            pointer-events: none;
            z-index: 99999;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 15px var(--accent-1);
            transition: width 0.2s, height 0.2s, background 0.2s;
            mix-blend-mode: screen;
        }
        #custom-cursor-dot.hover-active {
            width: 40px; height: 40px;
            background: transparent;
            border: 2px solid var(--accent-2);
            box-shadow: 0 0 20px var(--glow-2), inset 0 0 10px var(--glow-2);
        }
        
        .cursor-trail {
            position: fixed;
            top: 0; left: 0;
            width: 6px; height: 6px;
            background: var(--accent-2);
            border-radius: 50%;
            pointer-events: none;
            z-index: 99998;
            transform: translate(-50%, -50%);
            opacity: 0.6;
        }

        .click-ripple {
            position: fixed;
            border-radius: 50%;
            border: 2px solid var(--accent-1);
            pointer-events: none;
            z-index: 99997;
            animation: ripple-anim 0.5s cubic-bezier(0.1, 0.8, 0.3, 1) forwards;
            transform: translate(-50%, -50%);
        }

        @keyframes ripple-anim {
            0% { width: 0; height: 0; opacity: 1; }
            100% { width: 100px; height: 100px; opacity: 0; }
        }

        /* 11. THEME SWITCHER BUTTON */
        #theme-switcher-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100000;
            background: var(--bg-panel);
            border: 1px solid var(--glass-border-hi);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-family: 'DM Sans', sans-serif;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            transition: all 0.2s ease;
        }
        #theme-switcher-btn:hover {
            background: var(--accent-1);
            color: #000;
            transform: scale(1.05);
        }
        </style>

        <button id="theme-switcher-btn" onclick="toggleTheme()">✨ Theme: Dark Void</button>

        <script>
        // ── JS: THEME SWITCHING (DARK / LIGHT) ──
        const themes = ["dark-void", "light-editorial"];
        let currentThemeIdx = 0;
        
        // Give stApp a min height so we can scroll and cursor works everywhere
        document.querySelector('.stApp').style.minHeight = '100vh';

        // Wait a slight moment for DOM to settle, then set initial theme
        setTimeout(() => {
            const root = document.documentElement;
            root.setAttribute('data-theme', themes[currentThemeIdx]);
            // Also override Streamlit's iframe body explicitly if needed
            document.body.setAttribute('data-theme', themes[currentThemeIdx]);
        }, 100);

        window.toggleTheme = function() {
            currentThemeIdx = (currentThemeIdx + 1) % themes.length;
            const newTheme = themes[currentThemeIdx];
            document.documentElement.setAttribute('data-theme', newTheme);
            document.body.setAttribute('data-theme', newTheme);
            
            const btn = document.getElementById('theme-switcher-btn');
            const prettyName = newTheme.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            btn.innerText = `✨ Theme: ${prettyName}`;
        };

        // ── JS: CUSTOM CURSOR & TRAIL ──
        (function initCursor() {
            if(document.getElementById('custom-cursor-dot')) return; // prevent dupes
            
            const dot = document.createElement('div');
            dot.id = 'custom-cursor-dot';
            document.body.appendChild(dot);

            const trails = [];
            const NUM_TRAILS = 8;
            for(let i = 0; i < NUM_TRAILS; i++) {
                const t = document.createElement('div');
                t.className = 'cursor-trail';
                t.style.opacity = (1 - (i/NUM_TRAILS)) * 0.6;
                t.style.scale = 1 - (i/NUM_TRAILS)*0.5;
                document.body.appendChild(t);
                trails.push({ el: t, x: 0, y: 0 });
            }

            let mouseX = window.innerWidth/2;
            let mouseY = window.innerHeight/2;
            let dotX = mouseX, dotY = mouseY;

            // Track mouse
            window.addEventListener('mousemove', (e) => {
                mouseX = e.clientX;
                mouseY = e.clientY;
                
                // Active state hover check
                const target = e.target.closest('button, input, select, .stMarkdown, [data-testid="stMetric"]');
                if(target) {
                    dot.classList.add('hover-active');
                } else {
                    dot.classList.remove('hover-active');
                }
            });

            // Click shockwave
            window.addEventListener('mousedown', (e) => {
                const ripple = document.createElement('div');
                ripple.className = 'click-ripple';
                ripple.style.left = e.clientX + 'px';
                ripple.style.top = e.clientY + 'px';
                document.body.appendChild(ripple);
                setTimeout(() => ripple.remove(), 550);
            });

            // Animation loop for smooth trailing
            function animate() {
                // Lerp dot
                dotX += (mouseX - dotX) * 0.3;
                dotY += (mouseY - dotY) * 0.3;
                dot.style.transform = `translate(${dotX}px, ${dotY}px)`;

                // Cascade trail
                let tx = dotX;
                let ty = dotY;
                for(let i=0; i < NUM_TRAILS; i++) {
                    const tr = trails[i];
                    tr.x += (tx - tr.x) * 0.4;
                    tr.y += (ty - tr.y) * 0.4;
                    tr.el.style.transform = `translate(${tr.x}px, ${tr.y}px)`;
                    tx = tr.x;
                    ty = tr.y;
                }
                requestAnimationFrame(animate);
            }
            requestAnimationFrame(animate);
        })();

        // ── JS: MAGNETIC BUTTONS & 3D TILT ──
        setTimeout(() => {
            const buttons = document.querySelectorAll('.stButton > button, #theme-switcher-btn');
            buttons.forEach(btn => {
                btn.addEventListener('mousemove', (e) => {
                    const rect = btn.getBoundingClientRect();
                    const x = e.clientX - rect.left - rect.width/2;
                    const y = e.clientY - rect.top - rect.height/2;
                    // Magnetic pull
                    btn.style.transform = `translate(${x*0.2}px, ${y*0.2}px) scale(1.05)`;
                });
                btn.addEventListener('mouseleave', () => {
                    btn.style.transform = '';
                });
            });

            const cards = document.querySelectorAll('div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="metric-container"]');
            cards.forEach(card => {
                card.addEventListener('mousemove', (e) => {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const rotX = -((y - centerY) / centerY) * 8; // max 8 deg
                    const rotY = ((x - centerX) / centerX) * 8;
                    
                    card.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-6px)`;
                    card.style.transition = 'none'; // remove trans so it tracks instantly
                });
                card.addEventListener('mouseleave', () => {
                    card.style.transition = 'transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
                    card.style.transform = ''; // reset to default
                });
            });
        }, 1500); // wait for streamlit DOM to build
        </script>
        """,
        unsafe_allow_html=True
    )

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="RF Backscatter Demo", layout="wide")

load_surreal_ui()

# Stylish animated header wrapper (NO CHANGE to your title/info text)
st.markdown(
    """
    <div class="hero">
      <div class="hero__scanline"></div>
      <div class="hero__glow"></div>
      <div class="hero__content">
    """,
    unsafe_allow_html=True,
)

st.title("📡 Ambient RF Backscatter: Live Sensor Data Demo")
st.markdown("Simulating a battery-less sensor sending real-world data by reflecting Wi-Fi signals.")

st.markdown("</div></div>", unsafe_allow_html=True)

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

st.sidebar.divider()
enable_3d = st.sidebar.toggle("Enable 3D / Interactive View", value=True)
enable_downloads = st.sidebar.toggle("Enable Downloads", value=True)

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

# --- Matplotlib neon styling helper (visual only) ---
def style_neon_axes(ax, face="#04050D"):
    ax.set_facecolor(face)
    fig = ax.get_figure()
    fig.patch.set_facecolor("#04050D")
    for spine in ax.spines.values():
        spine.set_color((0.6, 0.85, 1.0, 0.15))
    ax.tick_params(colors=(0.72, 0.90, 1.0, 0.80))
    ax.xaxis.label.set_color((0.72, 0.90, 1.0, 0.80))
    ax.yaxis.label.set_color((0.72, 0.90, 1.0, 0.80))
    ax.grid(alpha=0.12, linestyle="--", color="#00E5FF")

def glow_line(ax, x, y, color, base_lw=1.8):
    # fake "glow" by plotting thicker transparent lines under the main line
    for lw, a in [(10, 0.05), (7, 0.07), (5, 0.10)]:
        ax.plot(x, y, color=color, linewidth=lw, alpha=a)
    ax.plot(x, y, color=color, linewidth=base_lw, alpha=0.95)

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

    # 3. Visualization
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sample Rate (Hz)", f"{fs}")
    m2.metric("Bit Duration (s)", f"{t_bit:.3f}")
    m3.metric("Noise Level", f"{noise_level:.2f}")
    m4.metric("Reflection Strength", f"{signal_strength:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Raw Antenna Feed (Noisy Wi-Fi)")
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        style_neon_axes(ax1)
        glow_line(ax1, time_axis, raw_rf, color="#00E5FF", base_lw=1.4)
        ax1.set_xlabel("Time (s)")
        st.pyplot(fig1, use_container_width=True)

    with col2:
        st.subheader("2. DSP Output (Envelope Detection)")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        style_neon_axes(ax2)
        glow_line(ax2, time_axis, envelope, color="#A78BFA", base_lw=2.0)
        threshold = np.mean(envelope)
        ax2.axhline(threshold, color="#FF4D8B", linestyle="--", linewidth=1.6, alpha=0.9, label="Detection Threshold")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

    # 4. Extract bits and convert back to text
    received_bits = extract_bits(envelope, fs, t_bit, threshold)
    decoded_text = binary_to_text(received_bits)

    # 5. 3D / interactive “effect” (toggle)
    if enable_3d:
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
                line=dict(width=6, color="rgba(167,139,250,0.95)")
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

    # --- 7. FINAL DECODED OUTPUT ---
    st.subheader("3. Final Receiver Output")

    # Check for bit errors
    if sensor_data == decoded_text:
        st.success(f"✅ Successfully Decoded Sensor Data: **{decoded_text}**")
    else:
        st.error(f"❌ Bit Error! Too much noise. Decoded text looks corrupted: **{decoded_text}**")
        st.write(f"Received Binary: `{received_bits}`")

    # Downloads (toggle)
    if enable_downloads:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download Received Binary",
                data=received_bits.encode("utf-8"),
                file_name="received_bits.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download Decoded Text",
                data=decoded_text.encode("utf-8"),
                file_name="decoded_text.txt",
                mime="text/plain",
                use_container_width=True,
            )

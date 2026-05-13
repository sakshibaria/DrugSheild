import streamlit as st
import pandas as pd
import os
import time
import tempfile

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="DrugShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "image_history" not in st.session_state:
    st.session_state.image_history = []

if "text_history" not in st.session_state:
    st.session_state.text_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ------------------------------------------------
# GLOBAL CSS
# ------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #e2e8f0 !important;
}
.stApp {
    background: linear-gradient(160deg, #0a0a0f 0%, #0d0d1a 40%, #0f0a1a 100%);
}
header, [data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: transparent !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #150d25 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown label {
    color: #c4b5fd !important;
}

/* ---- Typography ---- */
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, span, label, div, li { color: #cbd5e1 !important; }

/* ---- Glassmorphism Cards ---- */
.glass-card {
    background: rgba(15, 15, 30, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
    transform: translateY(-2px);
}

/* ---- Stat Cards ---- */
.stat-card {
    background: rgba(15, 15, 30, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-card:hover {
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.15);
    transform: translateY(-3px);
}
.stat-number {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
}
.stat-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8 !important;
    font-weight: 600;
}

/* ---- Drug / Alert Cards ---- */
.stat-card.drug {
    border-color: rgba(239, 68, 68, 0.4);
    background: rgba(239, 68, 68, 0.08);
}
.stat-card.drug .stat-number {
    background: linear-gradient(135deg, #ef4444, #f87171);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card.safe {
    border-color: rgba(34, 197, 94, 0.4);
    background: rgba(34, 197, 94, 0.08);
}
.stat-card.safe .stat-number {
    background: linear-gradient(135deg, #22c55e, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card.alert {
    border-color: rgba(245, 158, 11, 0.4);
    background: rgba(245, 158, 11, 0.08);
}
.stat-card.alert .stat-number {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- Results ---- */
.result-drug {
    background: rgba(239, 68, 68, 0.1);
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 8px 0;
}
.result-safe {
    background: rgba(34, 197, 94, 0.1);
    border-left: 4px solid #22c55e;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 8px 0;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 48px;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
    box-shadow: 0 6px 25px rgba(124, 58, 237, 0.5) !important;
    transform: translateY(-1px);
}

/* ---- Inputs ---- */
.stTextInput input, textarea, .stSelectbox select {
    background-color: rgba(15, 15, 30, 0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.stTextInput input:focus, textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

/* ---- File Uploader ---- */
[data-testid="stFileUploader"] {
    background: rgba(15, 15, 30, 0.5);
    border: 2px dashed rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 20px;
}

/* ---- Divider ---- */
hr {
    border-color: rgba(139, 92, 246, 0.15) !important;
}

/* ---- Metric styling ---- */
[data-testid="stMetric"] {
    background: rgba(15, 15, 30, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 16px;
}

/* ---- Hero glow ---- */
.hero-title {
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc, #8b5cf6, #6d28d9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 8px;
    line-height: 1.2;
}
.hero-subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 18px;
    font-weight: 400;
    margin-bottom: 40px;
}

/* ---- Fade animation ---- */
.fade-in {
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn {
    0% { opacity:0; transform: translateY(10px); }
    100% { opacity:1; transform: translateY(0); }
}

/* ---- Badge ---- */
.badge-drug {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.badge-safe {
    display: inline-block;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.badge-suspicious {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.badge-normal {
    display: inline-block;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}

/* Confidence bar */
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
}
.confidence-bar-fill {
    height: 8px;
    border-radius: 10px;
    transition: width 0.5s ease;
}
.confidence-bar-fill.drug {
    background: linear-gradient(90deg, #dc2626, #ef4444);
}
.confidence-bar-fill.safe {
    background: linear-gradient(90deg, #16a34a, #22c55e);
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# LOGIN PAGE
# ------------------------------------------------

def login():
    st.markdown("<div style='height: 80px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 2, 1.5])

    with col2:
        st.markdown("""
        <div class='glass-card' style='padding: 40px 36px; text-align: center;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>🛡️</div>
            <h1 style='font-size: 32px; margin-bottom: 4px;'>DrugShield AI</h1>
            <p style='color: #94a3b8 !important; margin-bottom: 28px;'>Unified Forensic Detection Platform</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔐  Login", use_container_width=True):
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Use admin / admin")


# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------

def sidebar_nav():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='font-size: 36px;'>🛡️</div>
            <h2 style='font-size: 22px; margin: 8px 0 4px 0;
                background: linear-gradient(135deg, #c084fc, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;'>DrugShield AI</h2>
            <p style='font-size: 12px; color: #64748b !important; letter-spacing: 1px;
                text-transform: uppercase;'>Forensic Detection Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p style='font-size:12px; letter-spacing:1.5px; color:#64748b !important; "
                     "text-transform:uppercase; font-weight:600;'>Navigation</p>",
                     unsafe_allow_html=True)

        if st.button("🏠  Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

        if st.button("🖼️  Image Analysis", use_container_width=True):
            st.session_state.page = "Image Analysis"
            st.rerun()

        if st.button("📝  Text Analysis", use_container_width=True):
            st.session_state.page = "Text Analysis"
            st.rerun()

        if st.button("📊  Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()

        st.markdown("---")

        # Stats summary
        total_img = len(st.session_state.image_history)
        total_txt = len(st.session_state.text_history)
        st.markdown(f"""
        <div style='padding:12px; background:rgba(139,92,246,0.08); border-radius:12px;
            border:1px solid rgba(139,92,246,0.15);'>
            <p style='font-size:12px; color:#8b5cf6 !important; font-weight:600;
                letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Session Stats</p>
            <p style='font-size:14px; margin:4px 0;'>🖼️ Images Analyzed: <b>{total_img}</b></p>
            <p style='font-size:14px; margin:4px 0;'>📝 Texts Analyzed: <b>{total_txt}</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.rerun()


# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

def home_page():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; padding: 40px 0 10px 0;'>
        <div style='font-size: 64px; margin-bottom: 12px;'>🛡️</div>
        <div class='hero-title'>DrugShield AI</div>
        <div class='hero-subtitle'>Unified Forensic Drug Detection Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:32px; margin-bottom:12px;'>🖼️</div>
            <h3 style='font-size:20px; margin-bottom:8px;'>Image Analysis</h3>
            <p style='color:#94a3b8 !important; font-size:14px; line-height:1.6;'>
                Upload images for automated drug detection using <b>VGG16 deep learning</b>
                feature extraction combined with a trained neural network classifier.
                Supports batch upload of multiple images with risk assessment.
            </p>
            <div style='margin-top:16px;'>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important;'>VGG16</span>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important; margin-left:6px;'>Neural Network</span>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important; margin-left:6px;'>Batch Processing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:32px; margin-bottom:12px;'>📝</div>
            <h3 style='font-size:20px; margin-bottom:8px;'>Text Analysis</h3>
            <p style='color:#94a3b8 !important; font-size:14px; line-height:1.6;'>
                Analyze text messages for drug trafficking indicators using
                <b>NLP and Machine Learning</b>. Supports direct text input, CSV batch
                upload, and screenshot analysis for comprehensive evidence examination.
            </p>
            <div style='margin-top:16px;'>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important;'>TF-IDF</span>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important; margin-left:6px;'>Logistic Regression</span>
                <span style='background:rgba(139,92,246,0.15); padding:4px 12px;
                    border-radius:20px; font-size:12px; color:#a78bfa !important; margin-left:6px;'>CSV Upload</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("""
    <div style='text-align:center; margin: 20px 0;'>
        <h2 style='font-size:28px;'>How It Works</h2>
        <p style='color:#94a3b8 !important;'>Our dual-model system analyzes both visual and textual evidence</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    steps = [
        ("📤", "Upload", "Upload images, text, or CSV files"),
        ("⚙️", "Process", "AI extracts features & patterns"),
        ("🧠", "Analyze", "ML models classify the content"),
        ("📊", "Report", "View results & download reports"),
    ]

    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; padding:28px 16px;'>
                <div style='font-size:36px; margin-bottom:10px;'>{icon}</div>
                <h4 style='font-size:16px; margin-bottom:6px;'>{title}</h4>
                <p style='color:#94a3b8 !important; font-size:13px;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------
# IMAGE ANALYSIS PAGE
# ------------------------------------------------

def image_analysis_page():
    from image_predictor import predict_image

    st.markdown("""
    <div class='fade-in' style='text-align:center; margin-bottom:20px;'>
        <h1 style='font-size:36px;'>🖼️ Image Analysis</h1>
        <p style='color:#94a3b8 !important;'>Upload images to detect drug-related visual content</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary stats
    img_hist = st.session_state.image_history
    total = len(img_hist)
    drug_count = sum(1 for i in img_hist if i["prediction"] == "Drug")
    safe_count = total - drug_count

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Total Analyzed</div>
            <div class='stat-number'>{total}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card drug'>
            <div class='stat-label'>🚨 Drug Detected</div>
            <div class='stat-number'>{drug_count}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card safe'>
            <div class='stat-label'>✅ Non-Drug</div>
            <div class='stat-number'>{safe_count}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Two input methods: File Upload or Folder Path
    upload_tab, folder_tab = st.tabs(["📤 Upload Files", "📁 Folder Path"])

    # ---- TAB 1: File Upload ----
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload Evidence Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload one or more images for analysis"
        )

        if uploaded_files:
            if st.button("🔍  Analyze Uploaded Images", use_container_width=True, key="analyze_upload"):
                progress = st.progress(0, text="Analyzing images...")

                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        label, confidence = predict_image(tmp_path)
                        st.session_state.image_history.append({
                            "image": uploaded_file.name,
                            "prediction": label,
                            "confidence": confidence,
                            "risk": "High" if label == "Drug" else "Low"
                        })
                    except Exception as e:
                        st.warning(f"⚠ Error processing {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                    progress.progress(
                        (idx + 1) / len(uploaded_files),
                        text=f"Analyzed {idx + 1}/{len(uploaded_files)} images..."
                    )

                progress.empty()
                st.success(f"✅ Analysis complete! {len(uploaded_files)} images processed.")
                st.rerun()

    # ---- TAB 2: Folder Path ----
    with folder_tab:
        st.markdown("""
        <div class='glass-card' style='padding:16px;'>
            <p style='font-size:14px; color:#94a3b8 !important; margin:0;'>
                📁 Enter the <b>full path</b> to a folder containing images. The app will
                scan for all <code style='background:rgba(139,92,246,0.2);
                padding:2px 8px; border-radius:4px; color:#c084fc !important;'>.jpg .jpeg .png</code>
                files and analyze them.
            </p>
        </div>
        """, unsafe_allow_html=True)

        folder_path = st.text_input(
            "Folder Path",
            placeholder=r"e.g. C:\Users\...\labeled_dataset\drug",
            key="folder_path_input"
        )

        if folder_path:
            if os.path.isdir(folder_path):
                VALID_EXT = (".jpg", ".jpeg", ".png")
                image_files = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith(VALID_EXT)
                ]

                st.markdown(f"""
                <div class='glass-card' style='padding:14px;'>
                    <p style='margin:0; font-size:15px;'>
                        ✅ Found <b style='color:#a78bfa !important;'>{len(image_files)}</b>
                        images in this folder
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if image_files:
                    if st.button("🔍  Analyze Folder", use_container_width=True, key="analyze_folder"):
                        progress = st.progress(0, text="Analyzing folder images...")

                        for idx, filename in enumerate(image_files):
                            filepath = os.path.join(folder_path, filename)
                            try:
                                label, confidence = predict_image(filepath)
                                st.session_state.image_history.append({
                                    "image": filename,
                                    "prediction": label,
                                    "confidence": confidence,
                                    "risk": "High" if label == "Drug" else "Low"
                                })
                            except Exception as e:
                                st.warning(f"⚠ Error: {filename}: {e}")

                            progress.progress(
                                (idx + 1) / len(image_files),
                                text=f"Analyzed {idx + 1}/{len(image_files)} images..."
                            )

                        progress.empty()
                        st.success(f"✅ Folder analysis complete! {len(image_files)} images processed.")
                        st.rerun()
            else:
                st.error("❌ Folder not found. Please check the path.")

    # Results table
    if st.session_state.image_history:
        st.markdown("### 📋 Analysis Results")

        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            df = pd.DataFrame(st.session_state.image_history)
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                "image_analysis_report.csv",
                use_container_width=True
            )

        # Render results
        for item in reversed(st.session_state.image_history[-20:]):
            css_class = "result-drug" if item["prediction"] == "Drug" else "result-safe"
            badge_class = "badge-drug" if item["prediction"] == "Drug" else "badge-safe"
            risk_icon = "🚨" if item["risk"] == "High" else "✅"
            bar_class = "drug" if item["prediction"] == "Drug" else "safe"

            st.markdown(f"""
            <div class='{css_class}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-weight:600; font-size:15px;'>{item["image"]}</span>
                    </div>
                    <div style='display:flex; align-items:center; gap:16px;'>
                        <span class='{badge_class}'>{item["prediction"]}</span>
                        <span style='font-size:14px; color:#94a3b8 !important;'>{item["confidence"]:.1f}%</span>
                        <span style='font-size:14px;'>{risk_icon} {item["risk"]} Risk</span>
                    </div>
                </div>
                <div class='confidence-bar-bg'>
                    <div class='confidence-bar-fill {bar_class}' style='width:{item["confidence"]:.0f}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Image History", use_container_width=True):
            st.session_state.image_history = []
            st.rerun()


# ------------------------------------------------
# TEXT ANALYSIS PAGE
# ------------------------------------------------

def text_analysis_page():
    from text_predictor import predict_text

    st.markdown("""
    <div class='fade-in' style='text-align:center; margin-bottom:20px;'>
        <h1 style='font-size:36px;'>📝 Text Analysis</h1>
        <p style='color:#94a3b8 !important;'>Analyze messages for drug trafficking indicators</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary stats
    txt_hist = st.session_state.text_history
    total = len(txt_hist)
    sus_count = sum(1 for i in txt_hist if i["prediction"] == "Suspicious")
    normal_count = total - sus_count

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Total Analyzed</div>
            <div class='stat-number'>{total}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card drug'>
            <div class='stat-label'>⚠ Suspicious</div>
            <div class='stat-number'>{sus_count}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card safe'>
            <div class='stat-label'>✅ Normal</div>
            <div class='stat-number'>{normal_count}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Input tabs
    tab1, tab2, tab3 = st.tabs(["💬 Direct Text", "📄 CSV Upload", "📸 Screenshot"])

    # ---- TAB 1: Direct Text ----
    with tab1:
        st.markdown("#### Enter a message to analyze")

        # Demo examples
        with st.expander("📌 Load Demo Examples"):
            examples = [
                "premium powder available tonight contact privately",
                "fast delivery guaranteed message privately tonight",
                "trusted supplier selling premium tablets",
                "we are watching movie together tonight",
                "professor uploaded lecture notes today",
                "I'll bring the stuff to the party, best quality guaranteed",
            ]
            demo_text = st.selectbox("Select a demo message", examples,
                                     label_visibility="collapsed")
            if st.button("Load Example"):
                st.session_state.input_text = demo_text
                st.rerun()

        user_input = st.text_area(
            "Message",
            height=120,
            key="input_text",
            placeholder="Type or paste a message here..."
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_btn = st.button("🔍  Analyze Text", use_container_width=True)
        with btn_col2:
            if st.button("🗑️  Clear", use_container_width=True):
                st.session_state.input_text = ""
                st.rerun()

        if predict_btn:
            if user_input.strip() == "":
                st.warning("⚠ Please enter a message to analyze.")
            else:
                label, confidence = predict_text(user_input)

                st.session_state.text_history.append({
                    "message": user_input[:100],
                    "prediction": label,
                    "confidence": confidence
                })

                st.markdown("---")
                if label == "Suspicious":
                    st.markdown(f"""
                    <div class='result-drug'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span style='font-weight:600;'>⚠ Suspicious / Drug Related Message Detected</span>
                            <span class='badge-suspicious'>{confidence:.1f}%</span>
                        </div>
                        <p style='margin-top:8px; font-size:14px; color:#fca5a5 !important;'>{user_input}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-safe'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span style='font-weight:600;'>✅ Normal Message</span>
                            <span class='badge-normal'>{confidence:.1f}%</span>
                        </div>
                        <p style='margin-top:8px; font-size:14px; color:#86efac !important;'>{user_input}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ---- TAB 2: CSV Upload ----
    with tab2:
        st.markdown("#### Upload a CSV file with text messages")
        st.markdown("""
        <div class='glass-card' style='padding:16px;'>
            <p style='font-size:14px; color:#94a3b8 !important; margin:0;'>
                📋 Your CSV should have a column named <code style='background:rgba(139,92,246,0.2);
                padding:2px 8px; border-radius:4px; color:#c084fc !important;'>text</code>
                containing the messages to analyze.
            </p>
        </div>
        """, unsafe_allow_html=True)

        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)

                # Handle special case: "text,label" combined column
                if "text,label" in df.columns and "text" not in df.columns:
                    df[['text', 'label']] = df['text,label'].str.rsplit(',', n=1, expand=True)
                    df = df.drop(columns=['text,label'])

                # Try to find the text column
                text_col = None
                for col_name in ["text", "Text", "TEXT", "message", "Message", "content", "Content"]:
                    if col_name in df.columns:
                        text_col = col_name
                        break

                if text_col is None:
                    st.warning("⚠ Could not find a 'text' column. Available columns:")
                    st.write(list(df.columns))
                    text_col = st.selectbox("Select the text column:", df.columns)

                st.markdown(f"**Found {len(df)} rows** in column `{text_col}`")
                st.dataframe(df[[text_col]].head(5), use_container_width=True)

                if st.button("🔍  Analyze All Messages", use_container_width=True, key="csv_analyze"):
                    results = []
                    progress = st.progress(0, text="Analyzing messages...")

                    for idx, row in df.iterrows():
                        text_val = str(row[text_col])
                        if text_val.strip():
                            label, confidence = predict_text(text_val)
                            results.append({
                                "message": text_val[:100],
                                "prediction": label,
                                "confidence": confidence
                            })
                            st.session_state.text_history.append({
                                "message": text_val[:100],
                                "prediction": label,
                                "confidence": confidence
                            })

                        progress.progress(
                            (idx + 1) / len(df),
                            text=f"Analyzed {idx + 1}/{len(df)} messages..."
                        )

                    progress.empty()

                    # Show results
                    result_df = pd.DataFrame(results)
                    sus = sum(1 for r in results if r["prediction"] == "Suspicious")
                    nor = len(results) - sus

                    st.success(f"✅ Analysis complete! {len(results)} messages processed.")

                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"""
                        <div class='stat-card drug'>
                            <div class='stat-label'>⚠ Suspicious</div>
                            <div class='stat-number'>{sus}</div>
                        </div>""", unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div class='stat-card safe'>
                            <div class='stat-label'>✅ Normal</div>
                            <div class='stat-number'>{nor}</div>
                        </div>""", unsafe_allow_html=True)

                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "📥 Download Results CSV",
                        result_df.to_csv(index=False),
                        "text_analysis_report.csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error processing CSV: {e}")

    # ---- TAB 3: Screenshot ----
    with tab3:
        st.markdown("#### Upload screenshot images for analysis")
        st.markdown("""
        <div class='glass-card' style='padding:16px;'>
            <p style='font-size:14px; color:#94a3b8 !important; margin:0;'>
                📸 Upload screenshots of conversations or messages. The system will use
                <b>OCR (Optical Character Recognition)</b> to extract text and then analyze it
                for drug-related content.
            </p>
        </div>
        """, unsafe_allow_html=True)

        screenshot_files = st.file_uploader(
            "Upload Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="screenshot_upload"
        )

        if screenshot_files:
            if st.button("🔍  Extract & Analyze Text", use_container_width=True, key="ss_analyze"):
                try:
                    import pytesseract
                    from PIL import Image

                    progress = st.progress(0, text="Processing screenshots...")

                    for idx, ss_file in enumerate(screenshot_files):
                        img = Image.open(ss_file)
                        extracted_text = pytesseract.image_to_string(img)

                        if extracted_text.strip():
                            # Analyze extracted text line by line
                            lines = [l.strip() for l in extracted_text.split("\n") if l.strip()]

                            st.markdown(f"**📸 {ss_file.name}** — Extracted {len(lines)} lines")

                            for line in lines:
                                label, confidence = predict_text(line)
                                css = "result-drug" if label == "Suspicious" else "result-safe"
                                badge = "badge-suspicious" if label == "Suspicious" else "badge-normal"

                                st.session_state.text_history.append({
                                    "message": line[:100],
                                    "prediction": label,
                                    "confidence": confidence
                                })

                                st.markdown(f"""
                                <div class='{css}'>
                                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                                        <span style='font-size:14px;'>{line}</span>
                                        <span class='{badge}'>{label}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning(f"⚠ No text extracted from {ss_file.name}")

                        progress.progress(
                            (idx + 1) / len(screenshot_files),
                            text=f"Processed {idx + 1}/{len(screenshot_files)} screenshots..."
                        )

                    progress.empty()
                    st.success("✅ Screenshot analysis complete!")

                except ImportError:
                    st.error("""
                    ❌ **pytesseract is not installed.**

                    To enable screenshot OCR, install Tesseract:
                    1. Download from: https://github.com/tesseract-ocr/tesseract
                    2. Then run: `pip install pytesseract`

                    Alternatively, you can copy text from screenshots manually and use the
                    **Direct Text** tab.
                    """)
                except Exception as e:
                    st.error(f"❌ Error processing screenshot: {e}")

    # Text history
    st.markdown("---")
    if st.session_state.text_history:
        st.markdown("### 📋 Text Analysis History")

        col_b1, col_b2 = st.columns([4, 1])
        with col_b2:
            df_txt = pd.DataFrame(st.session_state.text_history)
            st.download_button(
                "📥 Download CSV",
                df_txt.to_csv(index=False),
                "text_analysis_report.csv",
                use_container_width=True,
                key="txt_download"
            )

        for item in reversed(st.session_state.text_history[-20:]):
            css = "result-drug" if item["prediction"] == "Suspicious" else "result-safe"
            badge = "badge-suspicious" if item["prediction"] == "Suspicious" else "badge-normal"
            st.markdown(f"""
            <div class='{css}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:14px;'>{item["message"]}</span>
                    <div style='display:flex; gap:12px; align-items:center;'>
                        <span class='{badge}'>{item["prediction"]}</span>
                        <span style='color:#94a3b8 !important; font-size:13px;'>{item["confidence"]:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Text History", use_container_width=True, key="clear_txt"):
            st.session_state.text_history = []
            st.rerun()


# ------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------

def dashboard_page():
    st.markdown("""
    <div class='fade-in' style='text-align:center; margin-bottom:20px;'>
        <h1 style='font-size:36px;'>📊 Dashboard</h1>
        <p style='color:#94a3b8 !important;'>Combined analysis overview across all modalities</p>
    </div>
    """, unsafe_allow_html=True)

    img_hist = st.session_state.image_history
    txt_hist = st.session_state.text_history

    total_img = len(img_hist)
    total_txt = len(txt_hist)
    img_drug = sum(1 for i in img_hist if i["prediction"] == "Drug")
    txt_sus = sum(1 for i in txt_hist if i["prediction"] == "Suspicious")
    total_threats = img_drug + txt_sus
    total_analyzed = total_img + total_txt

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Total Analyzed</div>
            <div class='stat-number'>{total_analyzed}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card alert'>
            <div class='stat-label'>⚠ Threats Found</div>
            <div class='stat-number'>{total_threats}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>🖼️ Images</div>
            <div class='stat-number'>{total_img}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>📝 Texts</div>
            <div class='stat-number'>{total_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='font-size:18px; margin-bottom:16px;'>🖼️ Image Analysis Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)

        if total_img > 0:
            img_safe = total_img - img_drug
            drug_pct = (img_drug / total_img) * 100

            st.markdown(f"""
            <div style='padding: 0 12px;'>
                <div style='display:flex; justify-content:space-between; margin:8px 0;'>
                    <span>🚨 Drug Detected</span><span style='font-weight:700; color:#ef4444 !important;'>{img_drug}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0;'>
                    <span>✅ Non-Drug</span><span style='font-weight:700; color:#22c55e !important;'>{img_safe}</span>
                </div>
                <div class='confidence-bar-bg' style='margin-top:12px;'>
                    <div class='confidence-bar-fill drug' style='width:{drug_pct:.0f}%;'></div>
                </div>
                <p style='font-size:12px; color:#94a3b8 !important; margin-top:6px;'>{drug_pct:.1f}% flagged as drug-related</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#64748b !important; padding:12px;'>No images analyzed yet.</p>",
                        unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='font-size:18px; margin-bottom:16px;'>📝 Text Analysis Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)

        if total_txt > 0:
            txt_normal = total_txt - txt_sus
            sus_pct = (txt_sus / total_txt) * 100

            st.markdown(f"""
            <div style='padding: 0 12px;'>
                <div style='display:flex; justify-content:space-between; margin:8px 0;'>
                    <span>⚠ Suspicious</span><span style='font-weight:700; color:#ef4444 !important;'>{txt_sus}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0;'>
                    <span>✅ Normal</span><span style='font-weight:700; color:#22c55e !important;'>{txt_normal}</span>
                </div>
                <div class='confidence-bar-bg' style='margin-top:12px;'>
                    <div class='confidence-bar-fill drug' style='width:{sus_pct:.0f}%;'></div>
                </div>
                <p style='font-size:12px; color:#94a3b8 !important; margin-top:6px;'>{sus_pct:.1f}% flagged as suspicious</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#64748b !important; padding:12px;'>No texts analyzed yet.</p>",
                        unsafe_allow_html=True)

    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Combined Report")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        if img_hist:
            df_img = pd.DataFrame(img_hist)
            st.download_button(
                "📥 Image Report (CSV)",
                df_img.to_csv(index=False),
                "image_report.csv",
                use_container_width=True
            )

    with exp2:
        if txt_hist:
            df_txt = pd.DataFrame(txt_hist)
            st.download_button(
                "📥 Text Report (CSV)",
                df_txt.to_csv(index=False),
                "text_report.csv",
                use_container_width=True
            )

    with exp3:
        if img_hist or txt_hist:
            combined = []
            for item in img_hist:
                combined.append({
                    "type": "Image",
                    "input": item["image"],
                    "prediction": item["prediction"],
                    "confidence": f"{item['confidence']:.1f}%",
                    "risk": item.get("risk", "")
                })
            for item in txt_hist:
                combined.append({
                    "type": "Text",
                    "input": item["message"],
                    "prediction": item["prediction"],
                    "confidence": f"{item['confidence']:.1f}%",
                    "risk": "High" if item["prediction"] == "Suspicious" else "Low"
                })

            df_combined = pd.DataFrame(combined)
            st.download_button(
                "📥 Combined Report (CSV)",
                df_combined.to_csv(index=False),
                "combined_report.csv",
                use_container_width=True
            )

    # Clear all
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear All History", use_container_width=True, key="clear_all"):
        st.session_state.image_history = []
        st.session_state.text_history = []
        st.rerun()


# ------------------------------------------------
# ROUTER
# ------------------------------------------------

if st.session_state.logged_in:
    sidebar_nav()

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Image Analysis":
        image_analysis_page()
    elif st.session_state.page == "Text Analysis":
        text_analysis_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
else:
    login()

import streamlit as st
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# ---- MediaPipe compatibility shim ----
MEDIAPIPE_READY = True
MEDIAPIPE_ERROR = None
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        try:
            from mediapipe.python import solutions as mp_solutions
            mp.solutions = mp_solutions
        except Exception as shim_error:
            MEDIAPIPE_READY = False
            MEDIAPIPE_ERROR = (
                "MediaPipe is installed, but the legacy 'mp.solutions' API is not available. "
                "Install an older compatible version such as: "
                "python -m pip install --force-reinstall mediapipe==0.10.20"
            )
except Exception as mp_error:
    MEDIAPIPE_READY = False
    MEDIAPIPE_ERROR = f"MediaPipe import failed: {mp_error}"

# Add backend to path after shim
try:
    from backend.analysevideo_web_final import analyze_video
    ANALYSIS_AVAILABLE = True
except Exception as import_error:
    ANALYSIS_AVAILABLE = False
    IMPORT_ERROR = str(import_error)
else:
    IMPORT_ERROR = None

# ── STREAMLIT CONFIG ──
st.set_page_config(
    page_title="BowlFast.AI - Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── ENHANCED CUSTOM CSS - BowlFast Color Palette ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600;700;800&family=Syne:wght@400;500;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
  --bg: #020408;
  --surface: #060b12;
  --panel: #0a1220;
  --cyan: #00c8ff;
  --green: #00ff9d;
  --orange: #ff6b2b;
  --white: #e8f4f8;
  --muted: rgba(232, 244, 248, 0.4);
  --border: rgba(0, 200, 255, 0.08);
  --border-bright: rgba(0, 200, 255, 0.22);
  --cyan-glow: rgba(0, 200, 255, 0.35);
  --green-dim: rgba(0, 255, 157, 0.1);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
  background: #020408 !important;
  color: #e8f4f8 !important;
  font-family: 'Syne', sans-serif !important;
}

[data-testid="stHeader"] {
  background: rgba(2, 4, 8, 0.95) !important;
  border-bottom: 1px solid var(--border) !important;
  backdrop-filter: blur(20px) !important;
}

/* ── GRID BACKGROUND ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    linear-gradient(rgba(0, 200, 255, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 200, 255, 0.04) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, black 20%, transparent 100%);
}

/* ── FILE UPLOAD ── */
[data-testid="stFileUploadDropzone"] {
  border: 2px dashed var(--cyan) !important;
  border-radius: 12px !important;
  background: rgba(0, 200, 255, 0.04) !important;
  padding: 40px !important;
  transition: all 0.3s ease !important;
}

[data-testid="stFileUploadDropzone"]:hover {
  border-color: var(--green) !important;
  background: rgba(0, 255, 157, 0.06) !important;
  box-shadow: 0 0 30px rgba(0, 200, 255, 0.15) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, var(--cyan) 0%, var(--green) 100%) !important;
  color: var(--bg) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 14px 32px !important;
  letter-spacing: 1px !important;
  box-shadow: 0 0 25px rgba(0, 200, 255, 0.3) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 13px !important;
  text-transform: uppercase !important;
  transition: all 0.3s ease !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 35px rgba(0, 200, 255, 0.4) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
  background: linear-gradient(135deg, rgba(10, 18, 32, 0.8) 0%, rgba(0, 200, 255, 0.05) 100%) !important;
  border: 1px solid rgba(0, 200, 255, 0.2) !important;
  border-radius: 12px !important;
  padding: 20px !important;
  backdrop-filter: blur(10px) !important;
  transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.15) !important;
}

/* ── FEEDBACK BOXES ── */
.feedback-box {
  background: linear-gradient(135deg, rgba(0, 200, 255, 0.08) 0%, rgba(0, 255, 157, 0.04) 100%) !important;
  border-left: 4px solid var(--cyan) !important;
  border: 1px solid var(--border-bright) !important;
  border-left: 4px solid var(--cyan) !important;
  border-radius: 8px !important;
  padding: 20px !important;
  margin: 16px 0 !important;
  backdrop-filter: blur(10px) !important;
}

/* ── DRILL CARDS ── */
.drill-card {
  background: linear-gradient(135deg, rgba(0, 255, 157, 0.08) 0%, rgba(0, 255, 157, 0.02) 100%) !important;
  border: 1px solid rgba(0, 255, 157, 0.2) !important;
  border-radius: 10px !important;
  padding: 20px !important;
  margin-bottom: 16px !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(10px) !important;
}

.drill-card:hover {
  border-color: var(--green) !important;
  box-shadow: 0 0 20px rgba(0, 255, 157, 0.1) !important;
  transform: translateY(-2px) !important;
}

/* ── HIDE DEPLOY BUTTON & MENU ── */
[data-testid="stToolbar"] {
  display: none !important;
}

/* ── HEADINGS ── */
h1, h2, h3 {
  font-family: 'Oxanium', monospace !important;
  letter-spacing: 1px !important;
}

h1 {
  font-size: 48px !important;
  font-weight: 800 !important;
  background: linear-gradient(135deg, var(--white) 0%, var(--cyan) 100%) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin-bottom: 12px !important;
}

h2 {
  font-size: 28px !important;
  font-weight: 700 !important;
  color: var(--cyan) !important;
}

h3 {
  font-size: 18px !important;
  font-weight: 600 !important;
  color: var(--white) !important;
}

/* ── RADIO BUTTONS ── */
[data-testid="stRadio"] {
  display: flex !important;
  gap: 16px !important;
}

[data-testid="stRadio"] label {
  background: rgba(0, 200, 255, 0.08) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 10px 16px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

[data-testid="stRadio"] label:hover {
  border-color: var(--cyan) !important;
  background: rgba(0, 200, 255, 0.12) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: rgba(0, 200, 255, 0.04) !important;
}

/* ── DIVIDER ── */
[data-testid="stHorizontalBlock"] hr {
  border-color: var(--border) !important;
  margin: 32px 0 !important;
}

/* ── SPINNER MESSAGE ── */
.stSpinner {
  color: var(--cyan) !important;
}

/* ── SUCCESS/ERROR MESSAGES ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  border-left: 4px solid !important;
}

/* ── VIDEOS ── */
video {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 0 30px rgba(0, 200, 255, 0.1) !important;
  margin: 20px 0 !important;
}

/* ── MAIN CONTAINER ── */
[data-testid="stMainBlockContainer"] {
  padding: 40px 20px !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
}

/* ── COLUMNS ── */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 12px !important;
}

/* ── INFO BOX ── */
.info-box {
  background: linear-gradient(135deg, rgba(0, 200, 255, 0.06) 0%, rgba(0, 200, 255, 0.02) 100%) !important;
  border: 1px solid var(--cyan) !important;
  border-radius: 10px !important;
  padding: 20px !important;
  margin: 20px 0 !important;
  color: var(--white) !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
  background: linear-gradient(135deg, rgba(0, 200, 255, 0.8) 0%, rgba(0, 255, 157, 0.6) 100%) !important;
  border: 1px solid var(--cyan) !important;
}

/* ── SECTION SPACING ── */
section {
  margin: 40px 0 !important;
}

</style>
""", unsafe_allow_html=True)

def safe_read_bytes(path_str: str | None):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_bytes()

# ── HEADER - Integrated from HTML ──
st.markdown("""
<style>
.header-sticky {
  position: sticky;
  top: -20px;
  z-index: 100;
  background: rgba(2, 4, 8, 0.98);
  padding: 16px 20px;
  margin: -40px -20px 32px -20px;
  border-bottom: 1px solid rgba(0, 200, 255, 0.1);
  backdrop-filter: blur(20px);
  width: calc(100% + 40px);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
<div class="header-sticky">
    <div>
        <h1 style='margin: 0; font-size: 26px;'>
            <span style='color: var(--cyan);'>Bowl</span><span style='color: var(--white);'>Fast</span><span style='color: var(--green);'>.</span><span style='color: var(--cyan); font-weight: 300;'>AI</span>
        </h1>
        <p style='color: rgba(232, 244, 248, 0.5); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; margin: 4px 0 0 0;'>AI Fast Bowling Analysis</p>
    </div>
    <div style='text-align: right;'>
        <p style='color: rgba(0, 200, 255, 0.6); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; margin: 0;'>Frame-Level Biomechanical Precision</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ANALYSIS SECTION TITLE ──
st.markdown("""
<h2 style='text-align: center; margin: 40px 0 16px; font-size: 36px;'>📊 Upload Your Delivery</h2>
<p style='text-align: center; color: rgba(232, 244, 248, 0.6); margin-bottom: 32px; font-size: 16px;'>Get frame-level biomechanical analysis of your fast bowling action</p>
""", unsafe_allow_html=True)

if not MEDIAPIPE_READY:
    st.error(f"❌ {MEDIAPIPE_ERROR}")
if not ANALYSIS_AVAILABLE:
    st.error(f"❌ Could not import analysis module: {IMPORT_ERROR}")

# ── FILE UPLOADER ──
uploaded_file = st.file_uploader(
    "Drag and drop your bowling video or click to select",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload a video of your fast bowling delivery in any standard format"
)

if uploaded_file is not None:
    # ── SUCCESS MESSAGE ──
    st.success(f"✓ Video loaded: **{uploaded_file.name}** · {round(uploaded_file.size / (1024 * 1024), 2)} MB")
    
    # ── ANALYSIS PARAMETERS ──
    st.markdown("<h3 style='margin-top: 32px; margin-bottom: 20px;'>⚙️ Analysis Parameters</h3>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        bowling_arm = st.radio("🎯 Bowling Arm", ["right", "left"], horizontal=True)
    with right_col:
        bowler_entry_side = st.radio("🏏 Entry Side", ["left", "right"], horizontal=True)

    # ── FILE HANDLING ──
    uploads_dir = APP_DIR / "uploads"
    outputs_dir = APP_DIR / "outputs"
    uploads_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    temp_path = uploads_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ── START ANALYSIS BUTTON (MAIN CTA) ──
    st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        if st.button("🚀 START ANALYSIS", use_container_width=True, key="start_analysis"):
            if not MEDIAPIPE_READY:
                st.stop()
            if not ANALYSIS_AVAILABLE:
                st.stop()

            # ── ANALYSIS SPINNER ──
            with st.spinner("⚡ Analyzing your bowling action... This may take 1-2 minutes"):
                try:
                    result = analyze_video(
                        video_path=str(temp_path),
                        output_dir=str(outputs_dir),
                        bowling_arm=bowling_arm,
                        bowler_entry_side=bowler_entry_side
                    )
                    st.session_state.analysis_result = result
                    st.session_state.analysis_complete = True
                    st.success("✓ Analysis complete! Scroll down to see your results.")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Tips: ensure your video is valid, MediaPipe is compatible, and FFmpeg is available if your backend needs it.")

# ── RESULTS SECTION ──
if st.session_state.get("analysis_complete"):
    st.markdown("""
    <div style='margin: 60px 0 40px; padding: 20px; border: 1px solid rgba(0, 200, 255, 0.15); border-radius: 10px; background: rgba(0, 200, 255, 0.04);'>
        <h2 style='margin-top: 0; color: var(--green);'>✓ Analysis Complete</h2>
        <p style='color: rgba(232, 244, 248, 0.6); margin: 8px 0 0;'>Your biomechanical data is ready. Review the metrics, video, and coaching feedback below.</p>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.analysis_result
    metrics = result.get("metrics", {})
    coach = result.get("coach", {})

    # ── PERFORMANCE METRICS SECTION ──
    st.markdown("""
    <h3 style='color: var(--cyan); margin-bottom: 20px; margin-top: 40px;'>📈 Performance Metrics</h3>
    """, unsafe_allow_html=True)

    metric_cols = st.columns(5)
    metric_data = [
        ("Run-up Balance", metrics.get("run_up_balance_label", "—")),
        ("Release Timing", metrics.get("release_label", "—")),
        ("Shoulder Alignment", metrics.get("shoulder_alignment_label", metrics.get("alignment_label", "—"))),
        ("Follow-Through", metrics.get("follow_through_label", "—")),
        ("Hip-Shoulder Sep.", f"{metrics.get('hip_shoulder_separation_ffc_deg', '—')}°"),
    ]

    for i, (label, value) in enumerate(metric_data):
        with metric_cols[i]:
            value_str = str(value).lower()
            if "good" in value_str or "efficient" in value_str:
                color = "var(--green)"
                icon = "✓"
            elif "warning" in value_str or "moderate" in value_str:
                color = "var(--orange)"
                icon = "⚠"
            elif "bad" in value_str or "high" in value_str:
                color = "#ff4444"
                icon = "✗"
            else:
                color = "var(--white)"
                icon = "•"
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, rgba(0, 200, 255, 0.08) 0%, rgba(0, 200, 255, 0.04) 100%); border: 1px solid rgba(0, 200, 255, 0.15); border-radius: 10px; padding: 18px; text-align: center; backdrop-filter: blur(10px);'>
                    <div style='font-size: 11px; color: rgba(232, 244, 248, 0.5); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-family: "IBM Plex Mono", monospace;'>{label}</div>
                    <div style='font-size: 20px; font-weight: 700; color: {color}; font-family: "Oxanium", monospace; letter-spacing: 1px;'>{icon} {value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── ANALYZED VIDEO SECTION ──
    st.markdown("<h3 style='color: var(--cyan); margin-top: 40px; margin-bottom: 20px;'>🎥 Analyzed Video</h3>", unsafe_allow_html=True)
    video_path = result.get("annotated_video_path")
    video_bytes = safe_read_bytes(video_path)
    if video_bytes:
        st.video(video_bytes)
        st.download_button(
            "⬇️ Download Analyzed Video",
            data=video_bytes,
            file_name=Path(video_path).name,
            mime="video/mp4",
            use_container_width=True
        )
    else:
        st.warning("⚠️ Analyzed video file was not found.")

    # ── COACH FEEDBACK SECTION ──
    st.markdown("<h3 style='color: var(--cyan); margin-top: 40px; margin-bottom: 20px;'>💬 Coach Feedback</h3>", unsafe_allow_html=True)
    feedback_list = coach.get("feedback", [])
    if feedback_list:
        for i, feedback in enumerate(feedback_list, 1):
            icon = "✓" if "good" in feedback.lower() or "efficient" in feedback.lower() else "⚠"
            st.markdown(
                f"""
                <div class='feedback-box'>
                    <p style='margin: 0; color: var(--white);'><strong style='color: var(--cyan);'>{icon} Feedback {i}:</strong> {feedback}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("✓ No major red flags detected. Continue building consistency and strength!")

    # ── RECOMMENDED DRILLS SECTION ──
    st.markdown("<h3 style='color: var(--green); margin-top: 40px; margin-bottom: 20px;'>🎯 Recommended Drills</h3>", unsafe_allow_html=True)
    drills = coach.get("drills", [])
    if drills:
        drill_cols = st.columns(2)
        for i, drill in enumerate(drills):
            with drill_cols[i % 2]:
                st.markdown(
                    f"""
                    <div class='drill-card'>
                        <div style='color: var(--green); font-weight: 600; margin-bottom: 10px; font-family: "Oxanium", monospace; font-size: 16px;'>→ {drill}</div>
                        <p style='font-size: 13px; color: rgba(232, 244, 248, 0.6); margin: 0; line-height: 1.5;'>Practice this drill to improve your bowling technique and build muscle memory.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ── ADVANCED METRICS ──
    with st.expander("📊 Advanced Metrics (Raw Data)"):
        st.json(metrics)

    # ── DOWNLOAD REPORT ──
    report_path = result.get("report_json_path")
    report_bytes = safe_read_bytes(report_path)
    if report_bytes:
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Full Analysis Report",
            data=report_bytes,
            file_name=Path(report_path).name,
            mime="application/json",
            use_container_width=True
        )

    # ── RESET & CONTINUE ──
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    reset_col1, reset_col2 = st.columns(2)
    with reset_col1:
        if st.button("🔄 Analyze Another Delivery", use_container_width=True, key="analyze_again"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_result = {}
            st.rerun()
    with reset_col2:
        st.markdown(
            "<div style='text-align: center; padding-top: 12px; color: rgba(232, 244, 248, 0.5); font-size: 13px;'>💪 Keep practicing with the drills above to continuously improve your technique.</div>",
            unsafe_allow_html=True
        )

else:
    # ── EMPTY STATE ──
    if uploaded_file is None:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px; color: rgba(232, 244, 248, 0.6);'>
            <div style='font-size: 48px; margin-bottom: 20px;'>📹</div>
            <p style='font-size: 18px; margin-bottom: 8px;'>Ready to analyze your bowling action?</p>
            <p style='font-size: 14px; color: rgba(232, 244, 248, 0.4);'>Upload a video above to get started. Our AI will analyze every frame of your delivery.</p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER - Integrated from HTML ──
st.markdown("""
<div style='margin-top: 80px; padding-top: 40px; border-top: 1px solid rgba(0, 200, 255, 0.1); text-align: center;'>
    <div style='margin-bottom: 20px;'>
        <span style='color: var(--cyan); font-weight: 800; font-family: "Oxanium", monospace; font-size: 18px;'>Bowl</span><span style='color: var(--white); font-weight: 800; font-family: "Oxanium", monospace; font-size: 18px;'>Fast</span><span style='color: var(--green); font-weight: 800; font-family: "Oxanium", monospace; font-size: 18px;'>.</span><span style='color: var(--cyan); font-weight: 300; font-family: "Oxanium", monospace; font-size: 18px;'>AI</span>
    </div>
    <div style='font-size: 12px; color: rgba(232, 244, 248, 0.4); margin: 16px 0;'>
        <a href='#' style='color: rgba(0, 200, 255, 0.6); text-decoration: none; margin: 0 12px;'>How It Works</a>
        <a href='#' style='color: rgba(0, 200, 255, 0.6); text-decoration: none; margin: 0 12px;'>Features</a>
        <a href='#' style='color: rgba(0, 200, 255, 0.6); text-decoration: none; margin: 0 12px;'>Analysis</a>
        <a href='#' style='color: rgba(0, 200, 255, 0.6); text-decoration: none; margin: 0 12px;'>Waitlist</a>
    </div>
    <p style='font-size: 11px; color: rgba(232, 244, 248, 0.3); margin-top: 24px;'>© 2026 BowlFast.AI · Frame-level biomechanical analysis for every fast bowler</p>
</div>
""", unsafe_allow_html=True)
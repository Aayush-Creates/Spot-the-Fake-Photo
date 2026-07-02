import streamlit as st
import numpy as np
import traceback
import joblib
import cv2
import base64
import warnings
from scipy import stats
from skimage.feature import local_binary_pattern

warnings.filterwarnings('ignore')

# --- Page Configurations ---
st.set_page_config(page_title="Screen Recapture Detector", page_icon="🔍", layout="wide")

# --- Custom Theme Toggle ---
# Use session_state directly so the label can reflect the CURRENT active mode
# before the widget renders (the widget's return value isn't available until
# after st.toggle() is called, so the label must be derived from state instead).
if "dark_mode_toggle" not in st.session_state:
    st.session_state.dark_mode_toggle = True

colA, colB = st.columns([85, 15])
with colB:
    toggle_label = "🌙 Dark Mode" if st.session_state.dark_mode_toggle else "☀️ Light Mode"
    is_dark = st.toggle(toggle_label, key="dark_mode_toggle")

# Define theme colors based on toggle
if is_dark:
    bg_color = "#121212"
    text_color = "#E0E0E0"
    secondary_bg = "#1E1E1E"
    border_color = "#333333"
else:
    # Modern Beige Light Mode
    bg_color = "#F4EFE6"
    text_color = "#1C1C1C"
    secondary_bg = "#FFFFFF"
    border_color = "#D8CDBA"

# --- UI Styling: Dynamic CSS Injection ---
st.markdown(
    f"""
    <style>
        /* 1. Main Background and Global Text */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        html, body, [class*="st-"] {{
            color: {text_color} !important;
        }}

        /* 2. Reduced Header Sizes */
        h1 {{ font-size: 2.2rem !important; color: {text_color} !important; }}
        h2 {{ font-size: 1.8rem !important; color: {text_color} !important; }}
        h3 {{ font-size: 1.4rem !important; color: {text_color} !important; }}

        /* 3. Input Controls & Text */
        .stRadio p {{
            font-size: 20px !important;
        }}
        .stAlert p {{
            font-size: 18px !important;
        }}

        /* 4. Scale Result Metrics */
        [data-testid="stMetricValue"] {{
            font-size: 3rem !important;
            color: {text_color} !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 1.2rem !important;
            color: {text_color} !important;
        }}

        /* 5. File Uploader Contrast Fixes */
        /* FIX: correct data-testid is "stFileUploaderDropzone" (previously
           misspelled "stFileUploadDropzone" -> selector matched nothing, so the
           dropzone + its inner text/button kept Streamlit's native dark styling
           even in light mode, making everything unreadable). */
        section[data-testid="stFileUploader"] {{
            background-color: transparent !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {secondary_bg} !important;
            border: 2px dashed {border_color} !important;
            border-radius: 12px !important;
        }}
        /* Force all text/icons inside the dropzone to inherit the correct text color */
        [data-testid="stFileUploaderDropzone"] * {{
            color: {text_color} !important;
            fill: {text_color} !important;
            opacity: 1 !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] small {{
            color: {text_color} !important;
            opacity: 0.75 !important;
        }}
        /* Style the 'Browse files' button inside the uploader */
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        /* Uploaded file row (name, size, remove icon) */
        [data-testid="stFileUploaderFile"] {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
        }}
        [data-testid="stFileUploaderFile"] * {{
            color: {text_color} !important;
        }}

        /* 6. Divider Lines */
        hr {{
            border-bottom-color: {border_color} !important;
        }}

        /* 7. Fixed-size image frame (prevents tall camera captures from
           blowing out the layout / filling the whole screen) */
        .fixed-image-frame {{
            width: 100%;
            height: 380px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {border_color};
            background-color: {secondary_bg};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .fixed-image-frame img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .fixed-image-caption {{
            text-align: center;
            color: {text_color};
            opacity: 0.75;
            font-size: 14px;
            margin-top: 6px;
        }}

        /* 8. Footer Styling - Increased Size */
        .footer-note {{
            text-align: center;
            font-size: 18px !important;
            font-weight: 500;
            color: {text_color};
            opacity: 0.85;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid {border_color};
        }}

        /* Camera input fixes */
        [data-testid="stCameraInput"]{{
            background-color: transparent !important;
        }}
        [data-testid="stCameraInput"] button{{
            background-color:#0E6FFF !important;
            color:#FFFFFF !important;
            border:none !important;
            border-radius:8px !important;
            font-weight:600 !important;
        }}
        [data-testid="stCameraInput"] button:hover{{
            background-color:#0056d6 !important;
        }}
        [data-testid="stCameraInput"] *{{
            color: inherit !important;
        }}

    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model Loaded Successfully")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.code(traceback.format_exc())
        return None

model = load_model()

# --- Fixed-size image renderer ---
def render_fixed_image(img_bytes, caption, container):
    """Renders an image inside a fixed-size, cropped frame instead of letting
    it stretch to full container width/height (which was causing tall camera
    captures to take over the whole screen)."""
    b64 = base64.b64encode(img_bytes).decode()
    container.markdown(
        f"""
        <div class="fixed-image-frame">
            <img src="data:image/png;base64,{b64}" />
        </div>
        <div class="fixed-image-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )

# --- Image Processing & Feature Engineering ---
def extract_features(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Decoding failed. Invalid image format.")

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    features['fft_peak_ratio'] = float(magnitude.max() / (magnitude.mean() + 1e-6))
    features['fft_std'] = float(magnitude.std())
    h, w = magnitude.shape
    center_mask = np.zeros((h, w))
    center_mask[h//4:3*h//4, w//4:3*w//4] = 1
    features['fft_highfreq_ratio'] = float(magnitude[center_mask == 0].mean() / (magnitude[center_mask == 1].mean() + 1e-6))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blur.astype(float)
    features['noise_std'] = float(noise.std())
    features['noise_kurtosis'] = float(stats.kurtosis(noise.flatten()))
    features['noise_mean_abs'] = float(np.abs(noise).mean())

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, v in enumerate(hist): features[f'lbp_hist_{i}'] = float(v)
    features['lbp_uniformity'] = float((lbp == lbp.max()).mean())

    for i, ch in enumerate(['b', 'g', 'r']):
        channel = img[:, :, i].astype(float)
        features[f'{ch}_mean'] = float(channel.mean())
        features[f'{ch}_std'] = float(channel.std())
        features[f'{ch}_skew'] = float(stats.skew(channel.flatten()))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = float(hsv[:, :, 1].mean())
    features['saturation_std'] = float(hsv[:, :, 1].std())
    features['value_mean'] = float(hsv[:, :, 2].mean())
    features['value_std'] = float(hsv[:, :, 2].std())

    edges = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var'] = float(edges.var())
    features['laplacian_mean_abs'] = float(np.abs(edges).mean())
    features['laplacian_kurtosis'] = float(stats.kurtosis(edges.flatten()))

    h, w = gray.shape
    center_val = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
    corners = [gray[:h//8, :w//8].mean(), gray[:h//8, -w//8:].mean(), gray[-h//8:, :w//8].mean(), gray[-h//8:, -w//8:].mean()]
    corner_val = float(np.mean(corners))
    features['vignette_ratio'] = float(center_val / (corner_val + 1e-6))
    features['vignette_diff'] = float(center_val - corner_val)

    features['glare_ratio'] = float((gray > 240).mean())
    features['rb_ratio'] = float(img[:, :, 2].mean() / (img[:, :, 0].mean() + 1e-6))
    features['color_cast'] = float(img[:, :, 2].astype(float).mean() - img[:, :, 0].astype(float).mean())

    diff_h = np.abs(np.diff(gray.astype(float), axis=0))
    diff_v = np.abs(np.diff(gray.astype(float), axis=1))
    features['block_artifact_h'] = float(diff_h[7::8, :].mean() / (diff_h.mean() + 1e-6))
    features['block_artifact_v'] = float(diff_v[:, 7::8].mean() / (diff_v.mean() + 1e-6))

    return features

# --- Main UI Header ---
st.title("🔍 Screen Recapture Detection")
st.markdown("Upload an image file or capture a live shot using your device's camera to verify if it is a genuine photo or a picture taken of a screen.")
st.divider()

if model is None:
    st.error(f"🚨 Could not load '{MODEL_PATH}'. Ensure it is placed in the same directory as this script.")
    st.stop()

# --- Main Input Controls ---
st.subheader("⚙️ Input Options")
input_mode = st.radio("Select Image Source:", ("📁 Upload File", "📷 Use Camera"), horizontal=True)

image_payload = None
if input_mode == "📁 Upload File":
    file_stream = st.file_uploader("Browse supported media...", type=["jpg", "jpeg", "png"])
    if file_stream is not None:
        image_payload = file_stream.read()
else:
    st.markdown("### 📷 Capture Photo")
    camera_stream = st.camera_input("Position the asset facing the lens")
    if camera_stream is not None:
        image_payload = camera_stream.read()

    st.markdown("---")
    st.markdown("### 📁 Or Upload a Photo")
    upload_stream = st.file_uploader(
        "Choose an image instead",
        type=["jpg","jpeg","png"],
        key="camera_upload"
    )
    if upload_stream is not None:
        image_payload = upload_stream.read()

st.divider()

# --- Prediction Processing & Display ---
if image_payload is not None:

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Captured Image")
        # Fixed-size frame instead of use_container_width, so tall camera
        # captures get cropped to a consistent box rather than filling the screen
        image_placeholder = st.empty()
        render_fixed_image(image_payload, "Awaiting analysis...", image_placeholder)

    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Extracting structural features and scoring..."):
            try:
                extracted_metrics = extract_features(image_payload)
                feature_vector = np.array(list(extracted_metrics.values())).reshape(1, -1)

                recapture_probability = model.predict_proba(feature_vector)[0][1]

                # Update the image placeholder with the new "Complete" caption
                render_fixed_image(image_payload, "Analysis Complete!", image_placeholder)

                if recapture_probability >= 0.5:
                    st.error("🚨 **CLASSIFICATION: SCREEN RECAPTURE (FAKE)**")
                    st.metric(label="Recapture Probability", value=f"{recapture_probability * 100:.1f}%")
                    st.write("The structural features (noise patterns, moiré effects, edge sharpness) strongly indicate this is a picture of a digital screen.")
                else:
                    st.success("✅ **CLASSIFICATION: GENUINE REAL PHOTO**")
                    st.metric(label="Recapture Probability", value=f"{recapture_probability * 100:.1f}%")
                    st.write("The natural noise profile, lens vignetting, and color dispersion indicate this is a genuine organic photograph.")

            except Exception as error_context:
                st.error(f"Pipeline anomaly encountered during feature mapping: {error_context}")
else:
    st.info("👆 Please select an input source above to begin.")

# --- Footer Note ---
st.markdown(
    '<div class="footer-note">💡 <b>Note:</b> This system is optimized and works best with images taken via a mobile phone camera.</div>',
    unsafe_allow_html=True
)
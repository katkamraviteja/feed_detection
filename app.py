import streamlit as st
import sys
import os

# Force environment variables before ANY imports
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Try to handle the OpenCV import issue
def safe_import_yolo():
    try:
        # Try importing cv2 first to catch the error early
        import cv2
        from ultralytics import YOLO
        return YOLO, None
    except ImportError as e:
        if "libGL.so.1" in str(e):
            return None, "OpenGL library missing. Please ensure opencv-python-headless is installed instead of opencv-python."
        else:
            return None, f"Import error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Import other packages
from PIL import Image
import numpy as np
import json

# Page config
st.set_page_config(
    page_title="Shrimp Feed Classifier",
    page_icon="🦐",
    layout="wide"
)

st.title("🦐 Shrimp Feed Classification - YOLO Model")

# Try to import YOLO
YOLO, error_msg = safe_import_yolo()

# Show current environment info for debugging
with st.expander("🔍 Debug Information"):
    st.write("**Python version:**", sys.version)
    st.write("**Installed packages:**")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        st.code(result.stdout)
    except:
        st.write("Could not retrieve package list")

st.stop()

# Load model
@st.cache_resource
def load_model():
    try:
        model = YOLO("feed_50epochs.pt")
        return model, None
    except FileNotFoundError:
        return None, "Model file 'classify_feed.pt' not found. Please upload it to your repository."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, model_error = load_model()

if model is None:
    st.error(f"❌ {model_error}")
    st.info("Make sure the 'classify_feed.pt' file is in the root of your GitHub repository.")
    st.stop()

st.success("✅ YOLO model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image for classification",
    type=["jpg", "jpeg", "png"],
    help="Upload a shrimp feed image to classify"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
    
    with col2:
        # Run prediction
        with st.spinner("🔄 Analyzing image..."):
            try:
                results = model.predict(source=np.array(image), save=False, verbose=False)
                
                for r in results:
                    probs = r.probs
                    top_idx = probs.top1
                    top_conf = probs.top1conf.item()
                    top_class = model.names[top_idx]
                    
                    # Display prediction
                    st.markdown("### 📊 Prediction Results")
                    st.success(f"**Class:** {top_class}")
                    st.info(f"**Confidence:** {top_conf:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(top_conf)
                    
                    # Create JSON result
                    result_data = {
                        "filename": uploaded_file.name,
                        "predicted_class": top_class,
                        "confidence": round(top_conf, 4),
                        "timestamp": st.session_state.get('timestamp', 'N/A')
                    }
                    
                    # Show JSON
                    st.markdown("### 📄 JSON Output")
                    json_str = json.dumps(result_data, indent=2)
                    st.code(json_str, language="json")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Results",
                        data=json_str,
                        file_name=f"prediction_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please try uploading a different image or check the model file.")

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 and Streamlit*")


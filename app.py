import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json

# Load trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("feed_50epochs.pt")

model = load_model()

st.title("🦐 Shrimp Feed Classification - YOLO Model")

# Upload single or multiple images
uploaded_files = st.file_uploader(
    "Upload image(s) for classification",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False  # 👈 allow only one image at a time
)

if uploaded_files:
    results_list = []

    try:
        # Open image
        image = Image.open(uploaded_files).convert("RGB")

        # Show input image in Streamlit
        st.image(image, caption=f"Uploaded: {uploaded_files.name}", use_column_width=True)

        # Run YOLO prediction
        results = model.predict(source=np.array(image), save=False, verbose=False)

        for r in results:
            probs = r.probs
            top_idx = probs.top1
            top_conf = probs.top1conf.item()
            top_class = model.names[top_idx]

            # Create JSON result (only for this image)
            results_list.append({
                "filename": uploaded_files.name,
                "predicted_class": top_class,
                "confidence": round(top_conf, 2)
            })

    except Exception as e:
        results_list.append({
            "filename": uploaded_files.name,
            "error": str(e)
        })

    # Convert results to JSON
    results_json = json.dumps(results_list, indent=4)

    st.subheader("📄 JSON Results")
    st.code(results_json, language="json")

    # Download JSON
    st.download_button(
        label="📥 Download JSON Results",
        data=results_json,
        file_name="predictions.json",
        mime="application/json"
    )

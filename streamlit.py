import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Load the trained model
model = YOLO("runs/detect/YOLOv8n50epochs/weights/best.pt")

# Streamlit UI Configuration
st.set_page_config(page_title="Solar Panel Detection", layout="wide")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses AI to detect solar panels in aerial images. "
    "Upload an image to get started."
)

# Main Area
st.title("🔆 Solar Panel Detection AI")
st.markdown(
    """
    📸 **Upload an aerial image**, and this AI will detect solar panels automatically.
    """,
    unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model(image_cv)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = f"{confidence:.2f}"
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # Convert back to PIL for Streamlit
        result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Display results
    st.image(result_image, caption="Detection Result", use_container_width=True)

    # Download option
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="📥 Download Result",
        data=buf,
        file_name="solar_detection.png",
        mime="image/png"
    )
else:
    st.write("Please upload an image to proceed.")

# Footer
st.markdown("---")
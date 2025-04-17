# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import time

# Import model definition from model_def.py
from model_def import AttU_Net # Make sure model_def.py is in the same directory

# --- Configuration ---
MODEL_PATH = "best_att_unet_model.pth"
# Use CPU for broader compatibility, change to "cuda" if deploying on GPU server
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256 # Must match the training input size
IMAGE_WIDTH = 256  # Must match the training input size
IMG_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
IMG_STD = [0.229, 0.224, 0.225]  # ImageNet std
THRESHOLD = 0.5 # Threshold for converting probabilities to binary mask

# --- Model Loading (Cached) ---
@st.cache_resource # Cache the model loading function
def load_pytorch_model(model_path, device):
    """Loads the PyTorch model."""
    try:
        # Initialize model architecture (make sure parameters match your trained model)
        model = AttU_Net(img_ch=3, output_ch=1) # Adjust if your model has different channels
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        # Set the model to evaluation mode
        model.eval()
        # Move model to the specified device
        model.to(device)
        st.success(f"Model loaded successfully on {device.upper()}")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- Image Preprocessing ---
# Define the same transformations used for validation (without augmentations)
preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

def preprocess_image(image_pil, transform):
    """Preprocesses PIL Image for model inference."""
    # Apply transformations
    image_tensor = transform(image_pil)
    # Add batch dimension (model expects B, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# --- Prediction ---
def predict(model, image_tensor, device):
    """Runs inference and returns the raw output logits."""
    with torch.no_grad(): # IMPORTANT: Disable gradient calculation for inference
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
    return logits

# --- Postprocessing ---
def postprocess_mask(logits, threshold):
    """Converts model logits to a displayable binary mask."""
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits) # Shape: [1, 1, H, W]
    # Apply threshold to get binary mask
    binary_mask = (probs > threshold).float() # Shape: [1, 1, H, W]
    # Remove batch and channel dimensions for display -> [H, W]
    mask_np = binary_mask.squeeze().cpu().numpy()
    return mask_np

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("✨ Image Segmentation using Attention U-Net ✨")
st.write(f"Using device: **{DEVICE.upper()}** | Model: **{MODEL_PATH}**")
st.write("Upload an image and click 'Segment Image' to see the result.")

# Load the model
model = load_pytorch_model(MODEL_PATH, DEVICE)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None and model is not None:
    # Read the uploaded image using PIL
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.subheader("Uploaded Image")
            st.image(image_pil, use_container_width=True)

        # Add a button to trigger segmentation
        if st.button("Segment Image", key="segment_button"):
            start_time = time.time()
            # 1. Preprocess
            with st.spinner("Preprocessing image..."):
                input_tensor = preprocess_image(image_pil, preprocess_transform)

            # 2. Predict
            with st.spinner("Running segmentation model..."):
                output_logits = predict(model, input_tensor, DEVICE)

            # 3. Postprocess
            with st.spinner("Generating mask..."):
                segmented_mask_np = postprocess_mask(output_logits, THRESHOLD)

            end_time = time.time()
            processing_time = end_time - start_time

            with col2:
                st.subheader("Segmentation Result")
                # Display the mask - use clamp=True for binary masks
                st.image(segmented_mask_np, caption=f"Segmentation Mask (Threshold={THRESHOLD})", clamp=True, use_container_width=True)
                st.success(f"Segmentation completed in {processing_time:.2f} seconds.")
                # Optional: Add download button for the mask
                # Convert numpy mask back to PIL Image to save
                # mask_img = Image.fromarray((segmented_mask_np * 255).astype(np.uint8))
                # buf = io.BytesIO()
                # mask_img.save(buf, format="PNG")
                # byte_im = buf.getvalue()
                # st.download_button(label="Download Mask", data=byte_im, file_name="segmented_mask.png", mime="image/png")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.error("Please try uploading a different image file (JPG, PNG).")

elif model is None:
    st.warning("Model could not be loaded. Please check the model path and file integrity.")

st.sidebar.info(
    """
    **About this App:**
    This app uses a trained Attention U-Net model to perform semantic segmentation on uploaded images.
    - Upload an image (JPG, PNG).
    - Click 'Segment Image'.
    - View the original image and the predicted segmentation mask side-by-side.
    """
)
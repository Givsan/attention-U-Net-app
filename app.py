# app.py
import streamlit as st

# **** THIS MUST BE THE FIRST STREAMLIT COMMAND ****
st.set_page_config(layout="wide", page_title="Image Segmentation & Vectorization")
# **************************************************

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import time
import io # For BytesIO, useful for downloads
import zipfile # For zipping shapefiles

# Import model definition from model_def.py
from model_def import AttU_Net # Make sure model_def.py is in the same directory

# ---- ADD GEOSPATIAL LIBRARIES ----
RASTERIO_GEOPANDAS_AVAILABLE = False
geospatial_libs_message_type = "warning" # Default to warning
geospatial_libs_message = "Rasterio and/or GeoPandas not installed. GeoTIFF input and vectorization will be disabled."
try:
    import rasterio
    import rasterio.features
    import geopandas as gpd
    # from shapely.geometry import shape # Not directly used here but often a geopandas dependency
    RASTERIO_GEOPANDAS_AVAILABLE = True
    geospatial_libs_message = "Geospatial libraries (rasterio, geopandas) loaded."
    geospatial_libs_message_type = "info"
except ImportError:
    pass # Message will be displayed later

# --- Configuration ---
MODEL_PATH = "best_att_unet_model.pth" # Make sure this is your trained AttU_Net model
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
        model = AttU_Net(img_ch=3, output_ch=1) # Using AttU_Net from your model_def.py
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        model.to(device)
        # Message will be displayed in the main app flow after set_page_config
        return model, f"Model loaded successfully on {device.upper()}"
    except FileNotFoundError:
        return None, f"Model file not found at {model_path}. Please ensure it's in the correct directory."
    except Exception as e:
        return None, f"Error loading the model: {e}"

# --- Image Preprocessing ---
# For standard images (JPG, PNG)
preprocess_transform_rgb = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

def preprocess_pil_image(image_pil, transform):
    """Preprocesses PIL Image for model inference."""
    image_tensor = transform(image_pil)
    return image_tensor.unsqueeze(0) # Add batch dimension

def preprocess_geotiff_data_for_model(image_data_np_from_rasterio, target_height, target_width, mean, std):
    """
    Preprocesses NumPy array data from a GeoTIFF (read by rasterio) for the model.
    Assumes image_data_np_from_rasterio is (bands, height, width).
    """
    processed_bands = []
    num_bands = image_data_np_from_rasterio.shape[0]

    if num_bands >= 3:
        display_bands_np = image_data_np_from_rasterio[:3, :, :]
    elif num_bands == 1:
        display_bands_np = np.repeat(image_data_np_from_rasterio, 3, axis=0)
    else:
        st.warning(f"GeoTIFF has {num_bands} bands. Attempting to process, but results may vary. Model expects 3 bands.")
        if num_bands > 0:
             display_bands_np = np.repeat(image_data_np_from_rasterio[[0], :, :], 3, axis=0)
        else:
            st.error("GeoTIFF has 0 bands. Cannot process.")
            return None

    pil_compatible_bands = []
    for i in range(display_bands_np.shape[0]):
        band = display_bands_np[i, :, :].astype(np.float32)
        p2, p98 = np.percentile(band, (2, 98))
        if p98 - p2 > 1e-6 :
            scaled_band = np.clip((band - p2) / (p98 - p2), 0, 1) * 255
        else:
            scaled_band = np.zeros_like(band)
        pil_compatible_bands.append(scaled_band.astype(np.uint8))

    pil_image_np = np.stack(pil_compatible_bands, axis=-1)
    pil_image = Image.fromarray(pil_image_np, 'RGB')

    img_tensor = preprocess_transform_rgb(pil_image)
    return img_tensor.unsqueeze(0)


# --- Prediction ---
def predict(model, image_tensor, device):
    """Runs inference and returns the raw output logits."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
    return logits

# --- Postprocessing to get mask tensor ---
def get_mask_tensor(logits, threshold):
    """Converts model logits to a binary mask tensor [1, 1, H, W]."""
    probs = torch.sigmoid(logits)
    binary_mask_tensor = (probs > threshold).float()
    return binary_mask_tensor

# ---- RASTER TO VECTOR CONVERSION ----
def raster_to_vector_geopandas(mask_tensor_nchw, original_transform, original_crs,
                               output_vector_path, road_pixel_value=1):
    if not RASTERIO_GEOPANDAS_AVAILABLE:
        st.error("Cannot vectorize: Rasterio/GeoPandas libraries are not available.")
        return False, None
    try:
        binary_mask_array = mask_tensor_nchw.squeeze().cpu().numpy().astype(np.uint8)
        mask_for_shapes = (binary_mask_array == road_pixel_value)

        if not np.any(mask_for_shapes):
            st.info("No road features detected in the segmentation mask to vectorize.")
            gdf = gpd.GeoDataFrame(columns=['geometry', 'road_val'], crs=original_crs)
            # Determine driver based on extension for saving empty file
            file_ext = os.path.splitext(output_vector_path)[1].lower()
            driver = None
            if file_ext == ".shp": driver = "ESRI Shapefile"
            elif file_ext == ".geojson": driver = "GeoJSON"
            elif file_ext == ".gpkg": driver = "GPKG"
            if driver: gdf.to_file(output_vector_path, driver=driver)
            st.info(f"Empty vector file saved to {output_vector_path}")
            return True, output_vector_path # Success, but indicate it's empty

        results = [
            {'properties': {'road_val': int(v)}, 'geometry': s_geom}
            for i, (s_geom, v) in enumerate(
                rasterio.features.shapes(binary_mask_array, mask=mask_for_shapes, transform=original_transform)
            )
            if v == road_pixel_value
        ]

        if not results: # Should be caught by np.any above, but as a fallback
            st.info(f"No vector features extracted with pixel value {road_pixel_value}.")
            gdf = gpd.GeoDataFrame(columns=['geometry', 'road_val'], crs=original_crs)
        else:
            gdf = gpd.GeoDataFrame.from_features(results, crs=original_crs)

        output_dir = os.path.dirname(output_vector_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_ext = os.path.splitext(output_vector_path)[1].lower()
        driver = None
        if file_ext == ".shp": driver = "ESRI Shapefile"
        elif file_ext == ".geojson": driver = "GeoJSON"
        elif file_ext == ".gpkg": driver = "GPKG"
        else:
            st.error(f"Unsupported vector output format: {file_ext}. Use .shp, .geojson, or .gpkg.")
            return False, None

        gdf.to_file(output_vector_path, driver=driver)
        # st.success(f"Vector data saved locally: {output_vector_path}") # Message handled by download button
        return True, output_vector_path

    except Exception as e:
        st.error(f"Error during raster-to-vector conversion: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False, None

# --- Streamlit App UI ---
# st.set_page_config() was called at the top

# Display geospatial library status in the sidebar
if geospatial_libs_message_type == "info":
    st.sidebar.info(geospatial_libs_message)
else:
    st.sidebar.warning(geospatial_libs_message)

# Load the model and display status in sidebar
model, model_load_message = load_pytorch_model(MODEL_PATH, DEVICE)
if model:
    st.sidebar.success(model_load_message)
else:
    st.sidebar.error(model_load_message)

st.sidebar.write(f"Using device: **{DEVICE.upper()}**")
st.sidebar.write(f"Model: **{MODEL_PATH}**")
st.sidebar.write(f"Input Size for Model: **{IMAGE_WIDTH}x{IMAGE_HEIGHT}**")
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About this App:**
    This app uses a trained Attention U-Net model to perform semantic segmentation.
    - Upload an image (JPG, PNG).
    - For georeferenced vector output, upload a GeoTIFF (TIF/TIFF).
    - Click 'Segment Image'.
    - If a GeoTIFF was used, vectorization options will appear after segmentation.
    """
)
# Redundant check, but emphasizes if libraries are missing
if not RASTERIO_GEOPANDAS_AVAILABLE:
    st.sidebar.error("Geospatial libraries are missing. GeoTIFF processing and vectorization are disabled.")


st.title("✨ Image Segmentation & Vectorization ✨")

# File Uploader
uploaded_file = st.file_uploader("Choose an image (JPG, PNG, TIF, TIFF)...", type=["jpg", "jpeg", "png", "tif", "tiff"])

col1, col2, col3 = st.columns([2,2,1]) # Input Image | Segmentation | Vectorization

# Session state to store results
if 'segmented_mask_tensor_nchw' not in st.session_state:
    st.session_state.segmented_mask_tensor_nchw = None
if 'original_georef_info' not in st.session_state:
    st.session_state.original_georef_info = None


if uploaded_file is not None and model is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    is_geotiff = file_extension in [".tif", ".tiff"] and RASTERIO_GEOPANDAS_AVAILABLE

    input_tensor_for_model = None
    display_image_pil_or_np = None

    try:
        if is_geotiff:
            with col1: st.subheader("Uploaded GeoTIFF")
            bytes_io_file = io.BytesIO(uploaded_file.getvalue())
            with rasterio.open(bytes_io_file) as src:
                st.session_state.original_georef_info = {
                    "transform": src.transform, "crs": src.crs,
                    "width": src.width, "height": src.height,
                    "count": src.count, "dtype": str(src.dtypes[0]) if src.dtypes else "unknown"
                }
                display_bands_np = src.read(range(1, min(src.count, 3) + 1) if src.count > 0 else None)

                if display_bands_np is not None and display_bands_np.size > 0:
                    if display_bands_np.shape[0] == 1:
                        gray_band = display_bands_np[0].astype(np.float32)
                        p2, p98 = np.percentile(gray_band, (2,98))
                        vis_np = np.clip((gray_band - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
                        display_image_pil_or_np = Image.fromarray(vis_np.astype(np.uint8), 'L')
                    else:
                        rgb_bands_for_vis = []
                        for i in range(display_bands_np.shape[0]):
                            band = display_bands_np[i].astype(np.float32)
                            p2, p98 = np.percentile(band, (2,98))
                            scaled_band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
                            rgb_bands_for_vis.append(scaled_band.astype(np.uint8))
                        vis_np_cwh = np.stack(rgb_bands_for_vis)
                        display_image_pil_or_np = Image.fromarray(vis_np_cwh.transpose(1,2,0), 'RGB')
                    st.image(display_image_pil_or_np, caption=f"GeoTIFF Preview (Original: {src.width}x{src.height}, {src.count} bands, {src.dtypes[0]})", use_container_width=True)
                else:
                    st.warning("Could not generate a preview for the GeoTIFF.")
                input_tensor_for_model = preprocess_geotiff_data_for_model(src.read(), IMAGE_HEIGHT, IMAGE_WIDTH, IMG_MEAN, IMG_STD)
        else: # JPG, PNG
            display_image_pil_or_np = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.subheader("Uploaded Image")
                st.image(display_image_pil_or_np, use_container_width=True)
            input_tensor_for_model = preprocess_pil_image(display_image_pil_or_np, preprocess_transform_rgb)
            st.session_state.original_georef_info = None

        if st.button("Segment Image", key="segment_button"):
            if input_tensor_for_model is None:
                st.error("Could not preprocess the input image.")
            else:
                st.session_state.segmented_mask_tensor_nchw = None
                start_time = time.time()
                with st.spinner("Running segmentation model..."):
                    output_logits = predict(model, input_tensor_for_model, DEVICE)
                st.session_state.segmented_mask_tensor_nchw = get_mask_tensor(output_logits, THRESHOLD)
                processing_time = time.time() - start_time
                segmented_mask_np_hw_display = st.session_state.segmented_mask_tensor_nchw.squeeze().cpu().numpy()

                with col2:
                    st.subheader("Segmentation Result")
                    st.image(segmented_mask_np_hw_display, caption=f"Segmentation Mask (Model Output: {IMAGE_WIDTH}x{IMAGE_HEIGHT})", clamp=True, use_container_width=True)
                    st.success(f"Segmentation done in {processing_time:.2f}s.")
                    mask_img_pil = Image.fromarray((segmented_mask_np_hw_display * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    mask_img_pil.save(buf, format="PNG")
                    st.download_button(label="Download Mask (PNG)", data=buf.getvalue(), file_name="segmented_mask.png", mime="image/png")

        if st.session_state.segmented_mask_tensor_nchw is not None and \
           st.session_state.original_georef_info and \
           RASTERIO_GEOPANDAS_AVAILABLE:
            with col3:
                st.subheader("Vectorization")
                st.write("Convert segmentation to vector file.")
                st.caption(f"Original CRS: {st.session_state.original_georef_info['crs']}")
                output_format_ext = st.selectbox("Vector Output Format:", [".shp", ".geojson", ".gpkg"], key="vec_format_ext")

                if st.button("Create & Download Vector", key="vectorize_button_dl"):
                    with st.spinner("Vectorizing mask and preparing download..."):
                        resized_mask_tensor_for_vector = transforms.functional.resize(
                            st.session_state.segmented_mask_tensor_nchw,
                            [st.session_state.original_georef_info["height"], st.session_state.original_georef_info["width"]],
                            interpolation=transforms.InterpolationMode.NEAREST
                        )
                        temp_dir = "temp_vector_output" # Consider using tempfile module for true temp dirs
                        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                        vec_filename_base = f"vectorized_roads_{int(time.time())}"
                        temp_vector_path = os.path.join(temp_dir, vec_filename_base + output_format_ext)

                        success, saved_vector_path = raster_to_vector_geopandas(
                            resized_mask_tensor_for_vector,
                            st.session_state.original_georef_info["transform"],
                            st.session_state.original_georef_info["crs"],
                            temp_vector_path,
                            road_pixel_value=1
                        )
                        if success and saved_vector_path and os.path.exists(saved_vector_path) and os.path.getsize(saved_vector_path) > 0:
                             # Check if the file has content beyond just being empty (especially for shapefiles)
                            is_shp_empty = False
                            if output_format_ext == ".shp":
                                try:
                                    gdf_check = gpd.read_file(saved_vector_path)
                                    if gdf_check.empty:
                                        is_shp_empty = True
                                except Exception: # Handle cases where read_file might fail on truly empty/malformed shp
                                    is_shp_empty = True # Assume empty if cannot read
                            
                            if is_shp_empty or (os.path.getsize(saved_vector_path) < 150 and output_format_ext != ".shp"): # Heuristic for empty non-shp
                                st.info("Vectorization complete, but no road features were found in the mask.")
                                # Still offer download for the (potentially empty) file structure
                                if output_format_ext == ".shp": # Zip even if empty structure
                                     zip_buffer = io.BytesIO()
                                     with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                        for ext_comp in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                            comp_file = os.path.join(temp_dir, vec_filename_base + ext_comp)
                                            if os.path.exists(comp_file):
                                                zf.write(comp_file, arcname=os.path.basename(comp_file))
                                     download_file_name = f"{vec_filename_base}.zip"
                                     download_mime = "application/zip"
                                     download_data = zip_buffer.getvalue()
                                     st.download_button(label=f"Download Empty {download_file_name}",data=download_data,file_name=download_file_name,mime=download_mime)
                                else: # GeoJSON, GPKG empty
                                    with open(saved_vector_path, "rb") as fp_dl:
                                        st.download_button(label=f"Download Empty {os.path.basename(saved_vector_path)}", data=fp_dl.read(), file_name=os.path.basename(saved_vector_path), mime="application/octet-stream")

                            else: # File has content
                                if output_format_ext == ".shp":
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                        for ext_comp in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                            comp_file = os.path.join(temp_dir, vec_filename_base + ext_comp)
                                            if os.path.exists(comp_file):
                                                zf.write(comp_file, arcname=os.path.basename(comp_file))
                                    download_file_name = f"{vec_filename_base}.zip"
                                    download_mime = "application/zip"
                                    download_data = zip_buffer.getvalue()
                                else:
                                    with open(saved_vector_path, "rb") as fp_dl:
                                        download_data = fp_dl.read()
                                    download_file_name = os.path.basename(saved_vector_path)
                                    download_mime = "application/octet-stream"

                                st.download_button(
                                    label=f"Download {download_file_name}",
                                    data=download_data,
                                    file_name=download_file_name,
                                    mime=download_mime
                                )
                        elif success and saved_vector_path and os.path.exists(saved_vector_path): # File exists but might be empty
                             st.info("Vectorization complete. The file might be empty if no road features were found.")
                             # Offer download anyway
                             with open(saved_vector_path, "rb") as fp_dl:
                                st.download_button(label=f"Download {os.path.basename(saved_vector_path)}", data=fp_dl.read(), file_name=os.path.basename(saved_vector_path), mime="application/octet-stream")
                        else:
                            st.error("Vector file creation failed or file not found after process.")


        elif st.session_state.segmented_mask_tensor_nchw is not None:
            with col3:
                st.subheader("Vectorization")
                st.info("Vectorization is available only for GeoTIFF inputs to ensure correct georeferencing.")

    except rasterio.errors.RasterioIOError as rie:
        st.error(f"Error reading/processing the GeoTIFF file: {rie}.")
        st.session_state.original_georef_info = None
        st.session_state.segmented_mask_tensor_nchw = None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.session_state.original_georef_info = None
        st.session_state.segmented_mask_tensor_nchw = None

elif model is None and uploaded_file is not None: # Model failed to load but user uploaded file
    st.error("Model is not loaded. Cannot perform segmentation.")

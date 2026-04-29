import streamlit as st
import cv2
import numpy as np
import os
import time
import io
import zipfile
import base64
import logging
import concurrent.futures
from datetime import datetime
from PIL import Image
import requests
from cleaner import clean_and_restored_optimized

try:
    import pytesseract
    HAS_PYTESSERACT = True
    # --- Local Tessdata Configuration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    local_tessdata = os.path.join(PARENT_DIR, "tessdata")
    if os.path.exists(local_tessdata):
        os.environ["TESSDATA_PREFIX"] = local_tessdata
except ImportError:
    HAS_PYTESSERACT = False

# --- Setup Logging ---
LOG_FILE = "app_activity.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(msg, level="INFO"):
    if level == "ERROR": logging.error(msg)
    else: logging.info(msg)

# Increase PIL limit for very high-resolution scans
Image.MAX_IMAGE_PIXELS = None 

st.set_page_config(page_title="Image Cleaner Pro", page_icon="🗞️", layout="wide")

# --- Initialize Session State ---
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {} # {filename: {data: bytes, duration: float, mode: str, text: str}}
if 'result_rotations' not in st.session_state:
    st.session_state.result_rotations = {} # {filename: deg}
if 'previews' not in st.session_state:
    st.session_state.previews = {} # {filename: b64_preview}
if 'show_original' not in st.session_state:
    st.session_state.show_original = {} # {filename: bool}

# --- Helper Functions ---
def get_preview(file, filename):
    """Generates and caches a small preview to avoid re-decoding 250MB images on every rerun."""
    try:
        if filename in st.session_state.previews:
            return st.session_state.previews[filename]
        
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None: 
            log_event(f"Failed to decode preview for {filename}", "ERROR")
            return None
        
        # Create small preview (max 1200px)
        h, w = img.shape[:2]
        scale = max(1, max(h, w) / 1200)
        preview_img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', preview_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_str = base64.b64encode(buffer).decode()
        st.session_state.previews[filename] = b64_str
        return b64_str
    except Exception as e:
        log_event(f"Preview error for {filename}: {str(e)}", "ERROR")
        return None

def process_single_task(task_data):
    """Worker function for multiprocessing."""
    i, fname, file_bytes, restore_mode, do_upscale = task_data
    try:
        tmp_in = f"tmp_in_{i}_{fname}.png"
        tmp_out = f"tmp_out_{i}_{fname}.png"
        
        # Write bytes to disk for cv2
        with open(tmp_in, "wb") as f:
            f.write(file_bytes)
            
        start = time.time()
        from cleaner import clean_and_restored_optimized, super_resolve_image
        success = clean_and_restored_optimized(tmp_in, tmp_out, do_colorize=(restore_mode == "AI Colorized (Full Restore)"))
        
        # Apply Upscaling if requested
        if success and do_upscale:
            res_img = cv2.imread(tmp_out)
            if res_img is not None:
                upscaled = super_resolve_image(res_img, scale=2)
                cv2.imwrite(tmp_out, upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        duration = time.time() - start
        
        res_data = None
        if success:
            with open(tmp_out, "rb") as f:
                res_data = f.read()
        
        # Cleanup
        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_out): os.remove(tmp_out)
        
        return (i, fname, res_data, duration, success, None)
    except Exception as e:
        return (i, fname, None, 0, False, str(e))

def run_ocr_task(image_data, tesseract_cmd, lang="eng"):
    """Helper to run OCR in a separate thread if needed."""
    if not HAS_PYTESSERACT:
        return "[OCR Error: pytesseract module not installed. Please run setup.bat]"
        
    try:
        from cleaner import extract_text
        # Save bytes to temp file for OCR
        tmp_ocr = "tmp_ocr_process.png"
        with open(tmp_ocr, "wb") as f:
            f.write(image_data)
        
        text = extract_text(tmp_ocr, tesseract_cmd, lang=lang)
        if os.path.exists(tmp_ocr): os.remove(tmp_ocr)
        return text
    except Exception as e:
        return f"[OCR Error: {e}]"

def run_ocr_pdf_task(image_data, tesseract_cmd, lang="eng"):
    """Generates a Searchable PDF bytes."""
    if not HAS_PYTESSERACT: return None
    try:
        from cleaner import generate_searchable_pdf
        tmp_ocr = "tmp_ocr_pdf.png"
        with open(tmp_ocr, "wb") as f:
            f.write(image_data)
        pdf_bytes = generate_searchable_pdf(tmp_ocr, tesseract_cmd, lang=lang)
        if os.path.exists(tmp_ocr): os.remove(tmp_ocr)
        return pdf_bytes
    except: return None

def download_ocr_data(lang_code):
    """Downloads Tesseract language data directly to the project folder."""
    base_url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/"
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tessdata")
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    
    target_path = os.path.join(target_dir, f"{lang_code}.traineddata")
    url = f"{base_url}{lang_code}.traineddata"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# --- Callbacks for Fragments ---
def rotate_callback(storage_key):
    st.session_state.result_rotations[storage_key] = (st.session_state.result_rotations.get(storage_key, 0) + 90) % 360

@st.cache_data
def generate_batch_zip(processed_images, result_rotations, download_format):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zip_file:
        for skey, res in processed_images.items():
            nparr = np.frombuffer(res['data'], np.uint8)
            img_cv = cv2.imdecode(nparr, 1)
            
            # Apply Rotation to Result
            rot = result_rotations.get(skey, 0)
            if rot == 90: img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)
            elif rot == 270: img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            img_buf = io.BytesIO()
            fname = skey.rsplit('_', 1)[0]
            base_name = os.path.splitext(fname)[0]
            
            if download_format == "JPG":
                pil_img.save(img_buf, format="JPEG", quality=95)
                ext = "jpg"
            elif download_format == "PNG":
                pil_img.save(img_buf, format="PNG")
                ext = "png"
            else: # PDF
                pil_img.save(img_buf, format="PDF", resolution=300.0)
                ext = "pdf"
            
            zip_file.writestr(f"restored_{base_name}.{ext}", img_buf.getvalue())
    return zip_buf.getvalue()

@st.fragment
def display_image_card(i, up_file, restore_mode, download_format):
    fname = up_file.name
    storage_key = f"{fname}_{i}"
    st.divider()
    col1, col2 = st.columns(2)
    
    # Original Preview (CACHED)
    with col1:
        st.subheader(f"📄 Original: {fname}")
        b64_preview = get_preview(up_file, fname)
        if b64_preview:
            st.image(f"data:image/jpeg;base64,{b64_preview}", width="stretch")

    # Result View
    with col2:
        st.subheader("✨ Restored Result")
        if storage_key in st.session_state.processed_images:
            res = st.session_state.processed_images[storage_key]
            
            # --- APPLY ROTATION TO RESULT ONLY (INSTANT) ---
            nparr = np.frombuffer(res['data'], np.uint8)
            res_img = cv2.imdecode(nparr, 1)
            
            rot = st.session_state.result_rotations.get(storage_key, 0)
            if rot == 90: res_img = cv2.rotate(res_img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: res_img = cv2.rotate(res_img, cv2.ROTATE_180)
            elif rot == 270: res_img = cv2.rotate(res_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # --- BEFORE / AFTER TOGGLE ---
            view_mode = st.radio("View Perspective", ["Restored Result", "Original Scan"], 
                                 key=f"view_{storage_key}", horizontal=True)
            
            if view_mode == "Original Scan":
                b64_orig = get_preview(up_file, fname)
                st.image(f"data:image/jpeg;base64,{b64_orig}", width="stretch")
            else:
                # Preview Result
                rh, rw = res_img.shape[:2]
                r_scale = max(1, max(rh, rw) / 1200)
                r_preview = cv2.resize(res_img, (int(rw / r_scale), int(rh / r_scale)), interpolation=cv2.INTER_AREA)
                st.image(r_preview, channels="BGR", width="stretch")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.button(f"🔄 Rotate 90°", key=f"rot_res_{fname}_{i}", 
                          on_click=rotate_callback, args=(storage_key,))
            
            # --- OCR DISPLAY & Searchable PDF ---
            if 'text' in res and res['text']:
                with st.expander("📝 Extracted Text & Searchable Archive"):
                    st.text_area("Structured OCR Result", res['text'], height=250, key=f"ocr_text_{storage_key}")
                    
                    if st.button("📄 Generate Searchable PDF (Overlay)", key=f"gen_pdf_{storage_key}"):
                        with st.spinner("Generating OCR PDF..."):
                            pdf_bytes = run_ocr_pdf_task(res['data'], tesseract_path, lang=ocr_lang)
                            if pdf_bytes:
                                st.download_button(
                                    "📥 Ready! Download Searchable PDF",
                                    data=pdf_bytes,
                                    file_name=f"archive_{fname}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{storage_key}"
                                )
            elif do_ocr:
                if st.button("🔍 Run OCR Now", key=f"ocr_btn_{storage_key}"):
                    with st.spinner(f"Extracting {ocr_lang_label} text..."):
                        text = run_ocr_task(res['data'], tesseract_path, lang=ocr_lang)
                        st.session_state.processed_images[storage_key]['text'] = text
                        st.rerun(scope="fragment")

            # Dynamic Download
            res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(res_img_rgb)
            buf = io.BytesIO()
            base_name = os.path.splitext(fname)[0]
            if download_format == "JPG":
                pil_img.save(buf, format="JPEG", quality=95)
                final_data = buf.getvalue(); mime_type = "image/jpeg"; ext = "jpg"
            elif download_format == "PNG":
                pil_img.save(buf, format="PNG")
                final_data = buf.getvalue(); mime_type = "image/png"; ext = "png"
            else: # PDF
                pil_img.save(buf, format="PDF", resolution=300.0)
                final_data = buf.getvalue(); mime_type = "application/pdf"; ext = "pdf"
            
            size_mb = len(final_data) / (1024 * 1024)
            st.write(f"✅ Mode: {res['mode']} | ⏱️ {res['duration']:.1f}s | 📦 {size_mb:.2f} MB")
            st.download_button(
                label=f"📥 Download {download_format}",
                data=final_data,
                file_name=f"restored_{base_name}.{ext}",
                mime=mime_type,
                key=f"dl_{fname}_{i}"
            )
        else:
            st.info("Click 'Process All' or 'Restore' to start.")
            if st.button(f"Restore {fname} Only", key=f"single_{fname}_{i}"):
                try:
                    up_file.seek(0)
                    file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    if img is None: raise ValueError("Could not decode image.")

                    tmp_in = f"tmp_in_{i}_{fname}.png"
                    tmp_out = f"tmp_out_{i}_{fname}.png"
                    cv2.imwrite(tmp_in, img)
                    
                    start = time.time()
                    success = clean_and_restored_optimized(tmp_in, tmp_out, do_colorize=(restore_mode == "AI Colorized (Full Restore)"))
                    
                    if success and do_upscale:
                        from cleaner import super_resolve_image
                        res_img = cv2.imread(tmp_out)
                        if res_img is not None:
                            upscaled = super_resolve_image(res_img, scale=2)
                            cv2.imwrite(tmp_out, upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                            
                    duration = time.time() - start
                    
                    if success:
                        with open(tmp_out, "rb") as f:
                            st.session_state.processed_images[storage_key] = {
                                'data': f.read(), 'duration': duration, 'mode': restore_mode
                            }
                        log_event(f"Successfully restored {fname} in {duration:.1f}s")
                    else:
                        st.error(f"Restoration failed for {fname}. Check logs.")
                        log_event(f"Restoration failed for {fname}", "ERROR")

                    if os.path.exists(tmp_in): os.remove(tmp_in)
                    if os.path.exists(tmp_out): os.remove(tmp_out)
                    st.rerun(scope="fragment")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    log_event(f"Single process error for {fname}: {str(e)}", "ERROR")

# --- Header ---
st.title("🗞️ Historical Document Cleaner Pro")
st.markdown("Professional AI Restoration for high-resolution newspaper archives.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Restoration Settings")
    restore_mode = st.radio(
        "Quality Mode", 
        ["Gold Standard (B&W)", "AI Colorized (Full Restore)"], 
        index=1,
        help="v3 Gold Standard is perfect for text. AI Colorized adds realistic colors to photos."
    )
    
    auto_crop = st.toggle("Auto-Remove Borders", value=True, help="Automatically detects and crops out black scanner borders.")
    
    st.divider()
    st.header("💎 Quality Upgrades (Free)")
    do_upscale = st.toggle("Super-Resolution (2x Upscale)", value=False, help="Doubles image resolution for high-quality printing.")
    
    st.divider()
    st.header("🔍 OCR Settings (Free)")
    if HAS_PYTESSERACT:
        do_ocr = st.toggle("Enable Searchable Text (OCR)", value=False)
        
        langs = {"English": "eng", "Tamil": "tam", "English + Tamil": "eng+tam"}
        ocr_lang_label = st.selectbox("Document Language", options=list(langs.keys()), index=0)
        ocr_lang = langs[ocr_lang_label]
        
        tesseract_path = st.text_input(
            "Tesseract Path (Windows)", 
            value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            help="Download Tesseract-OCR via setup.bat to use this feature."
        )
        
        if ocr_lang_label != "English":
            local_tam = os.path.join(local_tessdata, "tam.traineddata")
            if not os.path.exists(local_tam):
                st.warning("⚠️ Tamil language data missing in local 'tessdata' folder.")
                if st.button("📥 Download Tamil Data Now (Best Quality)"):
                    with st.spinner("Downloading Tamil OCR pack (~15MB)..."):
                        if download_ocr_data("tam"):
                            st.success("Tamil Data installed! Please click 'Run OCR Now' again.")
                            st.rerun()
            else:
                st.info("💡 Tip: Using local 'tam.traineddata' from project folder.")
            
            st.caption("🗞️ *Newspaper Mode Activated:* High-accuracy layout analysis is now used to detect columns and fix tilted text.")
    else:
        st.warning("⚠️ OCR Disabled: 'pytesseract' module not found.")
        st.info("To fix this, please close the app and run **setup.bat** in the project folder.")
        do_ocr = False
        tesseract_path = ""
    
    st.divider()
    if st.button("🗑️ Reset All / Clear Cache"):
        st.session_state.processed_images = {}
        st.session_state.previews = {}
        st.session_state.result_rotations = {}
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.header("💾 Download Settings")
    download_format = st.radio(
        "Preferred Format",
        ["JPG", "PNG", "PDF"],
        index=0,
        help="JPG is smallest. PNG is best for text. PDF is best for documents."
    )
    
    st.divider()
    if st.session_state.processed_images:
        st.header("📦 Batch Download")
        zip_data = generate_batch_zip(
            st.session_state.processed_images, 
            st.session_state.result_rotations, 
            download_format
        )
        
        st.download_button(
            label=f"📥 Download All ({len(st.session_state.processed_images)} files)",
            data=zip_data,
            file_name=f"restored_archive_{download_format.lower()}.zip",
            mime="application/zip",
            width="stretch",
            key="download_all_button"
        )

    st.divider()
    st.markdown("### 🚀 Speed Info")
    st.info("Performance: Previews are cached for instant interaction.")

    with st.expander("📄 Activity Log"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                logs = f.readlines()
                st.text("".join(logs[-15:])) # Show last 15 lines
        else:
            st.write("No logs yet.")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1f77b4; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #28a745; color: white; }
    .stAppDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
uploaded_files = st.file_uploader(
    "Upload your newspaper scans (JPG/PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Action for Batch Processing
    if st.button("✨ Process All Images (Speed Boost)", width="stretch"):
        log_event(f"Starting parallel batch process for {len(uploaded_files)} images.")
        
        # 1. Prepare tasks for images that need processing
        tasks = []
        for i, up_file in enumerate(uploaded_files):
            fname = up_file.name
            storage_key = f"{fname}_{i}"
            needs_processing = (storage_key not in st.session_state.processed_images or 
                                st.session_state.processed_images[storage_key]['mode'] != restore_mode)
            if needs_processing:
                up_file.seek(0)
                tasks.append((i, fname, up_file.read(), restore_mode, do_upscale))

        if not tasks:
            st.info("Everything is already processed for this mode.")
        else:
            progress_bar = st.progress(0, text=f"Processing {len(tasks)} images in parallel...")
            
            # 2. Execute in Parallel
            # Using ProcessPoolExecutor for CPU-bound tasks (OpenCV)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Map tasks to worker
                futures = [executor.submit(process_single_task, t) for t in tasks]
                
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, fname, res_data, duration, success, err = future.result()
                        storage_key = f"{fname}_{i}"
                        
                        if success:
                            st.session_state.processed_images[storage_key] = {
                                'data': res_data, 'duration': duration, 'mode': restore_mode
                            }
                            log_event(f"Successfully restored {fname} in {duration:.1f}s")
                        else:
                            log_event(f"Parallel failure for {fname}: {err}", "ERROR")
                    except Exception as exc:
                        log_event(f"Future error: {exc}", "ERROR")
                    
                    completed += 1
                    progress_bar.progress(completed / len(tasks), text=f"Finished {completed}/{len(tasks)} images...")
            
            progress_bar.empty()
            st.success(f"Batch processing complete! ({len(tasks)} images)")
            time.sleep(1)
            st.rerun()

    # Display results
    for i, up_file in enumerate(uploaded_files):
        display_image_card(i, up_file, restore_mode, download_format)
else:
    st.info("👆 Please upload newspaper scans to begin.")

# --- Footer ---
st.divider()
st.caption("Developed by Sh | Historical Document Restoration V4.0")

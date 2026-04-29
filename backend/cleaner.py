import cv2
import numpy as np
import os

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

# Paths for the colorization model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROTOTXT = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODELS_DIR, "pts_in_hull.npy")

def remove_black_borders(img):
    """
    Improved: Detects the paper area and crops out black/near-black scanner borders.
    Uses morphological cleaning to avoid noise-induced mis-crops.
    """
    try:
        h_orig, w_orig = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur to remove scan noise which can trip up thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Threshold: anything above a low 'paper' value is considered data
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        # 3. Morphological closing to fill small holes in the paper area
        kernel = np.ones((11, 11), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Find the largest contour (the page)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
            
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # 5. Sanity Check: If the crop is tiny (less than 10% of original), ignore it
        if w < w_orig * 0.1 or h < h_orig * 0.1:
            return img

        # 6. Add a small margin to avoid cutting edges
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w_orig - x, w + 2*margin)
        h = min(h_orig - y, h + 2*margin)
        
        return img[y:y+h, x:x+w]
    except Exception as e:
        print(f"Border detection failed: {e}")
        return img

def colorize_image_segment_fast(img_gray):
    """
    Faster AI colorization using internal downscaling for inference.
    """
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    h, w = img_gray.shape[:2]
    
    # PERFORMANCE BOOST: If the photo is large, downscale it for inference
    # and upscale the 'ab' channels later.
    inf_h, inf_w = 224, 224 # Model fixed size
    
    # 1. Convert to Lab
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    scaled = img_rgb.astype("float32") / 255.0
    lab_full = cv2.cvtColor(scaled, cv2.COLOR_RGB2Lab)
    
    # 2. Prepare L channel for model
    L_full = cv2.split(lab_full)[0]
    L_resized = cv2.resize(L_full, (224, 224))
    L_resized -= 50

    # 3. Predict ab channels (Inference)
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab_small = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # 4. Upscale ab channels back to original size
    ab_full = cv2.resize(ab_small, (w, h))

    # 5. Recombine and convert back
    colorized_lab = np.concatenate((L_full[:, :, np.newaxis], ab_full), axis=2)
    colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2RGB)
    colorized_rgb = (255 * np.clip(colorized_rgb, 0, 1)).astype("uint8")
    
    return cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)

def clean_and_restored_optimized(image_path, output_path, do_colorize=True):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return False

        # A. Remove black borders first
        img = remove_black_borders(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.GaussianBlur(gray, (101, 101), 0)
        normalized = cv2.divide(gray, bg, scale=255)

        # 1. Sharp Bold Text
        text_sharp = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 20
        )
        text_sharp = cv2.medianBlur(text_sharp, 3)
        text_sharp = cv2.erode(text_sharp, np.ones((2, 2), np.uint8), iterations=1)

        # 2. Enhanced Image Details
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        img_soft = clahe.apply(normalized)
        img_soft_deep = cv2.convertScaleAbs(img_soft, alpha=1.1, beta=-20)

        # 3. Layout Sensing
        kernel = np.ones((25, 25), np.uint8)
        _, binary_ink = cv2.threshold(normalized, 200, 255, cv2.THRESH_BINARY_INV)
        closed = cv2.morphologyEx(binary_ink, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        image_mask = np.zeros_like(gray)
        image_regions = []
        page_h, page_w = gray.shape
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (200 < w < page_w * 0.8) and (200 < h < page_h * 0.8):
                 cv2.drawContours(image_mask, [cnt], -1, 255, -1)
                 image_regions.append((x, y, w, h))
        
        # Create a blurred mask for smooth blending (Keep as uint8 to save memory)
        mask_blurred = cv2.GaussianBlur(image_mask, (21, 21), 0)

        # 4. Fast Colorization
        colored_final = cv2.cvtColor(img_soft_deep, cv2.COLOR_GRAY2BGR)
        if do_colorize and image_regions:
            for x, y, w, h in image_regions:
                section = img_soft_deep[y:y+h, x:x+w]
                try:
                    # Use the fast colorization method
                    colored_section = colorize_image_segment_fast(section)
                    colored_final[y:y+h, x:x+w] = colored_section
                except Exception as e:
                    print(f"Colorization failed for region {x, y}: {e}")

        # 5. Composite (Memory-Efficient Local Blending)
        # Start with the sharp B&W text as the base
        final = cv2.cvtColor(text_sharp, cv2.COLOR_GRAY2BGR)
        
        if do_colorize and image_regions:
            # Instead of a global float32 blend (which crashes on large images),
            # we only blend the specific regions that were colorized.
            for x, y, w, h in image_regions:
                # Add a small margin to include the blurred edge of the mask
                margin = 25
                y1, y2 = max(0, y - margin), min(page_h, y + h + margin)
                x1, x2 = max(0, x - margin), min(page_w, x + w + margin)
                
                # Extract local slices
                m_local = mask_blurred[y1:y2, x1:x2, np.newaxis].astype(np.float32) / 255.0
                t_local = final[y1:y2, x1:x2].astype(np.float32)
                c_local = colored_final[y1:y2, x1:x2].astype(np.float32)
                
                # Blend locally
                blended = (1.0 - m_local) * t_local + m_local * c_local
                final[y1:y2, x1:x2] = blended.astype(np.uint8)

        # 6. Final White Cleaning
        mask_white = np.all(final > 250, axis=2)
        final[mask_white] = [255, 255, 255]

        # Save
        cv2.imwrite(output_path, final, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return True
    except Exception as e:
        print(f"Processing error for {image_path}: {e}")
        return False

def extract_text(image_path, tesseract_cmd=None, lang="eng"):
    """
    Extracts text from the cleaned image using block detection to preserve structure.
    Filters out low-confidence noise frequently found in old scans.
    """
    if not HAS_PYTESSERACT:
        return "[OCR Error: pytesseract module not installed]"
        
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        img = cv2.imread(image_path)
        if img is None: return ""
        
        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 2. Get Data with Confidence (Detailed Structure)
        # Using output_type=Output.DICT to get block/par/line info
        from pytesseract import Output
        data = pytesseract.image_to_data(binary, lang=lang, config=r'--psm 1', output_type=Output.DICT)
        
        # 3. reconstruct text by blocks, filtering out noise
        full_text = []
        current_block = -1
        current_line = []
        
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            
            # Filter noise: low confidence or single symbols that aren't letters
            if conf < 15 or not text: continue
            
            if data['block_num'][i] != current_block:
                if current_line: full_text.append(" ".join(current_line))
                if full_text: full_text.append("\n\n") # Double space between blocks
                current_block = data['block_num'][i]
                current_line = []
            
            current_line.append(text)
            
        if current_line: full_text.append(" ".join(current_line))
            
        return "".join(full_text).strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return f"[OCR Error: {e}]"

def generate_searchable_pdf(image_path, tesseract_cmd=None, lang="eng"):
    """
    Generates a Searchable PDF version of the image where text is overlaid.
    """
    if not HAS_PYTESSERACT: return None
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Tesseract natively can output PDF bytes
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(image_path, lang=lang, extension='pdf')
        return pdf_bytes
    except Exception as e:
        print(f"PDF OCR failed: {e}")
        return None

def super_resolve_image(img, scale=2):
    """
    Upscales the image using high-quality interpolation for better print quality.
    (Optional: can be expanded to use AI models via cv2.dnn_superres)
    """
    try:
        h, w = img.shape[:2]
        # Use INTER_CUBIC or INTER_LANCZOS4 for high-quality upscaling
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        upscaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
        return upscaled
    except Exception as e:
        print(f"Upscaling error: {e}")
        return img

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cleaner.py <input> <output>")
    else:
        clean_and_restored_optimized(sys.argv[1], sys.argv[2])

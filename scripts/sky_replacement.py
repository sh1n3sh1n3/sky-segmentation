"""
Full Sky + Human Segmentation Pipeline
---------------------------------------
1. Human segmentation (Robust Video Matting)
2. Sky segmentation (NCNN or color fallback)
3. Mask refinement via mask_refine.exe
4. Merge human mask with refined sky mask (human = black)
"""

import os, cv2, subprocess, numpy as np
from tqdm import tqdm
import torch

# Flag to check if ncnn is available
NCNN_AVAILABLE = False

try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    print("Warning: ncnn not available, will use color-based detection")
    NCNN_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"F:\Studio\sky_replacement_project"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SKY_MASK_DIR = os.path.join(BASE_DIR, "data", "sky_masks")
RUN_DIR = os.path.join(BASE_DIR, "run")
MASK_EXE = os.path.join(RUN_DIR, "mask_refine.exe")
HUMAN_MODEL = "mobilenetv3"   # RVM backbone

os.makedirs(SKY_MASK_DIR, exist_ok=True)

# ============================================================
# LOAD ROBUST VIDEO MATTING (RVM)
# ============================================================
def load_rvm_model(device):
    print("🔹 Loading Robust Video Matting model...")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", HUMAN_MODEL)
    model.eval().to(device)
    return model

# ============================================================
# STEP 1: HUMAN SEGMENTATION (exact boundary)
# ============================================================
def segment_human(image_path, output_mask_path, device, model, target_long=1920):
    """Generate human mask (black human, white background)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    h, w = img.shape[:2]
    ds = min(1.0, float(target_long) / float(max(h, w)))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    rec = [None, None, None, None]
    with torch.no_grad():
        _, pha, *_ = model(src, *rec, downsample_ratio=ds)

    pha = (pha[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    inv_mask = cv2.bitwise_not(pha)   # human = black, bg = white
    cv2.imwrite(output_mask_path, inv_mask)
    return inv_mask

# ============================================================
# NCNN MODEL LOADING

# ============================================================
def load_ncnn_model(param_path, bin_path):
    """Load NCNN model"""
    if not NCNN_AVAILABLE:
        return None
    
    if not os.path.exists(param_path) or not os.path.exists(bin_path):
        print(f"NCNN model files not found at:")
        print(f"  {param_path}")
        print(f"  {bin_path}")
        print(f"Falling back to color-based detection...")
        return None
    
    import ncnn
    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    
    return net

# ============================================================
# STEP 2: SKY DETECTION (NCNN or fallback color)
# ============================================================
def detect_sky_color_based(image_path, output_path):
    """Simple color-based sky detector (fallback if NCNN unavailable)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([95,30,80]), np.array([125,255,255])
    lower_white, upper_white = np.array([0,0,180]), np.array([180,40,255])
    lower_cyan, upper_cyan = np.array([85,15,140]), np.array([100,80,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask |= cv2.inRange(hsv, lower_white, upper_white)
    mask |= cv2.inRange(hsv, lower_cyan, upper_cyan)

    pos_weight = np.linspace(1, 0, h).reshape(h, 1)
    combined = (mask.astype(np.float32)/255.0)*0.6 + pos_weight*0.4
    _, mask = cv2.threshold((combined*255).astype(np.uint8), 140, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(output_path, mask)
    return mask

def segment_sky_ncnn(image_path, output_path, net):
    """
    Segment sky using NCNN model or fallback to color-based
    """
    # If NCNN not available or net is None, use color-based
    if not NCNN_AVAILABLE or net is None:
        return detect_sky_color_based(image_path, output_path)
    
    import ncnn
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_height, original_width = img.shape[:2]
    
    try:
        # Preprocess - Based on official ncnn_interence.cpp
        # Input size is 384x384
        mat_in = ncnn.Mat.from_pixels_resize(
            img.data,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            original_width,
            original_height,
            384,
            384
        )
        
        # Normalize (same as official)
        mean_vals = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        norm_vals = [1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255)]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        # Run inference
        ex = net.create_extractor()
        ex.input("input.1", mat_in)
        
        mat_out = ncnn.Mat()
        ex.extract("1959", mat_out)
        
        # Post-process - Based on official implementation
        # Output is 384x384 float values
        opencv_mask = np.array(mat_out).reshape(mat_out.h, mat_out.w)
        
        # Resize to original size
        opencv_mask = cv2.resize(opencv_mask, (original_width, original_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Convert to 0-255 range (official: 255*opencv_mask)
        mask_resized = (opencv_mask * 255).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
        
        # Save mask
        cv2.imwrite(output_path, mask_resized)
        
        return mask_resized
        
    except Exception as e:
        print(f"NCNN inference failed: {e}")
        print("Falling back to color-based detection...")
        return detect_sky_color_based(image_path, output_path)

# ============================================================
# STEP 3: REFINE SKY MASK
# ============================================================
def refine_mask_with_exe(original_image_path, initial_mask_path, refined_mask_path):
    if not os.path.exists(MASK_EXE):
        print(f"⚠️ {MASK_EXE} not found, skipping refinement")
        return False
    
    try:
        # Pass the desired output path directly - the exe will append _result.png
        subprocess.run(
            [MASK_EXE,
             os.path.abspath(original_image_path),
             os.path.abspath(initial_mask_path),
             os.path.abspath(refined_mask_path)],
            cwd=RUN_DIR, capture_output=True, text=True, check=True)
        
        # The actual output file will have _result.png appended
        actual_output_path = refined_mask_path + "_result.png"

        # Handle both possible output cases from the exe
        if os.path.exists(actual_output_path):
            try:
                # Overwrite target if it already exists (Windows-safe)
                os.replace(actual_output_path, refined_mask_path)
            except OSError:
                # Fallback: remove target then replace
                try:
                    if os.path.exists(refined_mask_path):
                        os.remove(refined_mask_path)
                except Exception:
                    pass
                os.replace(actual_output_path, refined_mask_path)
            print("✅ Refinement done")
            return True
        elif os.path.exists(refined_mask_path):
            # Some versions may write directly to the requested path
            print("✅ Refinement done")
            return True
        else:
            print(f"❌ Expected output file not found: {actual_output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print("❌ Refinement failed:", e)
        return False

# ============================================================
# STEP 3.5: BINARY MASK CLEANUP
# ============================================================
def binarize_mask(mask_path, threshold=190):
    """
    Convert mask to strict binary (only pure black and white).
    Any pixel not pure white becomes black.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read mask: {mask_path}")
        return False
    
    # Apply threshold to create strict binary mask
    # Values >= threshold become 255 (white), others become 0 (black)
    binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Save the binary mask
    cv2.imwrite(mask_path, binary_mask)
    print(f"🔧 Binary mask saved: {mask_path}")
    return True

# ============================================================
# STEP 3.6: BITWISE THRESHOLD AND MASK PROCESSING
# ============================================================
def bitwise_threshold_mask(mask_path, threshold=190):
    """
    Perform bitwise threshold operation on mask:
    - Values below threshold become black (0)
    - Values 100 or above become white (255)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read mask: {mask_path}")
        return None
    
    # Apply threshold: values >= threshold become 255, others become 0
    _, thresholded_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded_mask

def clip_black_section(thresholded_mask):
    """
    Clip (retain) only the black section of the thresholded result.
    Returns a mask where only black pixels (value 0) are preserved.
    """
    # Create a mask where only black pixels (value 0) are kept
    black_mask = np.zeros_like(thresholded_mask)
    black_mask[thresholded_mask == 0] = 255  # White where original was black
    
    return black_mask

def apply_bitwise_black_mask_to_refined(refined_mask_path, bitwise_black_mask, output_path):
    """
    Copy the bitwise black mask back onto the refined mask.
    The black areas from the bitwise operation will be applied to the refined mask.
    """
    refined_mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)
    if refined_mask is None:
        print(f"⚠️ Could not read refined mask: {refined_mask_path}")
        return False
    
    # Ensure masks have the same dimensions
    if refined_mask.shape != bitwise_black_mask.shape:
        bitwise_black_mask = cv2.resize(bitwise_black_mask, 
                                      (refined_mask.shape[1], refined_mask.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Apply the bitwise black mask: where bitwise_black_mask is white (255), 
    # set refined_mask to black (0)
    result_mask = refined_mask.copy()
    result_mask[bitwise_black_mask == 255] = 0
    
    # Save the result
    cv2.imwrite(output_path, result_mask)
    print(f"🔧 Bitwise processed mask saved: {output_path}")
    return True

def process_refined_mask_with_bitwise(refined_mask_path, output_path, threshold=190):
    """
    Complete bitwise processing pipeline:
    1. Apply bitwise threshold operation
    2. Clip black section
    3. Apply back to refined mask
    """
    # Step 1: Bitwise threshold operation
    thresholded_mask = bitwise_threshold_mask(refined_mask_path, threshold)
    if thresholded_mask is None:
        return False
    
    # Step 2: Clip black section
    black_section_mask = clip_black_section(thresholded_mask)
    
    # Step 3: Apply back to refined mask
    success = apply_bitwise_black_mask_to_refined(refined_mask_path, black_section_mask, output_path)
    
    if success:
        print(f"✅ Bitwise processing completed for: {os.path.basename(refined_mask_path)}")
    
    return success

# ============================================================
# STEP 4: MERGE SKY + HUMAN MASK
# ============================================================
def apply_human_mask_to_sky(refined_sky_path, human_mask_path, output_path):
    sky = cv2.imread(refined_sky_path, cv2.IMREAD_GRAYSCALE)
    human = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)
    if sky is None or human is None:
        print(f"⚠️ Missing mask for merge: {refined_sky_path}")
        return False
    if sky.shape != human.shape:
        human = cv2.resize(human, (sky.shape[1], sky.shape[0]), interpolation=cv2.INTER_NEAREST)

    # black human area → black on sky mask
    merged = sky.copy()
    merged[human < 190] = 0
    cv2.imwrite(output_path, merged)
    print(f"🤝 Combined mask saved: {output_path}")
    return True

# ============================================================
# STEP 5: REPLACE SKY USING FINAL RESULT MASK
# ============================================================
def replace_sky_using_final_mask(input_image_path, final_mask_path, output_image_path,
                                 sky_image_path=None, alpha_blend=True, mask_soften=True, use_poisson=True, feather_strength=25):
    """Replace sky in input image using the final combined mask and a provided sky image.

    Args:
        input_image_path: Path to the original input image.
        final_mask_path: Path to the final combined mask (white = sky, black = non-sky).
        output_image_path: Path where the composited result will be saved.
        sky_image_path: Optional custom sky image path. Defaults to data/skies/sky.png.
        alpha_blend: If True, uses soft alpha blending; otherwise hard paste.
        mask_soften: If True, slightly blurs mask edges for smoother blend.

    Returns:
        bool: True if success, False otherwise.
    """
    # Resolve default sky image path
    if sky_image_path is None:
        sky_image_path = os.path.join(BASE_DIR, "data", "skies", "sky.png")

    # Read inputs
    img = cv2.imread(input_image_path)
    sky_img = cv2.imread(sky_image_path)
    mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Cannot read input image: {input_image_path}")
        return False
    if sky_img is None:
        print(f"❌ Cannot read sky image: {sky_image_path}")
        return False
    if mask is None:
        print(f"❌ Cannot read final mask: {final_mask_path}")
        return False

    h, w = img.shape[:2]

    # Resize sky to match image WITHOUT stretching (preserve aspect ratio)
    if sky_img.shape[:2] != (h, w):
        sh, sw = sky_img.shape[:2]
        if sh == 0 or sw == 0:
            print("❌ Invalid sky image dimensions")
            return False
        scale = max(w / float(sw), h / float(sh))  # cover
        new_w = int(round(sw * scale))
        new_h = int(round(sh * scale))
        sky_resized = cv2.resize(sky_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # center-crop to target size
        x0 = max(0, (new_w - w) // 2)
        y0 = max(0, (new_h - h) // 2)
        sky_img = sky_resized[y0:y0 + h, x0:x0 + w]
        # In rare cases due to rounding, pad if needed
        if sky_img.shape[:2] != (h, w):
            sky_img = cv2.resize(sky_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ensure mask matches image size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Optionally soften mask edges
    if mask_soften:
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        soft = np.clip(dist / feather_strength, 0, 1)
        soft_mask = (mask.astype(np.float32) / 255.0) * soft
    else:
        soft_mask = mask.astype(np.float32) / 255.0
    
    # if mask_soften:
    #     mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Normalize mask for blending
    mask_f = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
    mask_3c = np.dstack([mask_f, mask_f, mask_f])
    

    if alpha_blend:
        result = img.astype(np.float32) * (1.0 - mask_3c) + sky_img.astype(np.float32) * mask_3c
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = img.copy()
        result[mask > 190] = sky_img[mask > 190]
        
    # if use_poisson:
    #     mask_bin = np.where(mask > 190, 255, 0).astype(np.uint8)
    #     center = (w // 2, h // 2)
    #     try:
    #         result = cv2.seamlessClone(result, sky_img, mask_bin, center, cv2.MIXED_CLONE)
    #         print("✨ Applied Poisson seamless blending.")
    #     except Exception as e:
    #         print(f"⚠️ Poisson blending failed: {e}")

    # Save result
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    ok = cv2.imwrite(output_image_path, result)
    if ok:
        print(f"🌤️ Sky replacement saved: {output_image_path}")
    else:
        print(f"❌ Failed to save output: {output_image_path}")
    return ok

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_all_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_rvm_model(device)

    # Try to load NCNN model
    param_path = os.path.join(BASE_DIR, "models", "skysegsmall_sim-opt-fp16.param")
    bin_path = os.path.join(BASE_DIR, "models", "skysegsmall_sim-opt-fp16.bin")
    
    ncnn_net = None
    if NCNN_AVAILABLE:
        print(f"🔹 Attempting to load NCNN model...")
        ncnn_net = load_ncnn_model(param_path, bin_path)
        if ncnn_net:
            print(f"✅ NCNN model loaded successfully!")
        else:
            print(f"⚠️ Using color-based detection fallback")
    else:
        print(f"⚠️ NCNN not available - using color-based detection")

    image_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    print(f"🖼 Found {len(image_files)} images")

    for fname in tqdm(image_files, desc="Processing images"):
        name, _ = os.path.splitext(fname)
        img_path = os.path.join(RAW_DIR, fname)

        # Paths
        human_mask = os.path.join(SKY_MASK_DIR, f"{name}_human.png")
        sky_initial = os.path.join(SKY_MASK_DIR, f"{name}_sky_initial.png")
        sky_refined = os.path.join(SKY_MASK_DIR, f"{name}_sky_refined.png")
        sky_final = os.path.join(SKY_MASK_DIR, f"{name}_final_combined.png")

        # 1️⃣ Human segmentation
        segment_human(img_path, human_mask, device, model)

        # 2️⃣ Sky detection (NCNN or color-based)
        segment_sky_ncnn(img_path, sky_initial, ncnn_net)

        # 3️⃣ Refinement
        refine_mask_with_exe(img_path, sky_initial, sky_refined)

        # 3.5️⃣ Binarize refined mask (clean binary output) - DISABLED
        # Check if refinement created a _result.png file
        actual_refined_path = sky_refined + "_result.png"
        # if os.path.exists(actual_refined_path):
        #     binarize_mask(actual_refined_path, 180)
        # else:
        #     binarize_mask(sky_refined, 180)

        # 3.6️⃣ Bitwise threshold processing
        # Check if refinement created a _result.png file
        actual_refined_path = sky_refined + "_result.png"
        if os.path.exists(actual_refined_path):
            # Process the actual refined mask with bitwise operations
            bitwise_processed_path = os.path.join(SKY_MASK_DIR, f"{name}_sky_bitwise_processed.png")
            process_refined_mask_with_bitwise(actual_refined_path, bitwise_processed_path, threshold=190)
            # Use the bitwise processed mask for final combination
            final_sky_mask = bitwise_processed_path
        else:
            # Process the refined mask with bitwise operations
            bitwise_processed_path = os.path.join(SKY_MASK_DIR, f"{name}_sky_bitwise_processed.png")
            process_refined_mask_with_bitwise(sky_refined, bitwise_processed_path, threshold=190)
            # Use the bitwise processed mask for final combination
            final_sky_mask = bitwise_processed_path

        # 4️⃣ Combine masks
        apply_human_mask_to_sky(final_sky_mask, human_mask, sky_final)

        # 5️⃣ Replace sky using final combined mask
        output_dir = os.path.join(BASE_DIR, "outputs", "final_test")
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, f"{name}_replaced.jpg")
        replace_sky_using_final_mask(
            input_image_path=img_path,
            final_mask_path=sky_final,
            output_image_path=output_image_path,
            sky_image_path=None,  # defaults to data/skies/sky.png
            alpha_blend=True,
            mask_soften=True,
            use_poisson=True,
            feather_strength=50
        )

    print("\n✅ All processing completed.")
    print(f"Masks saved to: {SKY_MASK_DIR}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    process_all_images()

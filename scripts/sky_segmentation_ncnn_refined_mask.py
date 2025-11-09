"""
Sky Segmentation using NCNN Model
Based on: https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

Uses the lightweight NCNN framework (only 2MB model!)

NOTE: If ncnn is not available, falls back to color-based detection
"""

import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

# Flag to check if ncnn is available
NCNN_AVAILABLE = False

try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    print("Warning: ncnn not available, will use color-based detection")
    NCNN_AVAILABLE = False


def detect_sky_color_based(image_path, output_path, visualize=False):
    """
    Fallback color-based sky detection
    (Used when NCNN is not available)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Sky color detection
    lower_blue = np.array([95, 30, 80])
    upper_blue = np.array([125, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_cyan = np.array([85, 15, 140])
    upper_cyan = np.array([100, 80, 255])
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    color_mask = cv2.bitwise_or(blue_mask, white_mask)
    color_mask = cv2.bitwise_or(color_mask, cyan_mask)
    
    # Position weighting
    position_weight = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        normalized_pos = i / height
        if normalized_pos < 0.3:
            weight = 1.0
        elif normalized_pos < 0.6:
            weight = 1.0 - ((normalized_pos - 0.3) / 0.3) ** 1.5
        else:
            weight = max(0, 0.2 - (normalized_pos - 0.6))
        position_weight[i, :] = weight
    
    # Combine
    color_mask_f = color_mask.astype(np.float32) / 255.0
    combined = (color_mask_f * 0.6 + position_weight * 0.4)
    _, sky_mask = cv2.threshold((combined * 255).astype(np.uint8), 140, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save mask
    cv2.imwrite(output_path, sky_mask)
    
    # Calculate percentage
    sky_percentage = (np.sum(sky_mask > 0) / (height * width)) * 100
    
    # Visualization
    if visualize:
        vis_path = output_path.replace('.png', '_overlay.jpg')
        overlay = img.copy()
        mask_colored = np.zeros_like(img)
        mask_colored[sky_mask > 0] = [0, 255, 255]
        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
        cv2.putText(overlay, f"Sky: {sky_percentage:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(vis_path, overlay)
    
    return sky_percentage


def refine_mask_with_exe(original_image_path, initial_mask_path, refined_mask_path):
    """
    Use mask_refine.exe to refine the mask
    Parameters: original_image_path, initial_mask_path, refined_mask_path
    """
    # Define base directory and run directory
    BASE_DIR = r"F:\Studio\sky_replacement_project"
    RUN_DIR = os.path.join(BASE_DIR, "run")
    
    exe_path = os.path.join(RUN_DIR, "mask_refine.exe")
    
    if not os.path.exists(exe_path):
        print(f"Warning: {exe_path} not found, skipping refinement")
        return False
    
    try:
        # Convert all paths to absolute paths to avoid path mistakes
        abs_original = os.path.abspath(original_image_path)
        abs_initial = os.path.abspath(initial_mask_path)
        abs_refined = os.path.abspath(refined_mask_path)
        
        # Ensure output folder exists
        os.makedirs(os.path.dirname(abs_refined), exist_ok=True)
        
        print(f"Running: {exe_path} {abs_original} {abs_initial} {abs_refined}")
        print(f"Working directory: {RUN_DIR}")
        
        # Run the mask_refine.exe with the three parameters using absolute paths
        # Set cwd=RUN_DIR so DLL is found automatically
        result = subprocess.run([exe_path, abs_original, abs_initial, abs_refined], 
                              capture_output=True, text=True, check=True, cwd=RUN_DIR)
        print("✅ Refinement successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Refinement failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running mask_refine.exe: {e}")
        return False


def create_refined_mask_folder_structure(base_name, refined_dir):
    """
    Create the refined mask folder structure with initial, refined, and visual files
    """
    # Create subfolder for this image
    image_folder = os.path.join(refined_dir, base_name)
    os.makedirs(image_folder, exist_ok=True)
    
    # Define paths for the three files
    initial_path = os.path.join(image_folder, f"{base_name}_initial.png")
    refined_path = os.path.join(image_folder, f"{base_name}_refined.png")
    visual_path = os.path.join(image_folder, f"{base_name}_visual.jpg")
    
    return initial_path, refined_path, visual_path


def create_visualization_comparison(original_image_path, initial_mask_path, refined_mask_path, visual_path):
    """
    Create a side-by-side comparison visualization
    """
    # Load images
    img = cv2.imread(original_image_path)
    initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)
    refined_mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or initial_mask is None or refined_mask is None:
        print(f"Warning: Could not load images for visualization")
        return
    
    h, w = img.shape[:2]
    
    # Create side-by-side comparison
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    # Original image
    comparison[:, :w] = img
    
    # Initial mask overlay
    initial_overlay = img.copy()
    initial_colored = np.zeros_like(img)
    initial_colored[initial_mask > 0] = [0, 255, 255]
    initial_overlay = cv2.addWeighted(img, 0.7, initial_colored, 0.3, 0)
    comparison[:, w:w*2] = initial_overlay
    
    # Refined mask overlay
    refined_overlay = img.copy()
    refined_colored = np.zeros_like(img)
    refined_colored[refined_mask > 0] = [0, 255, 255]
    refined_overlay = cv2.addWeighted(img, 0.7, refined_colored, 0.3, 0)
    comparison[:, w*2:] = refined_overlay
    
    # Calculate percentages
    initial_pct = (np.sum(initial_mask > 0) / (h * w)) * 100
    refined_pct = (np.sum(refined_mask > 0) / (h * w)) * 100
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, f"Initial ({initial_pct:.1f}%)", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, f"Refined ({refined_pct:.1f}%)", (w*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(visual_path, comparison)


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


def segment_sky_ncnn(image_path, output_path, net, visualize=False):
    """
    Segment sky using NCNN model or fallback to color-based
    """
    # If NCNN not available or net is None, use color-based
    if not NCNN_AVAILABLE or net is None:
        return detect_sky_color_based(image_path, output_path, visualize)
    
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
        
        # Calculate percentage
        sky_percentage = (np.sum(mask_resized > 0) / (original_height * original_width)) * 100
        
        # Visualization
        if visualize:
            vis_path = output_path.replace('.png', '_overlay.jpg')
            overlay = img.copy()
            mask_colored = np.zeros_like(img)
            mask_colored[mask_resized > 0] = [0, 255, 255]
            overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            cv2.putText(overlay, f"Sky: {sky_percentage:.1f}% (NCNN)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(vis_path, overlay)
        
        return sky_percentage
        
    except Exception as e:
        print(f"NCNN inference failed: {e}")
        print("Falling back to color-based detection...")
        return detect_sky_color_based(image_path, output_path, visualize)


def process_all_images(input_dir="data/raw", output_dir="data/masks", 
                       min_sky_percentage=5.0, visualize=False, use_refinement=True):
    """Process all images using NCNN model or color-based fallback"""
    
    # Try to load NCNN model
    param_path = "models/skysegsmall_sim-opt-fp16.param"
    bin_path = "models/skysegsmall_sim-opt-fp16.bin"
    
    net = None
    if NCNN_AVAILABLE:
        print(f"Attempting to load NCNN model...")
        net = load_ncnn_model(param_path, bin_path)
        if net:
            print(f"✓ NCNN model loaded successfully!")
        else:
            print(f"✓ Using color-based detection")
    else:
        print(f"NCNN not available - using color-based detection")
    
    print()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create refined mask directory if using refinement
    refined_dir = None
    if use_refinement:
        refined_dir = "refined_mask"
        os.makedirs(refined_dir, exist_ok=True)
        print(f"Refined masks will be saved to: {refined_dir}")
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    print()
    
    processed = 0
    skipped = 0
    refined = 0
    
    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        try:
            # Segment sky
            sky_pct = segment_sky_ncnn(input_path, output_path, net, visualize)
            
            if sky_pct < min_sky_percentage:
                print(f"\nSkipping {filename}: Sky only {sky_pct:.1f}% (< {min_sky_percentage}%)")
                os.remove(output_path)
                skipped += 1
            else:
                print(f"✓ {filename}: {sky_pct:.1f}% sky")
                processed += 1
                
                # Apply refinement if requested
                if use_refinement and refined_dir:
                    try:
                        # Create folder structure for refined masks
                        initial_path, refined_path, visual_path = create_refined_mask_folder_structure(base_name, refined_dir)
                        
                        # Copy initial mask to refined folder
                        import shutil
                        shutil.copy2(output_path, initial_path)
                        
                        # Run mask refinement
                        if refine_mask_with_exe(input_path, initial_path, refined_path):
                            # Create visualization
                            create_visualization_comparison(input_path, initial_path, refined_path, visual_path)
                            refined += 1
                            print(f"  → Refined mask created")
                        else:
                            print(f"  → Refinement failed")
                            
                    except Exception as e:
                        print(f"  → Error during refinement: {e}")
                
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped += 1
    
    print(f"\nProcessing complete!")
    print(f"  ✓ Successfully processed: {processed}")
    print(f"  ✗ Skipped/Failed: {skipped}")
    if use_refinement:
        print(f"  🔧 Refined masks: {refined}")
    print(f"\nMasks saved to: {output_dir}")
    if use_refinement and refined_dir:
        print(f"Refined masks saved to: {refined_dir}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sky masks using NCNN deep learning model")
    parser.add_argument("--input", default="data/raw", help="Input directory with images")
    parser.add_argument("--output", default="data/masks", help="Output directory for masks")
    parser.add_argument("--min-sky", type=float, default=5.0, 
                       help="Minimum sky percentage to keep (default: 5.0)")
    parser.add_argument("--visualize", action="store_true", 
                       help="Create visualization overlays")
    parser.add_argument("--no-refinement", action="store_true",
                       help="Skip mask refinement step")
    
    args = parser.parse_args()
    
    process_all_images(
        input_dir=args.input,
        output_dir=args.output,
        min_sky_percentage=args.min_sky,
        visualize=args.visualize,
        use_refinement=not args.no_refinement
    )


if __name__ == "__main__":
    main()


import os
import hashlib
import glob
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
CLEAN_IMG_DIR = os.path.join(DATA_PROC, "images_clean")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CLEAN_META_CSV = os.path.join(DATA_PROC, "image_metadata_clean.csv")
VALIDATION_CSV = os.path.join(REPORTS_DIR, "pixel_image_validation.csv")

def compute_phash(img_gray):
    """Compute 64-bit Perceptual Hash via 2D DCT"""
    img_resized = img_gray.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(img_resized, dtype=float)
    
    N = 32
    cols, rows = np.meshgrid(np.arange(N), np.arange(N))
    dct_matrix = np.cos((2 * cols + 1) * rows * np.pi / (2 * N))
    dct_matrix[0, :] /= np.sqrt(2)
    dct_matrix *= np.sqrt(2 / N)
    
    dct_2d = dct_matrix @ pixels @ dct_matrix.T
    top_left = dct_2d[:8, :8]
    med = np.median(top_left[1:, 1:])
    bits = (top_left > med).flatten()
    hex_str = ''.join(['1' if b else '0' for b in bits])
    return f"{int(hex_str, 2):016x}"

def hamming_distance(h1, h2):
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except Exception:
        return 64

def validate_all_pixel_images():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("==================================================")
    print("STARTING PHYSICAL PIXEL IMAGE VALIDATION ENGINE")
    print("==================================================")

    # 1. Discover all clean images in data/processed/images_clean/
    all_clean_files = []
    for root, dirs, files in os.walk(CLEAN_IMG_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                all_clean_files.append(os.path.join(root, f))

    print(f"Discovered {len(all_clean_files)} physical image files under data/processed/images_clean/")

    # Map image paths from clean metadata if available
    path_to_id = {}
    if os.path.exists(CLEAN_META_CSV):
        df_meta = pd.read_csv(CLEAN_META_CSV, dtype=str)
        for idx, r in df_meta.iterrows():
            img_id = r.get("Image_ID", "")
            img_path = r.get("Image_Path", "").replace("\\", "/")
            if img_id and img_path:
                path_to_id[img_path] = img_id

    records = []
    sha256_set = set()
    sha256_counts = {}
    phash_dict = {}

    # First pass: calculate hashes and basic metadata
    temp_data = []
    for fpath in sorted(all_clean_files):
        rel_p = os.path.relpath(fpath, BASE_DIR).replace("\\", "/")
        fname = os.path.basename(fpath)
        fsize = os.path.getsize(fpath)
        
        # Derive Image_ID if not in metadata mapping
        img_id = path_to_id.get(rel_p, os.path.splitext(fname)[0])
        
        is_readable = False
        has_pixel_data = False
        is_blank = True
        is_corrupted = False
        
        fmt = "UNKNOWN"
        w, h = 0, 0
        mode = "UNKNOWN"
        sha256_h = ""
        phash_h = ""
        p_min, p_max, p_mean, p_std = 0.0, 0.0, 0.0, 0.0

        if fsize == 0:
            is_corrupted = True
        else:
            try:
                with open(fpath, "rb") as fb:
                    sha256_h = hashlib.sha256(fb.read()).hexdigest()
                    sha256_counts[sha256_h] = sha256_counts.get(sha256_h, 0) + 1

                with Image.open(fpath) as img:
                    img.verify() # Verify file headers
                    
                # Re-open for pixel data analysis after verify()
                with Image.open(fpath) as img:
                    w, h = img.width, img.height
                    fmt = img.format if img.format else "PNG"
                    mode = img.mode
                    
                    if w > 0 and h > 0:
                        is_readable = True
                        
                    arr = np.array(img, dtype=float)
                    if arr.size > 0:
                        has_pixel_data = True
                        p_min = float(arr.min())
                        p_max = float(arr.max())
                        p_mean = float(round(arr.mean(), 2))
                        p_std = float(round(arr.std(), 2))
                        
                        if p_std > 0.0 and (p_max > p_min):
                            is_blank = False
                        else:
                            is_blank = True

                    img_gray = img.convert("L")
                    phash_h = compute_phash(img_gray)
                    phash_dict[rel_p] = phash_h

            except Exception as e:
                is_corrupted = True
                is_readable = False
                has_pixel_data = False

        temp_data.append({
            "Image_ID": img_id,
            "Image_Path": rel_p,
            "Format": fmt,
            "Width": w,
            "Height": h,
            "Color_Mode": mode,
            "File_Size": fsize,
            "SHA256": sha256_h,
            "Perceptual_Hash": phash_h,
            "Pixel_Min": p_min,
            "Pixel_Max": p_max,
            "Pixel_Mean": p_mean,
            "Pixel_Std": p_std,
            "Is_Readable": is_readable,
            "Has_Pixel_Data": has_pixel_data,
            "Is_Blank": is_blank,
            "Is_Corrupted": is_corrupted
        })

    # Second pass: duplicate & near duplicate flag calculations
    all_paths = [t["Image_Path"] for t in temp_data]
    
    for item in temp_data:
        path = item["Image_Path"]
        sha_h = item["SHA256"]
        ph_h = item["Perceptual_Hash"]

        # Duplicate check: SHA256 appears more than once
        is_duplicate = sha256_counts.get(sha_h, 0) > 1

        # Near duplicate check: pHash hamming distance <= 4 to another image
        is_near_duplicate = False
        if ph_h:
            for other_path, other_ph in phash_dict.items():
                if other_path != path and other_ph:
                    dist = hamming_distance(ph_h, other_ph)
                    if dist <= 4:
                        is_near_duplicate = True
                        break

        # Final Status Decision:
        # Only Is_Readable = TRUE, Has_Pixel_Data = TRUE, Is_Blank = FALSE, Is_Corrupted = FALSE
        if item["Is_Readable"] and item["Has_Pixel_Data"] and (not item["Is_Blank"]) and (not item["Is_Corrupted"]):
            final_status = "TRAINING_READY"
        else:
            final_status = "REJECTED"

        records.append({
            "Image_ID": item["Image_ID"],
            "Image_Path": item["Image_Path"],
            "Format": item["Format"],
            "Width": item["Width"],
            "Height": item["Height"],
            "Color_Mode": item["Color_Mode"],
            "File_Size": item["File_Size"],
            "SHA256": item["SHA256"],
            "Perceptual_Hash": item["Perceptual_Hash"],
            "Pixel_Min": item["Pixel_Min"],
            "Pixel_Max": item["Pixel_Max"],
            "Pixel_Mean": item["Pixel_Mean"],
            "Pixel_Std": item["Pixel_Std"],
            "Is_Readable": item["Is_Readable"],
            "Has_Pixel_Data": item["Has_Pixel_Data"],
            "Is_Blank": item["Is_Blank"],
            "Is_Corrupted": item["Is_Corrupted"],
            "Is_Duplicate": is_duplicate,
            "Is_Near_Duplicate": is_near_duplicate,
            "Final_Status": final_status
        })

    df_val = pd.DataFrame(records, columns=[
        "Image_ID", "Image_Path", "Format", "Width", "Height", "Color_Mode",
        "File_Size", "SHA256", "Perceptual_Hash", "Pixel_Min", "Pixel_Max",
        "Pixel_Mean", "Pixel_Std", "Is_Readable", "Has_Pixel_Data", "Is_Blank",
        "Is_Corrupted", "Is_Duplicate", "Is_Near_Duplicate", "Final_Status"
    ])

    df_val.to_csv(VALIDATION_CSV, index=False)
    print(f"Saved pixel image validation report to {VALIDATION_CSV}")

    # Summary Statistics
    total_imgs = len(df_val)
    readable_cnt = df_val["Is_Readable"].sum()
    pixel_data_cnt = df_val["Has_Pixel_Data"].sum()
    blank_cnt = df_val["Is_Blank"].sum()
    corrupt_cnt = df_val["Is_Corrupted"].sum()
    ready_cnt = (df_val["Final_Status"] == "TRAINING_READY").sum()

    print("\n==========================================")
    print("PIXEL IMAGE VALIDATION RESULTS")
    print("==========================================")
    print(f"Total Physical Images Validated: {total_imgs}")
    print(f"Is_Readable = TRUE: {readable_cnt} / {total_imgs}")
    print(f"Has_Pixel_Data = TRUE: {pixel_data_cnt} / {total_imgs}")
    print(f"Is_Blank = FALSE: {total_imgs - blank_cnt} / {total_imgs} (Blank: {blank_cnt})")
    print(f"Is_Corrupted = FALSE: {total_imgs - corrupt_cnt} / {total_imgs} (Corrupted: {corrupt_cnt})")
    print(f"TRAINING_READY Status: {ready_cnt} / {total_imgs}")
    print("==========================================")

    if ready_cnt == total_imgs and total_imgs > 0:
        print("ALL FINAL IMAGES ARE PHYSICALLY VERIFIED AND TRAINING-READY!")
    else:
        print(f"WARNING: {total_imgs - ready_cnt} images failed training readiness validation!")

if __name__ == "__main__":
    validate_all_pixel_images()

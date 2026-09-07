import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PIXEL = os.path.join(BASE_DIR, "data", "pixel_images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

META_CSV_PATH = os.path.join(DATA_PIXEL, "pixel_image_metadata.csv")
FAILURES_CSV_PATH = os.path.join(REPORTS_DIR, "pixel_conversion_failures.csv")
VAL_REPORT_PATH = os.path.join(REPORTS_DIR, "pixel_image_validation_report.txt")

def convert_and_verify():
    os.makedirs(DATA_PIXEL, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("==================================================")
    print("STARTING IMAGE FORMAT CONVERSION & PIXEL VALIDATION STAGE")
    print("==================================================")

    # 1. DISCOVER ALL EXISTING RAW IMAGES
    print("\n[1/5] Discovering all raw image files recursively under data/raw/...")
    raw_image_files = []
    for root, dirs, files in os.walk(DATA_RAW):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
                raw_image_files.append(os.path.join(root, f))

    raw_image_files = sorted(raw_image_files)
    total_discovered = len(raw_image_files)
    print(f"Total raw images discovered: {total_discovered}")

    # Track inventory metrics
    formats_found = set()
    color_modes_found = set()
    dim_set = set()
    channel_counts = {}

    metadata_records = []
    failure_records = []

    success_converted_cnt = 0
    success_verified_cnt = 0
    failed_cnt = 0

    histo_png_cnt = 0
    ct_png_cnt = 0
    mri_png_cnt = 0

    # 2 & 3. READ, CONVERT, SAVE TO PNG
    print("\n[2/5] Reading raw images, standardizing color channels, and saving PNGs...")
    for idx, fpath in enumerate(raw_image_files, 1):
        rel_orig = os.path.relpath(fpath, BASE_DIR).replace("\\", "/")
        fname = os.path.basename(fpath)
        fname_png = os.path.splitext(fname)[0] + ".png"
        orig_size = os.path.getsize(fpath)

        # Determine Modality & Subfolder
        parts = rel_orig.split("/")
        modality = "other"
        subfolder = ""
        
        if "histopathology_images" in rel_orig:
            modality = "histopathology"
            if len(parts) >= 4:
                subfolder = parts[-2]
        elif "ct_images" in rel_orig:
            modality = "ct"
            if len(parts) >= 4:
                subfolder = parts[-2]
        elif "mri_images" in rel_orig:
            modality = "mri"
            if len(parts) >= 4:
                subfolder = parts[-2]

        out_dir = os.path.join(DATA_PIXEL, modality)
        if subfolder:
            out_dir = os.path.join(out_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        out_png_path = os.path.join(out_dir, fname_png)
        rel_out_png = os.path.relpath(out_png_path, BASE_DIR).replace("\\", "/")

        img_id = os.path.splitext(fname)[0]

        # Read & decode original image
        readable = False
        has_pixel_data = False
        orig_fmt = "UNKNOWN"
        w, h = 0, 0
        mode = "UNKNOWN"
        channels = 0
        arr_shape = "()"
        p_min, p_max, p_mean, p_std = 0.0, 0.0, 0.0, 0.0

        try:
            with Image.open(fpath) as img:
                orig_fmt = img.format if img.format else os.path.splitext(fname)[1].replace(".", "").upper()
                formats_found.add(orig_fmt)
                
                w, h = img.width, img.height
                mode = img.mode
                color_modes_found.add(mode)
                dim_set.add(f"{w}x{h}")
                
                readable = True
                
                # Channel handling
                if mode == "L" or mode == "1":
                    channels = 1
                    img_to_save = img.copy()
                elif mode == "RGB":
                    channels = 3
                    img_to_save = img.copy()
                elif mode == "RGBA":
                    channels = 3 # Convert RGBA to RGB
                    img_to_save = Image.new("RGB", img.size, (255, 255, 255))
                    img_to_save.paste(img, mask=img.split()[3]) # Composite alpha onto white
                elif mode == "P":
                    img_converted = img.convert("RGBA" if "transparency" in img.info else "RGB")
                    if img_converted.mode == "RGBA":
                        img_to_save = Image.new("RGB", img.size, (255, 255, 255))
                        img_to_save.paste(img_converted, mask=img_converted.split()[3])
                        channels = 3
                    else:
                        img_to_save = img_converted
                        channels = 3
                else:
                    img_to_save = img.convert("RGB")
                    channels = 3

                channel_counts[channels] = channel_counts.get(channels, 0) + 1

                # Calculate original pixel stats
                arr_orig = np.array(img, dtype=float)
                if arr_orig.size > 0:
                    has_pixel_data = True
                    arr_shape = str(arr_orig.shape)
                    p_min = float(arr_orig.min())
                    p_max = float(arr_orig.max())
                    p_mean = float(round(arr_orig.mean(), 2))
                    p_std = float(round(arr_orig.std(), 2))

                # Save standardized PNG
                img_to_save.save(out_png_path, format="PNG")
                success_converted_cnt += 1

        except Exception as e:
            failed_cnt += 1
            failure_records.append({
                "Image_Path": rel_orig,
                "Failure_Reason": "READ_OR_CONVERT_ERROR",
                "Error_Message": str(e),
                "Recommended_Action": "INSPECT_RAW_FILE_IN_STAGE_03"
            })
            metadata_records.append({
                "Image_ID": img_id,
                "Original_Path": rel_orig,
                "Pixel_Image_Path": rel_out_png,
                "Original_Format": orig_fmt,
                "Output_Format": "PNG",
                "Width": w, "Height": h, "Color_Mode": mode, "Channels": channels,
                "Pixel_Array_Shape": arr_shape,
                "Pixel_Min": p_min, "Pixel_Max": p_max, "Pixel_Mean": p_mean, "Pixel_Std": p_std,
                "Original_File_Size": orig_size, "Output_File_Size": 0,
                "Conversion_Status": "FAILED_TO_READ"
            })
            continue

        # 5. VERIFY GENERATED PNG FILE (original -> array -> PNG -> array)
        out_size = os.path.getsize(out_png_path)
        verified = False
        try:
            with Image.open(out_png_path) as img_png:
                img_png.load()
                arr_png = np.array(img_png)
                if img_png.width > 0 and img_png.height > 0 and arr_png.size > 0 and out_size > 0:
                    verified = True
                    success_verified_cnt += 1
                    
                    if modality == "histopathology":
                        histo_png_cnt += 1
                    elif modality == "ct":
                        ct_png_cnt += 1
                    elif modality == "mri":
                        mri_png_cnt += 1
        except Exception as e:
            failed_cnt += 1
            failure_records.append({
                "Image_Path": rel_out_png,
                "Failure_Reason": "VERIFICATION_ERROR",
                "Error_Message": str(e),
                "Recommended_Action": "RE_GENERATE_PNG"
            })

        status_str = "SUCCESS" if verified else "FAILED_TO_VERIFY"

        metadata_records.append({
            "Image_ID": img_id,
            "Original_Path": rel_orig,
            "Pixel_Image_Path": rel_out_png,
            "Original_Format": orig_fmt,
            "Output_Format": "PNG",
            "Width": w,
            "Height": h,
            "Color_Mode": mode,
            "Channels": channels,
            "Pixel_Array_Shape": arr_shape,
            "Pixel_Min": p_min,
            "Pixel_Max": p_max,
            "Pixel_Mean": p_mean,
            "Pixel_Std": p_std,
            "Original_File_Size": orig_size,
            "Output_File_Size": out_size,
            "Conversion_Status": status_str
        })

    # 6. SAVE PIXEL IMAGE METADATA CSV
    df_meta = pd.DataFrame(metadata_records)
    df_meta.to_csv(META_CSV_PATH, index=False)
    print(f"-> Created {META_CSV_PATH} ({len(df_meta)} records)")

    # 7. SAVE FAILURES CSV
    df_failures = pd.DataFrame(failure_records, columns=["Image_Path", "Failure_Reason", "Error_Message", "Recommended_Action"])
    df_failures.to_csv(FAILURES_CSV_PATH, index=False)
    print(f"-> Created {FAILURES_CSV_PATH} ({len(df_failures)} failures logged)")

    # 8. CREATE PIXEL VALIDATION REPORT
    report_lines = []
    report_lines.append("===========================================")
    report_lines.append("PIXEL IMAGE CONVERSION REPORT")
    report_lines.append("=============================")
    report_lines.append("")
    report_lines.append(f"Total images discovered: {total_discovered}")
    report_lines.append(f"Successfully converted: {success_converted_cnt}")
    report_lines.append(f"Failed: {failed_cnt}")
    report_lines.append(f"Successfully verified: {success_verified_cnt}")
    report_lines.append("")
    report_lines.append(f"Image formats found: {', '.join(sorted(list(formats_found)))}")
    report_lines.append("")
    report_lines.append("Output format:")
    report_lines.append("PNG raster/pixel images")
    report_lines.append("")
    report_lines.append(f"Color modes: {', '.join(sorted(list(color_modes_found)))}")
    report_lines.append("")
    report_lines.append(f"Image dimensions: {', '.join(sorted(list(dim_set)))}")
    report_lines.append("")
    report_lines.append("Channel distribution:")
    for ch, cnt in sorted(channel_counts.items()):
        report_lines.append(f"  {ch} channel(s): {cnt} images")
    report_lines.append("")
    report_lines.append("===========================================")

    with open(VAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"-> Created {VAL_REPORT_PATH}")

    # 12. FINAL TERMINAL OUTPUT PRINT
    all_real_pixels = "YES" if (success_verified_cnt == total_discovered and total_discovered > 0) else "NO"
    all_pil_readable = "YES" if (success_verified_cnt == total_discovered and total_discovered > 0) else "NO"
    conversion_status = "READY FOR NEXT CLEANING STAGE" if (failed_cnt == 0 and total_discovered > 0) else "NOT READY"

    print("\n")
    print("===========================================")
    print("PIXEL FORMAT CONVERSION COMPLETE")
    print("================================ shadow")
    print("")
    print(f"Images Found: {total_discovered}")
    print(f"Images Successfully Converted: {success_converted_cnt}")
    print(f"Images Successfully Pixel-Verified: {success_verified_cnt}")
    print(f"Images Failed: {failed_cnt}")
    print("")
    print(f"Histopathology PNG Images: {histo_png_cnt}")
    print(f"CT PNG Images: {ct_png_cnt}")
    print(f"MRI PNG Images: {mri_png_cnt}")
    print("")
    print(f"All final images contain real pixel data:")
    print(f"{all_real_pixels}")
    print("")
    print(f"All final PNG images are readable by PIL:")
    print(f"{all_pil_readable}")
    print("")
    print("Pixel conversion status:")
    print(f"{conversion_status}")
    print("===========================================")

if __name__ == "__main__":
    convert_and_verify()

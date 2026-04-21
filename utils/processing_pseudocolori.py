import cv2
import os
import glob
import shutil

def combine_color_bands(source_dir, dest_dir, data_type, red, green, blue):
    output_dir = os.path.join(dest_dir, data_type, 'all_images')
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(source_dir, f"*@Image_{red}.png")
    reference_files = glob.glob(search_pattern)

    saved = 0
    skipped = 0

    for ref_file in reference_files:
        base_name = os.path.basename(ref_file).replace(f"@Image_{red}.png", "")
        save_path = os.path.join(output_dir, f"{base_name}.png")
        
        if os.path.exists(save_path):
            print(f"{base_name} already processed")
            skipped+=1
            continue

        path_R = os.path.join(source_dir, f"{base_name}@Image_{red}.png")
        path_G = os.path.join(source_dir, f"{base_name}@Image_{green}.png")
        path_B = os.path.join(source_dir, f"{base_name}@Image_{blue}.png")
        
        if os.path.exists(path_R) and os.path.exists(path_G) and os.path.exists(path_B):
            img_R = cv2.imread(path_R, cv2.IMREAD_GRAYSCALE)
            img_G = cv2.imread(path_G, cv2.IMREAD_GRAYSCALE)
            img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
            
            merged_img = cv2.merge([img_B, img_G, img_R])
            
            save_path = os.path.join(output_dir, f"{base_name}.png")
            cv2.imwrite(save_path, merged_img)

            print(f"{base_name} processed and saved")
            saved+=1
    
    print(f"\nCreated {saved} images in {output_dir}")
    print(f"{skipped} images already processed (skipped)")
    print(f"Total images: {saved + skipped}")


def use_COLOR_image(source_dir, dest_dir):
    output_dir = os.path.join(dest_dir, 'RGB', 'all_images')
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(source_dir, f"*@COLOR_Image.png")
    files = glob.glob(search_pattern)

    saved = 0
    skipped = 0

    for f in files:
        base_name = os.path.basename(f).replace(f"@COLOR_Image.png", "")
        save_path = os.path.join(output_dir, f"{base_name}.png")
        
        if os.path.exists(save_path):
            print(f"{base_name} already saved")
            skipped+=1
            continue

        shutil.copy(f, save_path)
        saved+=1
        print(f"Saving {base_name}")

    print(f"\nSaved {saved} images in {output_dir}")
    print(f"{skipped} images already processed (skipped)")
    print(f"Total images: {saved + skipped}")


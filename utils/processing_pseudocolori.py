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

            rnorm = cv2.normalize(img_R, None, 0, 255, cv2.NORM_MINMAX)
            gnorm = cv2.normalize(img_G, None, 0, 255, cv2.NORM_MINMAX)
            bnorm = cv2.normalize(img_B, None, 0, 255, cv2.NORM_MINMAX)
            
            merged_img = cv2.merge([bnorm, gnorm, rnorm])
            
            save_path = os.path.join(output_dir, f"{base_name}.png")
            cv2.imwrite(save_path, merged_img)

            print(f"{base_name} processed and saved")
            saved+=1
    
    print(f"\nCreated {saved} images in {output_dir}")
    print(f"{skipped} images already processed, skipped")
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


def _copy_single_channel_images(source_dir, output_dir, data_type, channel):
    search_pattern = os.path.join(source_dir, f"*@Image_{channel}.png")
    files = glob.glob(search_pattern)

    saved = 0
    skipped = 0

    for f in files:
        base_name = os.path.basename(f).replace(f"@Image_{channel}.png", f"_{data_type}")
        save_path = os.path.join(output_dir, f"{base_name}.png")

        if os.path.exists(save_path):
            print(f"{base_name} already saved")
            skipped+=1
            continue

        shutil.copy(f, save_path)
        saved+=1
        print(f"Saving {channel} image {base_name}")

    return saved, skipped


def use_single_band(source_dir, dest_dir, data_type, channel):
    output_dir = os.path.join(dest_dir, data_type, 'all_images')
    os.makedirs(output_dir, exist_ok=True)

    saved, skipped = _copy_single_channel_images(source_dir, output_dir, data_type, channel)

    print(f"\nSaved {saved} images in {output_dir}")
    print(f"{skipped} images already processed, skipped")
    print(f"Total images: {saved + skipped}")


def merge_datasets(source_dir, dest_dir, data_type, channel, dataset_to_merge):
    output_dir = os.path.join(dest_dir, f'{dataset_to_merge}_{data_type}', 'all_images')

    print(f"\nCoping {dataset_to_merge} into {output_dir}\n")
    dataset_dir = os.path.join(dest_dir, dataset_to_merge, 'all_images')
    shutil.copytree(dataset_dir, output_dir)

    saved, skipped = _copy_single_channel_images(source_dir, output_dir, data_type, channel)

    print(f"\nSaved {saved} images in {output_dir}")
    print(f"{skipped} images already processed (skipped)")
    print(f"{saved + skipped} total images: {len(os.listdir(dataset_dir))} {dataset_to_merge} and {saved + skipped} {channel}")
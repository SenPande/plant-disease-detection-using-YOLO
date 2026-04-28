import cv2
import numpy as np
import os
import glob

def get_vegetation_index(image, index_type):
    img = image.astype(np.float32) / 255.0
    epsilon = 1e-6 

    b, g, r = cv2.split(img)

    indices = {
        'NGRDI': lambda: (g - r) / (g + r + epsilon),       # Normalized Green-Red Difference Index, stato della clorofilla
        'EXG': lambda: 2 * g - r - b,                       # Excess Green Index, distingue vegetazione da suolo e altri oggetti
    }

    if index_type not in indices:
        print(f"\nIndice non supportato")
        return None

    idx = indices[index_type]()

    p_min, p_max = np.percentile(idx, (1, 99))

    idx_clipped = np.clip(idx, p_min, p_max)

    idx_normalized = cv2.normalize(idx_clipped, None, 0, 255, cv2.NORM_MINMAX)

    return idx_normalized.astype(np.uint8)


def save_index_images(source_dir, dest_dir, index_type):
    output_dir = os.path.join(dest_dir, index_type, 'all_images')
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(source_dir, f"*@COLOR_Image.png")
    images = glob.glob(search_pattern)

    saved = 0
    skipped = 0

    for img in images:
        base_name = os.path.basename(img).replace(f"@COLOR_Image.png", "")
        save_path = os.path.join(output_dir, f"{base_name}.png")

        if os.path.exists(save_path):
            print(f"{base_name} already saved")
            skipped+=1
            continue
        
        image = cv2.imread(img)
        index = get_vegetation_index(image, index_type)

        if(cv2.imwrite(save_path, index)):
            print(f"Immagine indicizzata salvata in {save_path}")
            saved+=1

    print(f"\nSaved {saved} images in {output_dir}")
    print(f"{skipped} images already processed (skipped)")
    print(f"Total images: {saved + skipped}")

import cv2
import numpy as np
import os

def get_vegetation_index(image, index_type='VARI'):
    """
    Calcola l'indice di vegetazione dell'immagine.
    """
    img = image.astype(np.float32) / 255.0 # Converti l'immagine in virgola mobile e normalizza
    epsilon = 1e-6 # Evita la divisione per zero

    # Verifica se l'immagine è RGB o multispettrale andando a vedere il numero di canali 
    # image.shape resituisce la tupla (altezza, larghezza, canali) con indici rispettivamente (0, 1, 2)

    # Immagine RGB
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        print("RGB image")

    # TODO: Cambiare in base al tipo di immagine
    # Immagine multispettrale (R, G, B, NIR)
    elif img.shape[2] == 4:
        b, g, r, nir = cv2.split(img)
        print("Multispectral image")

    # TODO: Gestire errori indice legati al tipo di immagine data (es. restituire "INVALID" se provo a usare
    #       NDVI su un'immagine RGB)
    indices = {
        # Indici basati su canali RGB
        'NGRDI': lambda: (g - r) / (g + r + epsilon),       # Normalized Green-Red Difference Index, stato della clorofilla
        'VARI': lambda: (g - r) / (g + r - b + epsilon),    # Visible Atmospherically Resistant Index, stato della clorofilla, resistente agli errori atmosferici
        'ExG': lambda: 2 * g - r - b,                       # Excess Green Index, distingue vegetazione da suolo e altri oggetti

        # Indici basati su canali multispettrali (principalmente NIR)
        'NDVI': lambda: (nir - r) / (nir + r + epsilon)   # Normalized Difference Vegetation Index, stato della clorofilla
    }

    if index_type not in indices:
        print(f"Indice non supportato")
        return None

    idx = indices[index_type]()

    return cv2.normalize(idx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def save_processed_image(image_path, vegetation_index, index_type):
    # TODO: folder = os.path.join('data', 'processed', index_type)
    folder = f'data/test/KAGGLE_processed/{index_type}'
    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, os.path.basename(image_path).split('.')[0] + f'_{index_type}.jpg')

    if(cv2.imwrite(save_path, vegetation_index)):
        print(f"Immagine indicizzata salvata in {save_path}")

import cv2
from utils.image_processing import get_vegetation_index, save_processed_image
import kagglehub
import os

def main():
    # TODO: chiedere indice e cartella come argomenti da terminale
    index = 'ExG'

    process_dataset(index)

def process_dataset(index):
    # Dataset temporaneo trovato su kaggle per testare funzione di processing
    # TODO: dataset_path = 'data/raw'
    dataset_path = kagglehub.dataset_download("ashishjstar/lettuce-diseases")
    print(f"Dataset scaricato in: {dataset_path}")

    for root, _, files in os.walk(dataset_path):
        print(f"\nProcessando la cartella: {os.path.basename(root)} ({len(files)} immagini)")

        for file in files:
            image_path = os.path.join(root, file)
            img = cv2.imread(image_path)

            print(f"\nProcessing: {os.path.basename(image_path)} {img.shape}")

            vegetation_index = get_vegetation_index(img, index)

            save_processed_image(image_path, vegetation_index, index)

    print("\nDataset processato")

if __name__ == "__main__":
    main()


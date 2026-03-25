import cv2
from utils.image_processing import get_vegetation_index, save_processed_image
import os

def main():

    test = 'data/test/test.jpg'
    indice = 'NGRDI'

    img = cv2.imread(test)

    v_indice = get_vegetation_index(img, indice)

    print(f"\nIndice {v_indice}")

    save_processed_image(test, img, v_indice, indice)

if __name__ == "__main__":
    main()

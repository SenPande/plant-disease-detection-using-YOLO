import cv2
import os
from utils.process_dataset import split_dataset


def main():
    split_dataset('rgb')

if __name__ == "__main__":
    main()
import os
from utils.process_dataset import split_dataset
import sys
from utils.processing_pseudocolori import combine_color_bands, use_COLOR_image
from utils.image_processing import save_index_images

def main():
    color_bands = {
        "blue": "460",
        "green": "540",
        "red": "630",
        "NIR": "850",
        "NIR_WATER": "980"
    }

    source_dir = os.path.join('data', 'raw', 'Dataset pomodori', 'images')
    dest_dir = os.path.join('data', 'processed', 'Dataset pomodori')

    if len(sys.argv) < 2:
        print("MISSING ARGS: Define dataset type")
        return None

    data_type = sys.argv[1].upper()

    match data_type:
        case 'RGB':
            print("Processing RGB dataset\n")

            use_COLOR_image(source_dir, dest_dir)

        case 'CIR':
            print(f"Processing CIR dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'CIR', 
                                color_bands["NIR"], 
                                color_bands["red"], 
                                color_bands["green"])

        case 'ERGB':
            print(f"Processing ERGB dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'ERGB', 
                                color_bands["red"], 
                                color_bands["green"], 
                                color_bands["blue"])        
            
        case 'WATER':
            print(f"Processing WATER dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'WATER', 
                                color_bands["NIR_WATER"], 
                                color_bands["red"], 
                                color_bands["green"])   

        case 'GBR':
            print(f"Processing GBR dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'GBR', 
                                color_bands["green"], 
                                color_bands["blue"], 
                                color_bands["red"])  
            
        case 'RBG':
            print(f"Processing RBG dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'RBG', 
                                color_bands["red"], 
                                color_bands["blue"], 
                                color_bands["green"])
        
        case _:
            print(f"Processing {data_type} vegetation index\n")

            save_index_images(source_dir,
                              dest_dir,
                              data_type)
            
    split_dataset(data_type)

if __name__ == "__main__":
    main()
import os
from utils.process_dataset import split_dataset
import sys
from utils.processing_pseudocolori import combine_color_bands, use_COLOR_image, merge_datasets
from utils.image_processing import save_index_images

def main():
    color_bands = {
        "B": "460",
        "G": "540",
        "R": "630",
        "NIR": "850",
        "NIR_F": "980"
    }

    source_dir = os.path.join('data', 'raw', 'Dataset pomodori', 'images')
    dest_dir = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')

    if len(sys.argv) < 2:
        print("MISSING ARGS: Define dataset type")
        return None

    data_type = sys.argv[1].upper()

    match data_type:
        case 'RGB':
            print("Processing RGB dataset\n")

            use_COLOR_image(source_dir, dest_dir)

        case 'ERGB':
            print("Processing ERGB dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'ERGB', 
                                color_bands["R"], 
                                color_bands["G"], 
                                color_bands["B"])        

        case 'GBR':
            print("Processing GBR dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'GBR', 
                                color_bands["G"], 
                                color_bands["B"], 
                                color_bands["R"])  
            
        case 'RBG':
            print("Processing RBG dataset\n")

            combine_color_bands(source_dir, 
                                dest_dir, 
                                'RBG', 
                                color_bands["R"], 
                                color_bands["B"], 
                                color_bands["G"])
            
        case 'G' | 'R' | 'B':
            dataset_to_merge = sys.argv[2].upper()

            print(f'Merging {data_type} single channel and {dataset_to_merge} datasets')

            merge_datasets(source_dir,
                           dest_dir,
                           data_type,
                           color_bands[data_type],
                           dataset_to_merge)
            
            data_type = f'{dataset_to_merge}_{data_type}'
            
        case _:
            print(f"Processing {data_type} vegetation index\n")

            save_index_images(source_dir,
                              dest_dir,
                              data_type)
            
    split_dataset(data_type)

if __name__ == "__main__":
    main()
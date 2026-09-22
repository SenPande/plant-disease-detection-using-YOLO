import argparse
import os
import sys
from utils.process_dataset import split_dataset
from utils.processing_pseudocolori import combine_color_bands, use_COLOR_image, use_single_band, merge_datasets
from utils.image_processing import save_index_images, SUPPORTED_INDICES

DEFAULT_SOURCE_DIR = os.path.join('data', 'raw', 'Dataset pomodori', 'images')
DEFAULT_DEST_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')
DEFAULT_LABELS_DIR = os.path.join('data', 'raw', 'Dataset pomodori', 'labels')

COLOR_BANDS = {
    "B": "460",
    "G": "540",
    "R": "630",
    "NIR": "850",
    "NIR_F": "980",
}

COMBINED_BAND_ORDERS = {
    'ERGB': (COLOR_BANDS["R"], COLOR_BANDS["G"], COLOR_BANDS["B"]),
    'GBR': (COLOR_BANDS["G"], COLOR_BANDS["B"], COLOR_BANDS["R"]),
    'RBG': (COLOR_BANDS["R"], COLOR_BANDS["B"], COLOR_BANDS["G"]),
    'GRB': (COLOR_BANDS["G"], COLOR_BANDS["R"], COLOR_BANDS["B"]),
    'BRG': (COLOR_BANDS["B"], COLOR_BANDS["R"], COLOR_BANDS["G"]),
    'BGR': (COLOR_BANDS["B"], COLOR_BANDS["G"], COLOR_BANDS["R"]),
}

SINGLE_CHANNEL_TYPES = ("R", "G", "B")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_type")
    parser.add_argument("--merge-with", metavar="DATASET")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dest-dir", default=DEFAULT_DEST_DIR)
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    return parser.parse_args()


def build_dataset(args, data_type):
    if data_type == 'RGB':
        print("Processing RGB dataset\n")
        use_COLOR_image(args.source_dir, args.dest_dir)
        return data_type

    if data_type in COMBINED_BAND_ORDERS:
        print(f"Processing {data_type} dataset\n")
        red, green, blue = COMBINED_BAND_ORDERS[data_type]
        combine_color_bands(args.source_dir, args.dest_dir, data_type, red, green, blue)
        return data_type

    if data_type in SINGLE_CHANNEL_TYPES:
        if args.merge_with:
            dataset_to_merge = args.merge_with.upper()
            print(f"Merging {data_type} single channel and {dataset_to_merge} datasets")
            merge_datasets(args.source_dir, args.dest_dir, data_type, COLOR_BANDS[data_type], dataset_to_merge)
            return f'{dataset_to_merge}_{data_type}'

        print(f"Processing {data_type} single band dataset\n")
        use_single_band(args.source_dir, args.dest_dir, data_type, COLOR_BANDS[data_type])
        return data_type

    if data_type not in SUPPORTED_INDICES:
        print(f"MISSING ARGS: '{data_type}' non supportato")
        print(f"Tipi validi: RGB, {', '.join(COMBINED_BAND_ORDERS)}, {', '.join(SINGLE_CHANNEL_TYPES)}, {', '.join(SUPPORTED_INDICES)}")
        sys.exit(1)

    print(f"Processing {data_type} vegetation index\n")
    save_index_images(args.source_dir, args.dest_dir, data_type)
    return data_type


def main():
    args = parse_args()
    data_type = build_dataset(args, args.data_type.upper())
    split_dataset(data_type, dest_dir=args.dest_dir, labels_dir=args.labels_dir)


if __name__ == "__main__":
    main()

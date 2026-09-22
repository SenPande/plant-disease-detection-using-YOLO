import argparse
import os
import shutil
import tempfile

from ultralytics.data.converter import convert_coco

DEFAULT_RAW_DIR = 'data/raw'
DEFAULT_SOURCE_NAME = 'DatasetPomodori'
DEFAULT_DEST_NAME = 'Dataset pomodori'
DEFAULT_ANNOTATIONS = 'soup-dataset.json'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--dest-name", default=DEFAULT_DEST_NAME)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    return parser.parse_args()


def copy_images(source_images_dir, dest_images_dir):
    if os.path.exists(dest_images_dir) and os.listdir(dest_images_dir):
        print(f"{dest_images_dir} esiste già, saltata")
        return 0

    os.makedirs(dest_images_dir, exist_ok=True)

    copied = 0
    for filename in os.listdir(source_images_dir):
        if filename.startswith('._'):
            continue
        shutil.copy(os.path.join(source_images_dir, filename), os.path.join(dest_images_dir, filename))
        copied += 1

    return copied


def convert_annotations(json_path, dest_labels_dir):
    with tempfile.TemporaryDirectory() as tmp_json_dir, tempfile.TemporaryDirectory() as tmp_out_root:
        shutil.copy(json_path, os.path.join(tmp_json_dir, os.path.basename(json_path)))

        tmp_out_dir = os.path.join(tmp_out_root, 'converted')
        convert_coco(labels_dir=tmp_json_dir, save_dir=tmp_out_dir)

        json_stem = os.path.splitext(os.path.basename(json_path))[0].replace('instances_', '')
        converted_labels_dir = os.path.join(tmp_out_dir, 'labels', json_stem)

        os.makedirs(dest_labels_dir, exist_ok=True)
        converted = 0
        for filename in os.listdir(converted_labels_dir):
            if not filename.endswith('.txt'):
                continue
            new_name = filename.replace('@COLOR_Image', '')
            shutil.copy(os.path.join(converted_labels_dir, filename), os.path.join(dest_labels_dir, new_name))
            converted += 1

        return converted


def main():
    args = parse_args()

    source_dir = os.path.join(args.raw_dir, args.source_name)
    dest_dir = os.path.join(args.raw_dir, args.dest_name)
    json_path = os.path.join(source_dir, args.annotations)

    if not os.path.isdir(source_dir):
        raise SystemExit(f"Cartella sorgente non trovata: {source_dir}")
    if not os.path.isfile(json_path):
        raise SystemExit(f"File di annotazioni non trovato: {json_path}")

    images_copied = copy_images(os.path.join(source_dir, 'images'), os.path.join(dest_dir, 'images'))
    print(f"Copiate {images_copied} immagini in {os.path.join(dest_dir, 'images')}")

    labels_dest = os.path.join(dest_dir, 'labels')
    if os.path.exists(labels_dest) and os.listdir(labels_dest):
        print(f"{labels_dest} esiste già, saltato")
    else:
        labels_converted = convert_annotations(json_path, labels_dest)
        print(f"Convertite {labels_converted} label (da {args.annotations}) in {labels_dest}")

    print(f"\nDataset in: {os.path.abspath(dest_dir)}")


if __name__ == "__main__":
    main()

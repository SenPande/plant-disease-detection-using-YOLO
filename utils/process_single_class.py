import argparse
import os
import shutil

DEFAULT_MULTI_CLASS_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')
DEFAULT_SINGLE_CLASS_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Single class')

SPLITS = ('train', 'valid', 'test')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_type")
    parser.add_argument("--multi-class-dir", default=DEFAULT_MULTI_CLASS_DIR)
    parser.add_argument("--single-class-dir", default=DEFAULT_SINGLE_CLASS_DIR)
    return parser.parse_args()

def convert_labels_to_single_class(src_labels_dir, dest_labels_dir):
    for label_file in os.listdir(src_labels_dir):
        if not label_file.endswith('.txt'):
            continue

        with open(os.path.join(src_labels_dir, label_file), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(dest_labels_dir, label_file), 'w') as f:
            for line in lines:
                parts = line.split()
                if parts:
                    parts[0] = '0'
                    f.write(" ".join(parts) + "\n")

def main():
    args = parse_args()
    data_type = args.data_type.upper()

    old_path = os.path.join(args.multi_class_dir, data_type)
    new_path = os.path.join(args.single_class_dir, data_type)

    for split in SPLITS:
        os.makedirs(os.path.join(new_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, split, 'labels'), exist_ok=True)

        img_dir = os.path.join(old_path, split, 'images')
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), os.path.join(new_path, split, 'images', img))

        convert_labels_to_single_class(
            os.path.join(old_path, split, 'labels'),
            os.path.join(new_path, split, 'labels'),
        )

    print(f"\nSingle class dataset created in: {os.path.abspath(new_path)}")

    yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['Disease']
"""

    yaml_path = os.path.join(new_path, f"{data_type}_data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content.strip())

    print(f"\nConfiguration file created: {yaml_path}")


if __name__ == "__main__":
    main()

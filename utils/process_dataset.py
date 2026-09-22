import os
import random
import shutil

DEFAULT_DEST_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')
DEFAULT_LABELS_DIR = os.path.join('data', 'raw', 'Dataset pomodori', 'labels')

TRAIN_SPLIT = 0.7  
VALID_SPLIT = 0.9  

RANDOM_SEED = 67
CLASSES = ['tuta', 'oidium']


def split_dataset(img_type, dest_dir=DEFAULT_DEST_DIR, labels_dir=DEFAULT_LABELS_DIR):
    dataset_dir = os.path.join(dest_dir, img_type)
    imgs_path = os.path.join(dataset_dir, "all_images")

    check_dir = os.path.join(dataset_dir, "train", "images")
    if os.path.exists(check_dir) and len(os.listdir(check_dir)) > 0:
        print(f"\n{img_type} dataset già splittato, saltato per evitare data leakage")
    else:
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

        all_images = [f.replace('.png', '') for f in os.listdir(imgs_path) if f.endswith('.png')]
        all_images.sort()

        random.seed(RANDOM_SEED)
        random.shuffle(all_images)

        train_idx = int(len(all_images) * TRAIN_SPLIT)
        valid_idx = int(len(all_images) * VALID_SPLIT)
        train_files = all_images[:train_idx]
        valid_files = all_images[train_idx:valid_idx]
        test_files = all_images[valid_idx:]

        def move_files(files, split):
            for f in files:
                shutil.copy(os.path.join(imgs_path, f"{f}.png"), os.path.join(dataset_dir, split, "images", f"{f}.png"))

                # i canali singoli fusi condividono la label dell'immagine originale
                if f.endswith(("_R", "_G", "_B")):
                    original_f = f[:-2]
                    label_src = os.path.join(labels_dir, f"{original_f}.txt")
                else:
                    label_src = os.path.join(labels_dir, f"{f}.txt")

                if os.path.exists(label_src):
                    shutil.copy(label_src, os.path.join(dataset_dir, split, "labels", f"{f}.txt"))

        move_files(train_files, 'train')
        move_files(valid_files, 'valid')
        move_files(test_files, 'test')

        print(f"\nTrain: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""

    yaml_path = os.path.join(dataset_dir, f"{img_type}_data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content.strip())

    print(f"\nConfiguration file created: {yaml_path}")

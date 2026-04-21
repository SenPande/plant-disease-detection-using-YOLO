import os 
import random
import shutil



def split_dataset(img_type):
    imgs_path = f"data/processed/{img_type}/all_images"
    labels_path = "data/raw/Dataset pomodori/labels"
    dest_path = f"data/processed/{img_type}"

    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{dest_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{dest_path}/{split}/labels", exist_ok=True)

    all_images = [f.replace('.png', '') for f in os.listdir(imgs_path) if f.endswith('.png')]
    random.shuffle(all_images)

    train_idx = int(len(all_images) * 0.7)
    valid_idx = int(len(all_images) * 0.9)
    train_files = all_images[:train_idx]
    valid_files = all_images[train_idx:valid_idx]
    test_files = all_images[valid_idx:]

    def move_files(files, split):
        for f in files:
            shutil.copy(f"{imgs_path}/{f}.png", f"{dest_path}/{split}/images/{f}.png")

            label_src = f"{labels_path}/{f}.txt"
            if os.path.exists(label_src):
                shutil.copy(label_src, f"{dest_path}/{split}/labels/{f}.txt")

    move_files(train_files, 'train')
    move_files(valid_files, 'valid')
    move_files(test_files, 'test')
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
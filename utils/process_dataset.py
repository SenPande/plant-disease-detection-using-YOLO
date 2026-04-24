import os 
import random
import shutil

def split_dataset(img_type):
    imgs_path = f"data/processed/Dataset pomodori/{img_type}/all_images"
    labels_path = "data/raw/Dataset pomodori/labels"
    dest_path = f"data/processed/Dataset pomodori/{img_type}"

    check_dir = f"{dest_path}/train/images"
    if os.path.exists(check_dir) and len(os.listdir(check_dir)) > 0:
        print(f"\n{img_type} dataset already split. Skipping to avoid data leakage.")
    else:
        for split in ['train', 'valid', 'test']:
            os.makedirs(f"{dest_path}/{split}/images", exist_ok=True)
            os.makedirs(f"{dest_path}/{split}/labels", exist_ok=True)

        all_images = [f.replace('.png', '') for f in os.listdir(imgs_path) if f.endswith('.png')]
        all_images.sort()

        random.seed(67)
        random.shuffle(all_images)

        train_idx = int(len(all_images) * 0.7)
        valid_idx = int(len(all_images) * 0.9)
        train_files = all_images[:train_idx]
        valid_files = all_images[train_idx:valid_idx]
        test_files = all_images[valid_idx:]

        def move_files(files, split):
            for f in files:
                shutil.copy(f"{imgs_path}/{f}.png", f"{dest_path}/{split}/images/{f}.png")

                if f.endswith(("_R", "_G", "_B")):
                    original_f = f[:-2] 
                    label_src = f"{labels_path}/{original_f}.txt"
                else:
                    label_src = f"{labels_path}/{f}.txt"
            
                if os.path.exists(label_src):
                    shutil.copy(label_src, f"{dest_path}/{split}/labels/{f}.txt")

        move_files(train_files, 'train')
        move_files(valid_files, 'valid')
        move_files(test_files, 'test')

        print(f"\nTrain: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    
    yaml_content = f"""
    train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['tuta', 'oidium']
    """
    
    yaml_path = os.path.join(dest_path, f"{img_type}_data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content.strip())
        
    print(f"\nConfiguration file created: {yaml_path}")

    
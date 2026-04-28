import os
import shutil
import sys

img_type = sys.argv[1].upper()

old_path = f'data/processed/Dataset pomodori/Multi class/{img_type}'
new_path = f'data/processed/Dataset pomodori/Single Class/{img_type}'

splits = ['train', 'valid', 'test']

for split in splits:
    os.makedirs(os.path.join(new_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_path, split, 'labels'), exist_ok=True)

    img_dir = os.path.join(old_path, split, 'images')
    for img in os.listdir(img_dir):
        shutil.copy(os.path.join(img_dir, img), os.path.join(new_path, split, 'images', img))

    lab_dir = os.path.join(old_path, split, 'labels')
    for lab in os.listdir(lab_dir):
        if lab.endswith('.txt'):
            with open(os.path.join(lab_dir, lab), 'r') as f:
                lines = f.readlines()
            
            with open(os.path.join(new_path, split, 'labels', lab), 'w') as f:
                for line in lines:
                    parts = line.split()
                    if parts:
                        parts[0] = '0'
                        f.write(" ".join(parts) + "\n")

print(f"\nSingle class dataset created in: {os.path.abspath(new_path)}")

yaml_content = f"""
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['Disease']
    """
    
yaml_path = os.path.join(new_path, f"{img_type}_data.yaml")
with open(yaml_path, "w") as yaml_file:
    yaml_file.write(yaml_content.strip())
        
print(f"\nConfiguration file created: {yaml_path}")
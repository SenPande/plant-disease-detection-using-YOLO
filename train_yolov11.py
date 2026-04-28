from ultralytics import YOLO
import torch
import os
import datetime as dt
import sys

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(device)}")

    ROOT = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 3:
        print("\nMISSING ARGS: Define dataset type for the training\n")
        return None

    data_type = sys.argv[1]
    data_name = sys.argv[2].upper()
    print(f"Training on {data_type} Class {data_name} dataset")

    model_scale = 's'

    model = YOLO(f'yolo11{model_scale}.pt')

    epochs = 150

    model.train(
        data = os.path.join(ROOT, 'data', 'processed', 'Dataset pomodori', f'{data_type} Class', data_name, f'{data_name}_data.yaml'),
        epochs = epochs,
        imgsz = 640,
        batch = 16,
        workers = 2,
        device = device,
        project = os.path.join(ROOT, 'models', 'train_results', f'{dt.date.today()}', f'{data_type} Class', 'normal'),
        name = f'{data_name}_{model_scale}_{epochs}eps',
        plots = True,
        cache = True
    )

if __name__ == '__main__':
    main()
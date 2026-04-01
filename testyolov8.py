from ultralytics import YOLO
import torch
import os
import datetime as dt

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(device)}")

    ROOT = os.path.dirname(os.path.abspath(__file__))

    model = YOLO('yolov8s.pt')

    optimizer = 'AdamW'

    model.train(
        data = os.path.join(ROOT, 'data', 'test', 'data.yaml'),
        optimizer = optimizer,
        lr0 = 0.001,
        epochs = 100,
        imgsz = 512,
        batch = 2,
        workers = 2,
        device = device,
        project = os.path.join(ROOT, 'models', 'train_results', f'{dt.date.today()}'),
        name = f'Test_ExG_KAGGLE/{optimizer}',
        plots = True,
        cache = False
    )

    print("Training terminato")

if __name__ == '__main__':
    main()
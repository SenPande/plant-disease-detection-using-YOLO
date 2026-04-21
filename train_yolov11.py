from ultralytics import YOLO
import torch
import os
import datetime as dt

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(device)}")

    ROOT = os.path.dirname(os.path.abspath(__file__))

    img_type = 'rgb'

    model = YOLO('yolo11n.pt') # cambiare in 11s dopo test pseudocolori

    epochs = 100

    model.train(
        data = os.path.join(ROOT, 'data', 'processed', img_type, 'data.yaml'),
        epochs = epochs,
        imgsz = 640,
        batch = 16,
        workers = 2,
        device = device,
        project = os.path.join(ROOT, 'models', 'train_results', f'{dt.date.today()}'),
        name = f'{img_type}_{epochs}',
        plots = True,
        cache = False
    )

    print("Training terminato")

if __name__ == '__main__':
    main()
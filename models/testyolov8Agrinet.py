from ultralytics import YOLO
import torch
import os
import datetime as dt

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(device)}")

    model = YOLO('/home/involtino2/tirocinio_tesi/models/yolov8_agrinet.yaml')

    model.load('yolov8s.pt')

    optimizer = 'Adam'

    results = model.train(
        data = 'data/test/data.yaml',
        optimizer = optimizer,
        lr0 = 0.001,
        epochs = 100,
        imgsz = 512,
        batch = 2,
        workers = 2,
        device = device,
        project = os.path.join(os.getcwd(), f'models/train_results/{dt.date.today()}'),
        name = f'Test_ExG_KAGGLE_Agrinet/{optimizer}',
        plots = True,
        cache = False
    )

    print("Training Agrinet terminato")

if __name__ == '__main__':
    main()
from ultralytics import YOLO
import torch
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(device)}")

    #model = YOLO('yolov8s.pt')

    model = YOLO('/home/involtino2/tirocinio_tesi/models/yolov8_agrinet.yaml')

    model.load('yolov8s.pt')

    results = model.train(
        data = 'data/test/data.yaml',
        epochs = 100,
        imgsz = 512,
        batch = 4,
        workers = 2,
        device = device,
        project = os.path.join(os.getcwd(), 'models/train_results'),
        name = 'Test_ExG_KAGGLE_Agrinet',
        plots = True,
        cache = False
    )

    print("Training terminato")

if __name__ == '__main__':
    main()
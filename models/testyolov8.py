from ultralytics import YOLO
import torch

def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzando: {torch.cuda.get_device_name(int(device))}")

    model = YOLO('yolov8s.pt')

    results = model.train(
        data = 'data/test/data.yaml',
        epochs = 100,
        imgsz = 512,
        batch = 8,
        workers = 2,
        device = device,
        project = 'models/train_results',
        name = 'Test_ExG_KAGGLE',
        plots = True,
        cache = False
    )

    print("Training terminato")

if __name__ == '__main__':
    main()
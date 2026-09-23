import argparse
import datetime as dt
import os
import torch
from ultralytics import YOLO

DEFAULT_DATASET_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')
# !!!!!!!!!!!!!!!! Cambiare con data e modello che si intende usare come checkpoint default
DEFAULT_WEIGHTS = 'models/train_results/2026-04-23/ERGB_s_50eps/weights/best.pt'
DEFAULT_PROJECT_DIR = os.path.join('models', 'train_results')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_type")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if device == 'cuda' else 'CPU'
    print(f"Utilizzando: {device_name}")

    root = os.path.dirname(os.path.abspath(__file__))
    data_type = args.data_type.upper()
    print(f"Training on {data_type} dataset")

    model = YOLO(args.weights)
    model_scale = model.model.yaml.get('scale', 'unknown')

    model.train(
        data=os.path.join(root, args.dataset_dir, data_type, f'{data_type}_data.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=os.path.join(root, args.project_dir, f'{dt.date.today()}', 'yolo11'),
        name=f'{data_type}_{model_scale}_{args.epochs}eps',
        plots=True,
        cache=True,
    )


if __name__ == '__main__':
    main()

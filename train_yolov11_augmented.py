import argparse
import datetime as dt
import os

import torch
from ultralytics import YOLO

DEFAULT_DATASET_DIR = os.path.join('data', 'processed', 'Dataset pomodori')
DEFAULT_PROJECT_DIR = os.path.join('models', 'train_results')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_type")
    parser.add_argument("data_name")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--model-scale", default='s')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)

    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=90.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--shear", type=float, default=15.0)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if device == 'cuda' else 'CPU'
    print(f"Utilizzando: {device_name}")

    root = os.path.dirname(os.path.abspath(__file__))
    data_name = args.data_name.upper()
    print(f"Training on {args.data_type} Class {data_name} dataset")

    model = YOLO(f'yolo11{args.model_scale}.pt')

    model.train(
        data=os.path.join(root, args.dataset_dir, f'{args.data_type} class', data_name, f'{data_name}_data.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=os.path.join(root, args.project_dir, f'{dt.date.today()}', f'{args.data_type} class', 'augmented'),
        name=f'{data_name}_{args.model_scale}_{args.epochs}eps',
        plots=True,
        cache=True,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
    )


if __name__ == '__main__':
    main()

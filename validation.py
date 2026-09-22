import argparse
import os

from ultralytics import YOLO

DEFAULT_DATASET_DIR = os.path.join('data', 'processed', 'Dataset pomodori', 'Multi class')
DEFAULT_PROJECT_DIR = os.path.join('models', 'inference_results')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("date")
    parser.add_argument("test_data")
    parser.add_argument("--run-name", default='ERGB_R_s_150eps')
    parser.add_argument("--run-group", default='focal_loss')
    parser.add_argument("--data-type")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, default=0.377)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    test_data = args.test_data.upper()

    weights_path = args.weights or os.path.join(
        'models', 'train_results', args.date,
        *([f'{args.data_type} class'] if args.data_type else []),
        args.run_group, args.run_name, 'weights', 'best.pt',
    )
    model = YOLO(weights_path)
    root = os.path.dirname(os.path.abspath(__file__))

    metrics = model.val(
        data=os.path.join(args.dataset_dir, test_data, f'{test_data}_data.yaml'),
        split="test",
        conf=args.conf,
        iou=args.iou,
        project=os.path.join(root, args.project_dir, args.date, f'{args.run_name}_{test_data}'),
        name='validation',
    )

    precision = metrics.box.mp
    recall = metrics.box.mr

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print(f"\n--- RISULTATI AL THRESHOLD {args.conf} ---")
    print(f"Precision media: {precision:.3f}")
    print(f"Recall media:    {recall:.3f}")
    print(f"F1-Score medio:  {f1_score:.3f}")
    print(f"mAP@50:          {metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()

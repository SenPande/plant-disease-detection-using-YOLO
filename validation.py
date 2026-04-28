import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ultralytics import YOLO
import os
import sys

date = sys.argv[1]
best = 'ERGB_R_s_150eps'
model = YOLO(f"models/train_results/{date}/focal_loss/{best}/weights/best.pt")

#features = []
#labels = []
#
#def hook(module, input, output):
#    features.append(output.mean(dim=[2, 3]).detach().cpu().numpy())
#
#target_layer = model.model.model[9] 
#handle = target_layer.register_forward_hook(hook)
#
test_data = sys.argv[2].upper()
#
#print(f"\nValidating on {test_data} using {best} best.pt")
#
#results = model.predict(source=f"data/processed/Dataset pomodori/Multi class/{test_data}/test/images", 
#                        imgsz=640, 
#                        conf=0.25)
#
#for i, r in enumerate(results):
#    if len(r.boxes.cls) > 0:
#        labels.append(int(r.boxes.cls[0]))
#    else:
#        labels.append(-1)
#
#handle.remove()
#
#X = np.concatenate(features)
#y = np.array(labels[:len(X)])
#
#if len(X) != len(y):
#    min_len = min(len(X), len(y))
#    X = X[:min_len]
#    y = y[:min_len]
#
#print("\nCalcolo t-SNE in corso...")
#tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
#X_embedded = tsne.fit_transform(X)
#
#plt.figure(figsize=(10, 7))
#classes = ['Tuta', 'Oidium']
#colors = ['blue', 'cyan']
#
#for i, class_name in enumerate(classes):
#    indices = y == i
#    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], c=colors[i], label=class_name, alpha=0.6)
#
#plt.legend()
#plt.title("t-SNE Plot: Separazione delle classi Tuta e Oidio")
#plt.xlabel("t-SNE 1")
#plt.ylabel("t-SNE 2")
#
#output_path = f"models/inference_results/{date}/{best}_{test_data}"
#os.makedirs(output_path, exist_ok=True)
#
#plt.savefig(os.path.join(output_path, 't-SNE'), dpi=300, bbox_inches='tight')
#print(f"\nSaved in: {os.path.abspath(output_path)}")

ROOT = os.path.dirname(os.path.abspath(__file__))

metrics = model.val(data=f"data/processed/Dataset pomodori/Multi class/{test_data}/{test_data}_data.yaml", 
                    split="test", 
                    conf=0.377, 
                    iou=0.4,
                    project = os.path.join(ROOT, 'models', 'inference_results', date, f'{best}_{test_data}'),
                    name = 'validation') 

precision = metrics.box.mp
recall = metrics.box.mr

if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

print(f"\n--- RISULTATI AL THRESHOLD 0.377 ---")
print(f"Precision media: {precision:.3f}")
print(f"Recall media:    {recall:.3f}")
print(f"F1-Score medio:  {f1_score:.3f}")
print(f"mAP@50:          {metrics.box.map50:.3f}")
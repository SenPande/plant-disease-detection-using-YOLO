import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ultralytics import YOLO
import os
import sys

best = 'ERGB_G_s_150eps'
model = YOLO(f"models/train_results/2026-04-27/normal/{best}/weights/best.pt")

features = []
labels = []

def hook(module, input, output):
    features.append(output.mean(dim=[2, 3]).detach().cpu().numpy())

target_layer = model.model.model[9] 
handle = target_layer.register_forward_hook(hook)

test_data = sys.argv[1].upper()

print(f"\nValidating on {test_data} using {best} best.pt")

results = model.predict(source=f"data/processed/Dataset pomodori/{test_data}/test/images", imgsz=640, conf=0.25)

for i, r in enumerate(results):
    if len(r.boxes.cls) > 0:
        labels.append(int(r.boxes.cls[0]))
    else:
        labels.append(-1)

handle.remove()

X = np.concatenate(features)
y = np.array(labels[:len(X)])

if len(X) != len(y):
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

print("\nCalcolo t-SNE in corso...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
classes = ['Tuta', 'Oidium']
colors = ['blue', 'cyan']

for i, class_name in enumerate(classes):
    indices = y == i
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], c=colors[i], label=class_name, alpha=0.6)

plt.legend()
plt.title("t-SNE Plot: Separazione delle classi Tuta e Oidio")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

output_path = f"models/inference_results/{best}_{test_data}"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved in: {os.path.abspath(output_path)}")
from ultralytics import YOLO
import cv2
import os
import datetime as dt

optimizer = 'AdamW'

ROOT = os.path.dirname(os.path.abspath(__file__))

today = str(dt.date.today())

index = 'ExG'

# TODO: richiamare questa funzione nel main dando come parametro l'indice di vegetazione
model = YOLO(os.path.join(ROOT, 'models', 'train_results', today, f'Test_{index}_KAGGLE', optimizer, 'weights', 'best.pt'))
model_agrinet = YOLO(os.path.join(ROOT, 'models', 'train_results', today, f'Test_{index}_KAGGLE_Agrinet', optimizer, 'weights', 'best.pt'))
model_leafnet = YOLO(os.path.join(ROOT, 'models', 'train_results', today, f'Test_{index}_KAGGLE_Leafnet', optimizer, 'weights', 'best.pt'))

IMAGES_FOLDER = os.path.join('data', 'test', 'valid', 'images')
test_images = {
    "bacterial": os.path.join(ROOT, IMAGES_FOLDER, 'bct-12-_ExG_jpg.rf.292f6521ad2d776ebe4e15a0060e12d0.jpg'),
    "downy": os.path.join(ROOT, IMAGES_FOLDER, 'dml-16-_ExG_jpg.rf.35afa5f4da75adc94f9a4b1f1de6dbe9.jpg'),
    "healthy": os.path.join(ROOT, IMAGES_FOLDER, 'h-22-_ExG_jpg.rf.17458ae8eb60b0b44bd9ade1791e7266.jpg'),
    "powdery": os.path.join(ROOT, IMAGES_FOLDER, 'pml-2-_ExG_jpg.rf.c2d34cd04a10f320ab53d733c85b5236.jpg'),
    "septoria": os.path.join(ROOT, IMAGES_FOLDER, 'sbl-10-_ExG_jpg.rf.6e8e7743e18faa1fbf22b0c9f87762c3.jpg')
}

path_std = os.path.join(ROOT, 'models', 'inference_results', today, 'yolov8', index, optimizer)
path_agrinet = os.path.join(ROOT, 'models', 'inference_results', today, 'yolov8_agrinet', index, optimizer)
path_leafnet = os.path.join(ROOT, 'models', 'inference_results', today, 'yolov8_leafnet', index, optimizer)
os.makedirs(path_std, exist_ok=True)
os.makedirs(path_agrinet, exist_ok=True)
os.makedirs(path_leafnet, exist_ok=True)

for disease, path in test_images.items():
    results = model.predict(
        source=path, 
        conf=0.25,     
        device = 0 
    )[0]
    res_plotted = results.plot()
    cv2.imwrite(os.path.join(path_std, f"result_{disease}.jpg"), res_plotted)

    results_agrinet = model_agrinet.predict(
        source=path, 
        conf=0.1,   
        device = 0   
    )[0]
    res_agrinet_plotted = results_agrinet.plot()
    cv2.imwrite(os.path.join(path_agrinet, f"result_agrinet_{disease}.jpg"), res_agrinet_plotted)

    results_leafnet = model_leafnet.predict(
        source=path, 
        conf=0.1,     
        device = 0 
    )[0]
    res_leafnet_plotted = results_leafnet.plot()
    cv2.imwrite(os.path.join(path_leafnet, f"result_{disease}.jpg"), res_leafnet_plotted)

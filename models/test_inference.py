from ultralytics import YOLO
import cv2
import os
import datetime

optimizer = 'AdamW'

model = YOLO(f'/home/involtino2/tirocinio_tesi/models/train_results/Test_ExG_KAGGLE/{optimizer}/weights/best.pt')
model_agrinet = YOLO(f'/home/involtino2/tirocinio_tesi/models/train_results/Test_ExG_KAGGLE_Agrinet/{optimizer}/weights/best.pt')

test_images = {
    "bacterial": "/home/involtino2/tirocinio_tesi/data/test/valid/images/bct-12-_ExG_jpg.rf.292f6521ad2d776ebe4e15a0060e12d0.jpg",
    "downy": "/home/involtino2/tirocinio_tesi/data/test/valid/images/dml-16-_ExG_jpg.rf.35afa5f4da75adc94f9a4b1f1de6dbe9.jpg",
    "healthy": "/home/involtino2/tirocinio_tesi/data/test/valid/images/h-22-_ExG_jpg.rf.17458ae8eb60b0b44bd9ade1791e7266.jpg",
    "powdery": "/home/involtino2/tirocinio_tesi/data/test/valid/images/pml-2-_ExG_jpg.rf.c2d34cd04a10f320ab53d733c85b5236.jpg",
    "septoria": "/home/involtino2/tirocinio_tesi/data/test/valid/images/sbl-10-_ExG_jpg.rf.6e8e7743e18faa1fbf22b0c9f87762c3.jpg"
}

path_std = "/home/involtino2/tirocinio_tesi/models/train_results/inference_results/yolov8"
path_agrinet = "/home/involtino2/tirocinio_tesi/models/train_results/inference_results/yolov8_agrinet"

for disease, path in test_images.items():
    results = model.predict(
        source=path, 
        conf=0.25,     
        device = 0 
    )[0]
    res_plotted = results.plot()
    cv2.imwrite(os.path.join(path_std, f"result_{disease}.jpg"), res_plotted)


    res_agrinet = model_agrinet.predict(
        source=path, 
        conf=0.1,   
        device = 0   
    )[0]
    res_agrinet_plotted = res_agrinet.plot()
    cv2.imwrite(os.path.join(path_agrinet, f"result_agrinet_{disease}.jpg"), res_agrinet_plotted)


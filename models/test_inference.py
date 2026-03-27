from ultralytics import YOLO
import cv2
import os

model = YOLO('/home/involtino2/tirocinio_tesi/models/train_results/Test_ExG_KAGGLE/weights/best.pt')
model_agrinet = YOLO('/home/involtino2/tirocinio_tesi/models/train_results/Test_ExG_KAGGLE_Agrinet/weights/best.pt')


# PER RISULTATI
# /home/involtino2/tirocinio_tesi/data/test/valid/images/bct-12-_ExG_jpg.rf.292f6521ad2d776ebe4e15a0060e12d0.jpg bacterial
# /home/involtino2/tirocinio_tesi/data/test/valid/images/dml-16-_ExG_jpg.rf.35afa5f4da75adc94f9a4b1f1de6dbe9.jpg downyMildew
# /home/involtino2/tirocinio_tesi/data/test/valid/images/h-22-_ExG_jpg.rf.17458ae8eb60b0b44bd9ade1791e7266.jpg healty
# /home/involtino2/tirocinio_tesi/data/test/valid/images/pml-2-_ExG_jpg.rf.c2d34cd04a10f320ab53d733c85b5236.jpg powderyMildew
# /home/involtino2/tirocinio_tesi/data/test/valid/images/sbl-10-_ExG_jpg.rf.6e8e7743e18faa1fbf22b0c9f87762c3.jpg septoriaBlight
results = model.predict(
    source = '/home/involtino2/tirocinio_tesi/data/test/valid/images/sbl-10-_ExG_jpg.rf.6e8e7743e18faa1fbf22b0c9f87762c3.jpg',
    conf = 0.25,
    device = 0
)

for r in results:
    res_plotted = r.plot()
    save_path = os.path.join('models/train_results', 'inference_result_septoriaBlight2.jpg')
    cv2.imwrite(save_path, res_plotted)


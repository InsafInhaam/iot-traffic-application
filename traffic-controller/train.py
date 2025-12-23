from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # stronger than n-model

model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8
)

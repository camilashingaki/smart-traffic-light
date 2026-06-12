from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../dataset_modelv2/data.yaml",
    epochs=100,
    imgsz=640,
    patience=15,
    plots=True,
)
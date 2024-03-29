from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
model.train(data='custom.yaml', epochs=5, imgsz=640, device='mps')
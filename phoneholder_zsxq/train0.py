from ultralytics import YOLO

# Load the model.
model = YOLO('./yolov8n.pt')
 
# Training.
results = model.train(data="/自己的目录地址替换/pholderv11/data/phoneholder.yaml")
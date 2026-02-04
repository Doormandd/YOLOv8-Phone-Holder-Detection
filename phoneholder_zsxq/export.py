from ultralytics import YOLO

# Load the model.
model = YOLO('/Users/zhaozhipeng/Documents/code/yolo/projects/pholderv11/best.pt')
model.export(format="saved_model")
 
from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model="./yolov8n.pt", data="/自己的目录地址替换/pholderv11/data/phoneholder.yaml", epochs=30)
trainer = DetectionTrainer(overrides=args)
trainer.train()
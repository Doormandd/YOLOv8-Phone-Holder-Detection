from ultralytics import YOLO, checks, hub
checks()

hub.login('b7ce7357b50ba9d6dfda38b73dc0bdc8dbc896b45c')

model = YOLO('https://hub.ultralytics.com/models/joERieNz8iRtC64DjXTl')
results = model.train()
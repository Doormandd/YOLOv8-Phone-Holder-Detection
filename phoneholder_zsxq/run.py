import cv2
from ultralytics import YOLO
import datetime

# 加载YOLOv8模型
model = YOLO('/自己的目录地址替换/pholderv11/best.pt')

# 打开默认摄像头 (设备索引为 0)
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取摄像头的每一帧
    ret, frame = cap.read()

    # 如果读取帧失败，则退出循环
    if not ret:
        print("无法接收帧（可能是摄像头断开）")
        break

    # 使用YOLOv8进行检测
    #frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA)
    results = model(frame)

    # 解析结果并绘制检测框
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
            confidence = box.conf[0]  # 获取置信度
            if confidence < 0.7 :
                continue
            cls = int(box.cls[0])  # 获取类别
            label = model.names[cls]  # 获取类别名称

            # 绘制检测框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 获取当前时间
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示当前帧
    cv2.imshow('Camera', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
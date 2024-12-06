from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO(r"E:\python_pj\yolov8\YOLOv8-main\runs\obb\train5\weights\best.pt")

# 导出模型为ONNX格式
model.export(format="onnx")

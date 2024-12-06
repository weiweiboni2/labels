import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

# 加载标签映射
with open(r'E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\label.json', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

# 定义模型路径和保存路径
model_path = Path(r'E:\python_pj\yolov8\YOLOv8-main\runs\obb\train5\weights\best.pt')
save_path = Path(r"E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\2.mp4")

# 加载中文字体
font_path = r'C:\Windows\Fonts\msyh.ttc'  # 替换为系统中的中文字体路径
font = ImageFont.truetype(font_path, 20)

def yolo_pre():
    yolo = YOLO(model_path)

    # 输入视频路径
    video_path = r'E:\python_pj\yolov8\YOLOv8-main\data\caise.mp4'
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建 VideoWriter 对象用于保存检测后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        # 使用 YOLO 模型进行预测
        results = yolo.predict(source=frame, save=False)

        # 使用 PIL 格式转换 OpenCV 图像
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for result in results:
            new_names = {class_id: label_mapping.get(name, name) for class_id, name in result.names.items()}
            result.names = new_names

            # 绘制蓝色的边界框
            for box, cls, score in zip(result.boxes.xywh, result.boxes.cls, result.boxes.conf):
                x1, y1, w, h = box
                x1, y1, w, h = int(x1 - w / 2), int(y1 - h / 2), int(w), int(h)

                label_names = result.names.get(int(cls))
                color = (255, 0, 0)  # 蓝色框 (BGR格式)

                # 绘制边界框和标签
                draw.rectangle([(x1, y1), (x1 + w, y1 + h)], outline=color, width=2)
                draw.text((x1, y1 - 20), label_names, font=font, fill=color)

            # 将 PIL 图像转换回 OpenCV 格式
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 写入带注释的帧
            out.write(frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('保存完成')

    return str(save_path)


# 调用函数
saved_video_path = yolo_pre()
print(f"检测结果视频已保存至: {saved_video_path}")

# import torch
 
# # 加载模型
# w=torch.load('./weights\best.pt')
 
# #打印所有name
# print(w.get('model').names)
 
# # 定义一个将英文单词映射到中文单词的字典
# word_map = {
#     'person': '人',
#     
# }
 
# # 遍历列表，将每个英文单词替换为其中文对应词
# for i in range(len(w.get('model').names)):
#     if w.get('model').names[i] in word_map:
#         w.get('model').names[i] = word_map[w.get('model').names[i]]
 
# # 打印替换后的列表
# print('替换后')
# print(w.get('model').names)
# #保存替换后的模型
# torch.save(w,'./outputs/new_model.pt')
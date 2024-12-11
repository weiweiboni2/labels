import os
import cv2
import numpy as np


def select_color(id):
    if id == "0":
        color = (0, 255, 0)
    elif id == "1":
        color = (0, 0, 255)
    elif id == "2":
        color = (255, 0, 0)
    elif id == "3":
        color = (125, 125, 0)
    elif id == "4":
        color = (125, 0, 255)
    return color


# 'Gloves', 'Helmet', 'Human', 'Safety Boot', 'Safety Vest'
def select_name(id):
    if id == "0":
        name = 'Gloves'
    elif id == "1":
        name = 'Helmet'
    elif id == "2":
        name = 'Human'
    elif id == "3":
        name = 'Safety_Boot'
    elif id == "4":
        name = 'Safety_Vest'
    return name


def visualize_polygons(img_path, label_path, output_path=None):
    # 读取原始图像
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # 打开对应的标签文件
    with open(label_path, "r") as f:
        labels = f.readlines()

    # 遍历每个标签
    for label in labels:
        parts = label.strip().split()
        if not parts:  # 修复：检查parts是否为空
            continue  # 修复：跳过空行
        class_id = int(float(parts[0]))  # 类别ID
        coords = list(map(float, parts[1:]))  # 归一化的顶点坐标

        # 反归一化顶点坐标
        points = [(int(coords[i] * width), int(coords[i + 1] * height)) for i in range(0, len(coords), 2)]
        points = np.array(points, dtype=np.int32)

        # 绘制多边形框
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=5)

        # # 在框的中心写类别ID
        # center_x = int(sum([p[0] for p in points]) / len(points))
        # center_y = int(sum([p[1] for p in points]) / len(points))
        # cv2.putText(img, str(class_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5)

    # 如果指定输出路径，则保存可视化结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)


def visualize_box(img_path, label_path, output_path=None):
    # 读取原始图像
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # 打开对应的标签文件
    with open(label_path, "r") as f:
        labels = f.readlines()

    # 遍历每个标签
    for label in labels:
        temp = label.strip().split()
        color = select_color(temp[0])
        if not temp:  # 修复：检查parts是否为空
            continue  # 修复：跳过空行
        x_, y_, w_, h_ = eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])

        x1, y1, x2, y2 = int((x_ - w_ / 2) * img.shape[1]), int((y_ - h_ / 2) * img.shape[0]), \
                         int((x_ + w_ / 2) * img.shape[1]), int((y_ + h_ / 2) * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
        # 在框的中心写类别ID
        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        cv2.putText(img, str(select_name(temp[0])), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    2)

    # 保存或显示图像

    # 如果指定输出路径，则保存可视化结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)


# 示例调用
img_folder = ''
label_folder = ''
output_folder = ''

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历图片和标签文件进行可视化
for img_file in os.listdir(img_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.bmp', '.png')):
        img_path = os.path.join(img_folder, img_file)
        name, _ = os.path.splitext(img_file)
        label_path = os.path.join(label_folder, name + ".txt")
        output_path = os.path.join(output_folder, img_file)
        # 旋转框显示
        # visualize_polygons(img_path, label_path, output_path)
        # 矩形框显示
        visualize_box(img_path, label_path, output_path)

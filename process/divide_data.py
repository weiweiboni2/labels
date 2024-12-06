import os
import random
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, output_dir, split_ratio=0.8):
    image_files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]

    # 按照split_ratio划分数据集
    train_images, test_images = train_test_split(image_files, train_size=split_ratio, random_state=42)

    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

    # 定义移动文件的函数
    def move_files(image_list, subset_dir):
        for image_file in image_list:
            # 复制图像文件
            shutil.copy(os.path.join(data_dir, image_file), os.path.join(subset_dir, "images", image_file))

            # 查找对应的标签文件（假设标签文件的扩展名为 .xml）
            label_file = os.path.splitext(image_file)[0] + '.txt'
            if os.path.exists(os.path.join(txt_dir, label_file)):
                shutil.copy(os.path.join(txt_dir, label_file), os.path.join(subset_dir, "labels", label_file))
            else:
                print(f"Warning: {label_file} 标签文件不存在，跳过该文件。")

    # 移动训练集和测试集文件
    move_files(train_images, train_dir)
    move_files(test_images, test_dir)

    print(f"数据集已成功划分：训练集 {len(train_images)} 张，测试集 {len(test_images)} 张。")


# 使用示例
data_dir = r"E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\pro_data\deeee\test\images"  # 原始数据集路径
txt_dir = r'E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\pro_data\deeee\test\labels'
output_dir = r"E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\pro_data\deeee\val"  # 输出路径
split_ratio = 0.8  # 划分比例，例如 0.8 表示80%的数据划入训练集

split_dataset(data_dir, output_dir, split_ratio)

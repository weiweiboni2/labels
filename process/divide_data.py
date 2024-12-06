import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    image_files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]

    # 第一次划分：将数据集划分为训练集和临时集
    train_images, temp_images = train_test_split(image_files, train_size=train_ratio, random_state=42)

    # 第二次划分：将临时集划分为测试集和验证集
    test_ratio = val_ratio / (1 - train_ratio)
    val_images, test_images = train_test_split(temp_images, train_size=test_ratio, random_state=43)

    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

    # 定义移动文件的函数
    def move_files(image_list, subset_dir):
        for image_file in image_list:
            # 复制图像文件
            shutil.copy(os.path.join(data_dir, image_file), os.path.join(subset_dir, "images", image_file))

            # 查找对应的标签文件（假设标签文件的扩展名为 .txt）
            label_file = os.path.splitext(image_file)[0] + '.txt'
            if os.path.exists(os.path.join(txt_dir, label_file)):
                shutil.copy(os.path.join(txt_dir, label_file), os.path.join(subset_dir, "labels", label_file))
            else:
                print(f"Warning: {label_file} 标签文件不存在，跳过该文件。")

    # 移动训练集、验证集和测试集文件
    move_files(train_images, train_dir)
    move_files(val_images, val_dir)
    move_files(test_images, test_dir)

    print(f"数据集已成功划分：训练集 {len(train_images)} 张，验证集 {len(val_images)} 张，测试集 {len(test_images)} 张。")

# 使用示例
data_dir = ""  # 原始数据集路径
txt_dir = ''
output_dir = ""  # 输出路径
train_ratio = 0.8  # 训练集占比
val_ratio = 0.1  # 验证集在剩余数据中的占比

split_dataset(data_dir, output_dir, train_ratio, val_ratio)
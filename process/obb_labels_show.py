import cv2
import os
import numpy as np


# 遍历文件夹获取指定扩展名的文件
def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            if ext is None or os.path.splitext(filepath)[1][1:] in ext:
                allfiles.append(filepath)
    return allfiles


# 在图像上可视化旋转框
def visualise_gt(label_path, pic_path, newpic_path):
    results = GetFileFromThisRootDir(label_path, ext=['txt'])  # 假设标签文件为.txt格式
    for result in results:
        with open(result, 'r') as f:
            lines = f.readlines()

            # 检查文件是否为空
            if not lines:
                print('文件为空', result)
                continue

            # 获取对应图像文件路径并读取图像
            name = os.path.splitext(os.path.basename(result))[0]
            filepath = os.path.join(pic_path, f"{name}.png")
            im = cv2.imread(filepath)

            # 如果图像无法打开，跳过该文件
            if im is None:
                print(f"无法打开图片: {filepath}")
                continue

            # 获取图像尺寸
            img_height, img_width = im.shape[:2]

            # 解析并过滤标签行
            boxes = []
            for line in lines:
                parts = line.strip().split(' ')
                # 将相对坐标转换为绝对坐标
                parts = list(map(float, filter(None, parts)))  # 先将字符串转换为浮点数
                if len(parts) >= 8:
                    # 按照x坐标和y坐标分别乘以图像的宽和高
                    abs_coords = [
                        parts[i] * (img_width if i % 2 == 1 else img_height) for i in range(1, 9)
                    ]
                    boxes.append(np.array(abs_coords, dtype=np.float64))

            # 如果没有有效的框，跳过
            if not boxes:
                print('没有有效的框数据', result)
                continue

            # 绘制每个旋转框
            for box in boxes:
                box = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
                box = box.reshape((-1, 1, 2))
                cv2.polylines(im, [box], True, (0, 0, 255), 2)  # 红色框线，厚度为2

            # 保存绘制后的图像
            output_path = os.path.join(newpic_path, f"{name}.png")
            cv2.imwrite(output_path, im)
            print(f"已保存: {output_path}")


if __name__ == '__main__':
    # 设置路径
    pic_path = ""  # 原始图片文件夹路径
    label_path = ""  # 标签文件夹路径
    newpic_path = ''  # 输出图片保存路径

    # 创建输出文件夹（若不存在）
    if not os.path.isdir(newpic_path):
        os.makedirs(newpic_path)

    # 可视化旋转框
    visualise_gt(label_path, pic_path, newpic_path)

import os
from tqdm import tqdm
import random
import cv2
import numpy as np
import pyhocon
import albumentations as A
import shutil
import time
from PIL import Image
import skimage


class aug:
    # 图像增强
    def __init__(self, ImP: str, LaP: str, ExImP: str, ExLaP: str, file_format: str = 'yolo_txt') -> None:
        self.ImagePath = ImP
        self.LabelPath = LaP
        self.ExportImagePath = ExImP
        self.ExportLabelPath = ExLaP
        self.FileFormat = file_format

    # --------------------------------------------------Foundation Work--------------------------------------------------------------

    def get_file_name(self, path):
        '''
        获取目录下的文件名字(带后缀)
        Args:
            path: 文件路径
        '''
        file_names = []
        for file_entry in os.scandir(path):
            if file_entry.is_file():
                file_names.append(file_entry.name)
        return file_names

    def get_bboxes(self, txt_file_path):
        '''
        获取yolo_txt文件中的矩形框坐标
        Args:
            txt_file_path: txt文件路径
        '''
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split())
            bboxes.append([class_id, x1, y1, x2, y2, x3, y3, x4, y4])
        return bboxes

    def label2txt(self, labelInfo, txtPath):
        with open(txtPath, 'w') as f:
            f.writelines([line + os.linesep for line in labelInfo])

    # -----------------------------------------------Label Conversion-------------------------------------------------------------
    # 1. 旋转
    # 2. 水平镜像
    # 3. 垂直镜像
    # 4. 水平垂直镜像
    # 5. 随机裁剪
    # 6. 添加噪声（添加天气情况）

    def __rotate_and_scale_point(self, x, y, w, h, cos, sin,  crop_coords):
        '''辅助函数： 旋转图片对应的label变换
        Args:
        每个点的坐标变换 
        new_height, height, new_width, width = crop_coords
        '''
        y00, y11, x00, x11 = crop_coords
        # 缩放比例
        d = y00 /y11

        if w == h:
            new_x = (x - 0.5) * cos - (y - 0.5) * sin + 0.5*d
            new_y = (y - 0.5) * cos + (x - 0.5) * sin + 0.5*d
        else:
            cx = w / 2
            cy = h / 2
            # 反归一化
            code_h = y*h
            code_w = x*w
            # 将点移动到旋转中心
            tx = code_w - cx
            ty = code_h - cy
            # 旋转点
            rotated_x = tx * cos - ty * sin
            rotated_y = tx * sin + ty * cos
            
            # 将旋转后的点移回原位置
            new_x = rotated_x + cx
            new_y = rotated_y + cy

        return new_x, new_y

    def getBboxRotate(self, txt_file_path, angle, w, h, crop_coords):
        '''旋转图像和坐标 width, height
        '''
        bbox = []
        new_height, height, new_width, width = crop_coords
        bboxes = self.get_bboxes(txt_file_path)
        center = (w / 2,h / 2)
        cos = np.cos(np.deg2rad(-angle))  # 修正了角度，使其符合常规的顺时针旋转
        sin = np.sin(np.deg2rad(-angle))  # 同上
        # 旋转框坐标
        rotated_boxes = []
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            
            # 对每个点进行旋转和缩放变换

            new_x1, new_y1 = self.__rotate_and_scale_point(x1, y1, w, h, cos, sin, crop_coords)
            new_x2, new_y2 = self.__rotate_and_scale_point(x2, y2, w, h, cos, sin, crop_coords)
            new_x3, new_y3 = self.__rotate_and_scale_point(x3, y3, w, h, cos, sin, crop_coords)
            new_x4, new_y4 = self.__rotate_and_scale_point(x4, y4, w, h, cos, sin, crop_coords)
            # 计算中心位置偏移
            x_py = (new_width-width)/2
            y_py = (new_height-height)/2

            new_x1 = (new_x1 + x_py)/new_width

            new_y1 = (new_y1 + y_py)/new_height

            new_x2 = (new_x2 + x_py)/new_width

            new_y2 = (new_y2 + y_py)/new_height

            new_x3 = (new_x3 + x_py)/new_width

            new_y3 = (new_y3 + y_py)/new_height

            new_x4 = (new_x4 + x_py)/new_width

            new_y4 = (new_y4 + y_py)/new_height
            # 将坐标转换为字符串形式
            space_separated_string = ' '.join(map(str, [class_id, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]))
            bbox.append(space_separated_string)
        
        return bbox

    def getBbox2MirrorHorizon(self, txt_path):
        '''水平镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边界框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, 1 - x1, y1, 1 - x2, y2, 1 - x3, y3, 1 - x4, y4]))
            bbox.append(space_separated_string)
        return bbox

    def getBbox2Vertical(self, txt_path):
        '''垂直镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边界框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, x1, 1 - y1, x2, 1 - y2, x3, 1 - y3, x4, 1 - y4]))
            bbox.append(space_separated_string)
        return bbox

    def getBbox2HV(self, txt_path):
        '''水平垂直镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(
                map(str, [class_id, 1 - x1, 1 - y1, 1 - x2, 1 - y2, 1 - x3, 1 - y3, 1 - x4, 1 - y4]))
            bbox.append(space_separated_string)
        return bbox

    def getBbox2Resize(self, txt_path, start_x, start_y, crop_width, crop_height,w , h):
        '''随机裁剪
        Args:
            txt_path: txt文件路径
            scale: 缩放比例
            left: 左边距
            top: 顶部边距
        Returns:
            bbox: 变换后的边框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            id = bbox_data[:1][0]
            codes = bbox_data[1:]
            adjusted_bbox_data = [codes[i] * w if i % 2 == 0 else codes[i] * h for i in range(len(codes))]
            # 解包调整后的坐标值
            x1, y1, x2, y2, x3, y3, x4, y4 = adjusted_bbox_data
            if (start_x <= x1 <= start_x + crop_width and
                    start_y <= y1 <= start_y + crop_height and
                    start_x <= x2 <= start_x + crop_width and
                    start_y <= y2 <= start_y + crop_height and
                    start_x <= x3 <= start_x + crop_width and
                    start_y <= y3 <= start_y + crop_height and
                    start_x <= x4 <= start_x + crop_width and
                    start_y <= y4 <= start_y + crop_height):
                    # 更新边界框坐标以反映裁剪后的图像尺寸
                    new_x1 = max(x1 - start_x, 0) / crop_width
                    new_y1 = max(y1 - start_y, 0) / crop_height
                    new_x2 = min(x2 - start_x, crop_width) / crop_width
                    new_y2 = min(y2 - start_y, crop_height) / crop_height
                    new_x3 = min(x3 - start_x, crop_width) / crop_width
                    new_y3 = min(y3 - start_y, crop_height) / crop_height
                    new_x4 = min(x4 - start_x, crop_width) / crop_width
                    new_y4 = min(y4 - start_y, crop_height) / crop_height
                    space_separated_string = ' '.join(map(str, [id, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]))
                    bbox.append(space_separated_string)

        return bbox

    # --------------------------------------------Image Conversion------------------------------------------------------
    # 1. 旋转
    # 2. 水平镜像
    # 3. 垂直镜像
    # 4. 水平垂直镜像
    # 5. 随机裁剪
    # 6. 添加噪声（添加天气情况）

    def Rotate(self, angle=225, ratio=1.0):
        '''旋转图像和坐标
        Args:
            angle: 旋转角度
            ratio: 数据增强的概率
        '''
        flag = '225'
        Filelist = self.get_file_name(self.ImagePath)
        for filename in tqdm(Filelist):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            center = (width / 2, height / 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            matrix = cv2.getRotationMatrix2D(center, angle, 1)

            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            matrix[0, 2] += (new_width - width) / 2
            matrix[1, 2] += (new_height - height) / 2
            
            # 裁剪坐标y0, y1, x0, x1
            crop_coords = (new_height, height, new_width, width)

            rotated = cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(0, 0, 0))

            rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)

            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.png', rotated_bgr)
            labelInfo = self.getBboxRotate(self.LabelPath + '/' + name_only + '.txt', angle, width, height, crop_coords)
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')

    def MirrorHorizon(self, ratio=1.0):
        '''水平镜像
        '''
        flag = '001'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(image, 1)

            flipped_bgr = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.png', flipped_bgr)
            labelInfo = self.getBbox2MirrorHorizon(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')

    def MirrorVertical(self, ratio=1.0):
        '''垂直镜像
        Args:
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '010'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_foloat = random.uniform(0, 1)
            if ratio < random_foloat:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filpped = cv2.flip(image, 0)
            filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.png', filpped_bgr)
            labelInfo = self.getBbox2Vertical(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')

    def MirrorHV(self, ratio=1.0):
        '''水平垂直镜像
        Args:
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '011'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filpped = cv2.flip(image, -1)
            filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.png', filpped_bgr)
            labelInfo = self.getBbox2HV(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')

    def RandomCrop(self, min_scale=0.3, max_scale=0.7, ratio=1.0):
        '''随机裁剪:
            min_scale (float): 裁剪尺寸的最小比例,默认为0.3。
            max_scale (float): 裁剪尺寸的最大比例,默认为0.7。
            ratio (float): 裁剪比例阈值,默认为1.0。
        '''
        flag = '200'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_foloat = random.uniform(0, 1)
            if ratio < random_foloat:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            random_scale = random.uniform(min_scale, max_scale)
            crop_height = int(height * random_scale)
            crop_width = int(width * random_scale)

            # 随机选择裁剪的起始点
            start_x = random.randint(0, width - crop_width)
            start_y = random.randint(0, height - crop_height)

            # 裁剪图像
            cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.png', cropped_image)
            labelInfo = self.getBbox2Resize(self.LabelPath + '/' + name_only + '.txt', start_x, start_y, crop_width, crop_height, width, height)
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')

    def AddWeather(self, ratio=1):
        '''添加噪声(AddWeather:对文件夹中的图片进行天气增强 1:1:1:1=雨天:雪天:日光:阴影)
        Args:
            ratio: 数据增强的概率
        '''
        flag = '003'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # random_number = random.randint(0, 3)
            random_number = 3
        
            if random_number == 0:
                # 随机雨效果
                transform = A.Compose(
                    [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],
                )
            elif random_number == 1:
                # 随机遮挡
                transform = A.Compose(
                    [A.CoarseDropout(max_holes=10, max_height=50, max_width=35, p=1)],
                )
            elif random_number == 2:
                # 随机亮度和对比度调整：
                brightness_limit = random.uniform(-0.1, 0.1)  # 亮度调整范围，可以根据需要调整
                contrast_limit = random.uniform(0.9, 1.1)  # 对比度调整范围，可以根据需要调整
                transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1),
                     A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=1.0)]  # 调整色调、饱和度和亮度],
                )
            elif random_number == 3:
                # 高斯噪声
                var_limit = (250, 1500)  # 高斯噪声的方差范围，可以根据需要调整
                transform = A.Compose(
                    [A.GaussNoise(var_limit=var_limit, p=1) 
                     ],
                )
            elif random_number == 4:
                # 随机阴影效果
                transform = A.Compose(
                    [A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_dimension=6, num_shadows_lower=1,
                                     num_shadows_upper=1, shadow_dimension_order="snrd", always_apply=True, p=1)],
                )
            elif random_number == 5:
                # 随机模糊# 以50%的概率应用模糊，模糊程度在3到7之间
                transform = A.Compose(
                    [A.Blur(blur_limit=(3, 7), p=1)],
                )
            transformed = transform(image=image)
            # Convert the transformed image back to a PIL image
            transformed_pil = Image.fromarray(transformed['image'])

            # Save the transformed image as a TIF image
            transformed_pil.save(self.ExportImagePath + '/' + name_only + flag + '.png')

            shutil.copy(self.LabelPath + '/' + name_only + '.txt',
                        self.ExportLabelPath + '/' + name_only + flag + '.txt')


if __name__ == "__main__":
    out_img = r'D:\soft\git\weiweiboni2_project\github\labels\process\sa\img'
    os.makedirs(out_img, exist_ok=True)
    out_label = r'D:\soft\git\weiweiboni2_project\github\labels\process\sa\label'
    os.makedirs(out_label, exist_ok=True)
    aug = aug(r'D:\soft\git\weiweiboni2_project\github\labels\process\va',
              r'D:\soft\git\weiweiboni2_project\github\labels\process\la',
              out_img,
              out_label)

    # aug.Rotate()
    # aug.MirrorHV()
    # aug.AddWeather()
    # aug.RandomCrop()
    # aug.MirrorVertical()
    # aug.MirrorHorizon()

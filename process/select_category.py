import os
import cv2 as cv
import argparse
import shutil

class CategoryFilter:
    def __init__(self, txt_path, save_txt_path, images_path, save_img_path, dataset_type):
        self.txt_path = txt_path
        self.save_txt_path = save_txt_path
        self.images_path = images_path
        self.save_img_path = save_img_path
        self.dataset_type = dataset_type
   
    def process(self):
        self.category_filter()
        self.check_and_delete_empty_txt_files()

     # 只保留’0‘person这个类别，其余全剔除
    def category_filter(self):
        txt_path = os.path.join(self.txt_path, self.dataset_type)
        save_path = os.path.join(self.save_txt_path, self.dataset_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in os.listdir(txt_path):
            txt = os.path.join(txt_path, i)
            with open(txt, "r") as file:  #使用绝对路径 开文件
                lines = file.readlines()
                # 保留以 '0' 开头的行
                filtered_lines = [line for line in lines if line.strip().startswith("1") or line.strip().startswith("4")] 
                save_dir = os.path.join(save_path, i)
                with open(save_dir, "w") as file:
                    file.writelines(filtered_lines)

    def check_and_delete_empty_txt_files(self):
        # 遍历生成的所选类别的txt目录中的所有文件
        input_dir = os.path.join(self.save_txt_path, self.dataset_type)
        input_img_path = os.path.join(self.images_path, self.dataset_type)
        output_img_path = os.path.join(self.save_img_path, self.dataset_type)
        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)
        for filename in os.listdir(input_dir):
            # 检查文件是否是.txt文件
            if filename.endswith(".txt"):
                file_path = os.path.join(input_dir, filename)
                image_name = os.path.splitext(filename)[0] + '.jpg'
                image_path = os.path.join(input_img_path, image_name)
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    # 文件为空，删除该文件并记录文件名
                    os.remove(file_path)
                    print(f"Deleted empty text file: {filename}")
                elif os.path.getsize(file_path) > 0:
                    # 构建新的图片路径
                    new_image_path = os.path.join(output_img_path, image_name) 
                    # 复制图片到新的路径
                    shutil.copy2(image_path, new_image_path)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--txt_path', type=str, default=r'E:\python_pj\yolov8\YOLOv8-main\data\Security_Gear_Check5\label5')
    parser.add_argument('--save_txt_path', type=str, default=r'E:\python_pj\yolov8\YOLOv8-main\data\Security_Gear_Check5\labels')
    parser.add_argument('--images_path', type=str, default=r'E:\python_pj\yolov8\YOLOv8-main\data\Security_Gear_Check5\images5')
    parser.add_argument('--save_img_path', type=str, default=r'E:\python_pj\yolov8\YOLOv8-main\data\Security_Gear_Check5\images')
    parser.add_argument('--dataset_type', type=str, default='train', help='train/val/test')
    args = parser.parse_args()
    c_filter = CategoryFilter(args.txt_path, args.save_txt_path, args.images_path, args.save_img_path, args.dataset_type)
    c_filter.process()

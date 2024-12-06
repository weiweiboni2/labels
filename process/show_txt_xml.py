# import cv2

# img = cv2.imread(r'E:\data\test3\0000133_00846_d_0000143.jpg')

# with open(r'E:\data\test3\0000133_00846_d_0000143.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         # temp = f.read()
#         temp = line.split()
#         x_, y_, w_, h_ = eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])

#         x1, y1, x2, y2 = int((x_ - w_ / 2) * img.shape[1]), int((y_ - h_ / 2) * img.shape[0]), \
#                          int((x_ + w_ / 2) * img.shape[1]), int((y_ + h_ / 2) * img.shape[0])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))

# cv2.imshow('windows', img)
# cv2.waitKey(0)
import cv2
import xml.etree.ElementTree as ET

# 文件路径
img_path = r'D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\out_img\LLVIP_train_padding_00000.jpg'
xml_path = r'D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\out_xml\LLVIP_train_padding_00000.xml'

# 读取图像
img = cv2.imread(img_path)

# 解析 XML 文件
tree = ET.parse(xml_path)
root = tree.getroot()

# 遍历每个目标框
for obj in root.findall('object'):
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.find('xmin').text))
    ymin = int(float(bndbox.find('ymin').text))
    xmax = int(float(bndbox.find('xmax').text))
    ymax = int(float(bndbox.find('ymax').text))

    # 绘制矩形框
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

# 显示图像
# cv2.imshow('Image with Bounding Boxes', img)
cv2.imwrite(r"D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\test.png", img)
cv2.waitKey(0)
# cv2.destroyAllWindows()

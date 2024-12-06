"""
Joint small images to be a big image.
"""
import os
import argparse
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
# from bboxes2xml import bboxes2xml
# bboxes2xml.py file


def bboxes2xml(folder, img_name, width, height, gts, xml_save_to):
    """ Generate VOC type annotations from prediction results
    Args:
        img_name: raw image name without suffix;
        width, height: size of img;
        xml_save_to: path of xml will save to;
        gts: Ground True, [[label, xmin, ymin, xmax, ymax],...,[...]];
    """

    # write in xml file
    xml_file = open((xml_save_to + '/' + img_name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>' + folder + '</folder>\n')
    xml_file.write('    <filename>' + str(img_name) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for gt in gts:
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(gt[0]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(gt[1]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(gt[2]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(gt[3]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(gt[4]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    xml_file.close()


def list_dir(path, list_name, suffix='xml'):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name)
        else:
            if file_path.split('.')[-1] == suffix:
                file_path = file_path.replace('\\', '/')
                list_name.append(file_path)


def get_bboxes(xml_path):
    tree = ET.parse(open(xml_path, 'rb'))
    root = tree.getroot()
    bboxes, cls = [], []
    for obj in root.iter('object'):
        obj_cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
        cls.append(obj_cls)
    bboxes = np.asarray(bboxes, int)
    return bboxes, cls


def main(args):
    os.makedirs(args.new_data_path, exist_ok=True)
    imgs = []
    list_dir(args.raw_data_path, imgs, suffix='jpg')

    def img_save_name(i, zfill=5):
        file = args.filename + "_{}.jpg".format(str(i).zfill(zfill))
        return os.path.join(args.new_data_path, file)

    for i in range(len(imgs)):
        # 随机选四张图片, 用于对角拼接
        if len(imgs) < 4:
            print("The number of images is less than 4.")
            break
        img_joint = np.random.choice(imgs, 4, replace=False)
        imgs.remove(img_joint[0])
        imgs.remove(img_joint[1])
        imgs.remove(img_joint[2])
        imgs.remove(img_joint[3])
        img1 = Image.open(img_joint[0])
        img2 = Image.open(img_joint[1])
        img3 = Image.open(img_joint[2])
        img4 = Image.open(img_joint[3])
        bboxes1, cls1 = get_bboxes(img_joint[0].replace('.jpg', '.xml'))
        bboxes2, cls2 = get_bboxes(img_joint[1].replace('.jpg', '.xml'))
        bboxes3, cls3 = get_bboxes(img_joint[2].replace('.jpg', '.xml'))
        bboxes4, cls4 = get_bboxes(img_joint[3].replace('.jpg', '.xml'))

        # 大图宽高，及拼接中心点
        W = max(img1.size[0], img3.size[0]) + max(img2.size[0], img4.size[0])
        H = max(img1.size[1], img2.size[1]) + max(img3.size[1], img4.size[1])
        P = [0, 0]
        P[0] = max(img1.size[0], img3.size[0])
        P[1] = max(img1.size[1], img2.size[1])

        # 新建大图，并使用小图填充
        img_big = Image.new('RGB', (W, H))
        img_big.paste(img1, (P[0] - img1.size[0], P[1] - img1.size[1]))
        img_big.paste(img2, (P[0], P[1] - img2.size[1]))
        img_big.paste(img3, (P[0] - img3.size[0], P[1]))
        img_big.paste(img4, (P[0], P[1]))
        img_big.save(img_save_name(i))

        # 修改每个小图的Bbox，并合并所有子图的bbox和cls
        bbox_list = []
        if len(bboxes1) != 0:
            bboxes1[:, 0], bboxes1[:, 2] = bboxes1[:, 0] + P[0] - img1.size[0], bboxes1[:, 2] + P[0] - img1.size[0]
            bboxes1[:, 1], bboxes1[:, 3] = bboxes1[:, 1] + P[1] - img1.size[1], bboxes1[:, 3] + P[1] - img1.size[1]
            bbox_list.append(bboxes1)
        if len(bboxes2) != 0:
            bboxes2[:, 0], bboxes2[:, 2] = bboxes2[:, 0] + P[0], bboxes2[:, 2] + P[0]
            bboxes2[:, 1], bboxes2[:, 3] = bboxes2[:, 1] + P[1] - img2.size[1], bboxes2[:, 3] + P[1] - img2.size[1]
            bbox_list.append(bboxes2)
        if len(bboxes3) != 0:
            bboxes3[:, 0], bboxes3[:, 2] = bboxes3[:, 0] + P[0] - img3.size[0], bboxes3[:, 2] + P[0] - img3.size[0]
            bboxes3[:, 1], bboxes3[:, 3] = bboxes3[:, 1] + P[1], bboxes3[:, 3] + P[1]
            bbox_list.append(bboxes3)
        if len(bboxes4) != 0:
            bboxes4[:, 0], bboxes4[:, 2] = bboxes4[:, 0] + P[0], bboxes4[:, 2] + P[0]
            bboxes4[:, 1], bboxes4[:, 3] = bboxes4[:, 1] + P[1], bboxes4[:, 3] + P[1]
            bbox_list.append(bboxes4)

        if len(bbox_list) != 0:
            bboxes = np.vstack(tuple(bbox_list))
        cls = cls1 + cls2 + cls3 + cls4
        gts = [[c] + b.tolist() for c, b in zip(cls, bboxes)]
        bboxes2xml(folder=args.new_data_path.split('/')[-1],
                   img_name=img_save_name(i).replace('\\', '/').split('/')[-1].replace('.jpg', ''),
                   width=W, height=H,
                   gts=gts, xml_save_to=args.new_xml_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", default=r"D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\test_data", type=str,
                        help="raw dataset files")
    parser.add_argument("--new_data_path", default=r"D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\out_img", type=str,
                        help="images new path")
    parser.add_argument("--new_xml_path", default=r"D:\soft\git\weiweiboni2_project\github\labels\process\data\padding\out_xml", type=str,
                        help="xml new path")
    parser.add_argument("--filename", default="LLVIP_val", type=str, help="save name")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

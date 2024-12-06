# labels

目标检测数据集及标签制作  
![python](https://img.shields.io/badge/python-3.9+-yellow.svg)  ![opencv](https://img.shields.io/badge/opnecv-4.10.0-green.svg)  ![PyQt5](https://img.shields.io/badge/PyQt5-5.15.7-orange.svg)  ![torch](https://img.shields.io/badge/torch-2.2.0-blue.svg)

## 简介

- [x] 利用X-anylabeling进行数据集标注，生成xml格式和yolo格式的标签文件。
- [x] 制作自己的目标检测数据集，包含数据增强、数据扩充、数据集划分、标签转换等。

## 标注工具介绍

[X-anylabeling](https://github.com/CVHub520/X-AnyLabeling)  

- X-anylabeling是一款开源的图像标注工具，支持多种格式的标签文件。
- 将预训练权重转换为onnx格式，导入X-anylabeling模型，进行自动推理，实现半自动标注。

## 数据集扩充

1. 对数据集进行旋转、随机裁剪、加噪、随机遮挡等操作，对数据集进行扩充(标签为归一化之后的yolo类型)：

    ```python
    # 运行enhance.py脚本，对数据集进行扩充
    python enhance.py
    ```

2. 对数据集进行缩放和拼接，对数据集进行扩充(标签为xml类型)：

    ```python
    python splice.py
    python padding4.py
    ```

3. 对标签文件进行可视化：

    ```python
    # 检测框 id xmin ymin xmax ymax
    python show_txt_xml.py
    # 旋转检测框 id x1 y1 x2 y2 x3 y3 x4 y4
    python obb_labels_show.py
    ```

4. 将标签文件转换为yolo格式，并进行归一化(标签为xml类型)：

   ```python
   python xml2txt.py
   ```

5. 对结果输出的标签名进行更改，显示中文名称(或直接对模型权重中的标签名进行替换)：

    ```python
    python replace_labels.py
    ```

6. 对数据集进行划分：

    ```python
    python replace_labels.py
    ```

## 引用

>@ARTICLE{sun2020drone,
  title={Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3168279}
}
>
>@misc{X-AnyLabeling,
  year = {2023},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {Advanced Auto Labeling Solution with Added Features},
  howpublished = <https://github.com/CVHub520/X-AnyLabeling}}>
}
>

import json
import cv2 as cv
import os
from PIL import Image, ImageDraw
import numpy as np

jsons_path = './labels_json/'        # labelme生成的json文件夹
colors_path = '../colors.txt'  # 色彩配置文件
out_path = './annotations_train'     # 标签输出路径


# 读取json文件
def json_info(json_path):
    with open(json_path, 'rb') as f:
        img_info = json.load(f)
        return img_info


# 读取颜色配置文件
def colors_conf(colors_path):
    config = {}
    labels = []
    with open(colors_path, 'r') as f:
        for line in f.readlines():
            cur_line = line.strip().split(' ')
            labels.append(cur_line[0])
            config[cur_line[0]] = (int(cur_line[1]), int(cur_line[2]), int(cur_line[3]))
        return config, labels


# 生成标签
def gen_label(json_path):
    info = json_info(json_path)
    config, labels = colors_conf(colors_path)
    shape = (info['imageHeight'], info['imageWidth'], 3)
    mask = np.zeros(shape, dtype='uint8')
    shapes = info['shapes']
    name = os.path.splitext(info['imagePath'])[0]
    print(name)
    for s in shapes:
        label = s['label']
        l_index = labels.index(label)
        if s['shape_type'] == 'polygon':
            p = np.array([s['points']], dtype=np.int32)
            cv.polylines(mask, p, 1, (l_index, l_index, l_index))
            cv.fillPoly(mask, p, (l_index, l_index, l_index))
    mask = Image.fromarray(mask)
    l = mask.convert('L')    # 24位转8位灰度
    mask = Image.fromarray(np.uint8(l))
    p = mask.convert('P')
    out = os.path.join(out_path, f'{name}.png')
    p.save(out)


def labels():
    files = os.listdir(jsons_path)
    for file in files:
        ext = os.path.splitext(file)[1].strip('.')
        if ext != 'json':
            continue
        file = os.path.join(jsons_path, file)
        gen_label(file)


labels()

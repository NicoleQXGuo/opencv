import time

from keras_segmentation.predict import predict, predict_multiple
import os
import cv2 as cv
import random
from methods.pt import Pt
import numpy as np


# 读取颜色配置文件
def colors_conf(colors_path='colors.txt'):
    colors = []
    labels = []
    with open(colors_path, 'r') as f:
        for line in f.readlines():
            cur_line = line.strip().split(' ')
            labels.append(cur_line[0])
            bgr = (int(cur_line[3]), int(cur_line[2]), int(cur_line[1]))
            colors.append(bgr)
        return labels, colors


# 批量处理
def idcard_multiple(idcard_dir, out_dir):
    if idcard_dir == out_dir:
        raise NameError('输入和输出文件夹不可相同')
    print(f'开始生成预测蒙版')
    labels, colors = colors_conf()
    predict_multiple(
        inp_dir=idcard_dir,
        out_dir=out_dir,
        checkpoints_path='logs/vgg_unet_1',
        class_names=labels,
        colors=colors
    )
    ids = os.listdir(idcard_dir)
    index = 0
    for id in ids:
        index += 1
        print(f'共{len(ids)}个，当前处理第{index}个')
        img = os.path.join(idcard_dir, id)
        out_name = os.path.join(out_dir, id)
        origin = cv.imread(img)
        mask = cv.imread(out_name, 0)
        mask = cv.threshold(mask, 20, 255, cv.THRESH_BINARY)[1]
        pt = Pt()
        result = pt.perspective_transform(img=origin, mask=mask)
        cv.imwrite(out_name, result)


# 单个图片处理
def idcard_single(idcard_path, out_path):
    print(f'开始生成预测蒙版')
    tmp_path = os.path.splitext(idcard_path)
    rad = random.randint(1, 100)
    mask_path = tmp_path[0] + '_mask_' + str(rad) + tmp_path[1]
    labels, colors = colors_conf()
    # 只需要优化这一块predict
    predict(
        inp=idcard_path,
        out_fname=mask_path,
        checkpoints_path='logs/vgg_unet_1',
        class_names=labels,
        colors=colors
    )
    origin = cv.imread(idcard_path)
    mask = cv.imread(mask_path, 0)
    mask = cv.threshold(mask, 20, 255, cv.THRESH_BINARY)[1]
    pt = Pt()
    result = pt.perspective_transform(img=origin, mask=mask)
    dst_height, dst_width = result.shape[:2]
    if dst_height > dst_width:
        # 若高大于宽,逆时针旋转90°
        result = np.rot90(result)
    cv.imwrite(out_path, result)
    os.remove(mask_path)

import numpy as np
import cv2 as cv
from collections import Counter
import os
import shutil


# 判断是图片是否偏暗
def is_dark(gray_img, min=110, rate=0.6):
    w, h = gray_img.shape[:2]
    pixel_sum = w * h

    arr = np.array(gray_img)
    dark = np.maximum(arr, min)
    flatten = dark.flatten()
    count = Counter(flatten).get(min)
    dark_prop = count / pixel_sum
    # 灰度图亮度小于110的像素点占比超过60%，则判定图片偏暗
    if dark_prop >= rate:
        return True
    else:
        return False


def cv_show(img, scale=0.3):
    w, h = img.shape[:2]
    img = cv.resize(img, (int(h*scale), int(w*scale)))
    cv.imshow('demo', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 提升图片亮度
def lightness_multiple(idcard_dir):
    ids = os.listdir(idcard_dir)
    tmp = os.path.join(idcard_dir, 'tmp')
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.mkdir(tmp)
    for id in ids:
        if id == 'tmp':
            continue
        img = os.path.join(idcard_dir, id)
        out = os.path.join(idcard_dir, 'tmp', id)
        origin = cv.imread(img)
        gray = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
        dark = is_dark(gray, rate=0.55, min=120)
        if dark:
            h, w, c = origin.shape
            blank = np.ones([h, w, c], origin.dtype) * 255
            res = cv.addWeighted(origin, 0.8, blank, 0.2, 5)
            cv.imwrite(out, res)
        else:
            shutil.copy(img, tmp)


# 美化原始公式
# Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100
def blur(img):
    dark = is_dark(img, min=110, rate=0.4)
    if dark:
        v2 = 15
    else:
        v2 = 1
    v1 = 3
    dx = v1 * 5
    fc = v1 * 12.5
    p = 0.1

    tmp = cv.bilateralFilter(img, dx, fc, fc)
    tmp = cv.subtract(tmp, img)
    tmp = cv.add(tmp, (10, 10, 10, 128))
    tmp = cv.GaussianBlur(tmp, (2 * v2 - 1, 2 * v2 - 1), 0)
    tmp = cv.add(img, tmp)
    dst = cv.addWeighted(img, p, tmp, 1-p, 0)
    dst = cv.add(dst, (10, 10, 10, 255))
    return dst

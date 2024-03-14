import math

from methods.predict import idcard_multiple, idcard_single
import time
import cv2
import numpy as np

if __name__ == '__main__':
    # 批量处理
    # idcard_multiple(idcard_dir='./assets/pre1', out_dir='./assets/out1')
    # 单个处理
    idcard_single(idcard_path='./assets/explain/13.jpg', out_path='13_res.jpg')

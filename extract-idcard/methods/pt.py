import cv2 as cv
import numpy as np


class Pt:
    def __init__(self):
        self.second = False

    # 透视变换
    @staticmethod
    def __transform(img, dis_cnt):
        dis_cnt = dis_cnt.reshape(4, 2)
        (t_left, b_left, b_right, t_right) = dis_cnt
        t_width = np.sqrt((t_left[0] - t_right[0]) ** 2 + (t_left[1] - t_right[1]) ** 2)
        b_width = np.sqrt((b_left[0] - b_right[0]) ** 2 + (b_left[1] - b_right[1]) ** 2)
        des_width = int(sorted([t_width, b_width], reverse=True)[0])
        left_height = np.sqrt((t_left[0] - b_left[0]) ** 2 + (t_left[1] - b_left[1]) ** 2)
        right_height = np.sqrt((t_right[0] - b_right[0]) ** 2 + (t_right[1] - b_right[1]) ** 2)
        des_height = int(sorted([left_height, right_height], reverse=True)[0])
        src = np.float32(dis_cnt)
        dst = np.float32([[0, 0], [0, des_height], [des_width, des_height], [des_width, 0]])
        p_transform = cv.getPerspectiveTransform(src, dst)
        result = cv.warpPerspective(img, p_transform, (des_width, des_height))
        return result

    # 寻找用于透视变换的最小外接四边形
    def __perspective(self, img, mask, erode_iter=5, dilate_iter=10):
        kernel = np.ones((5, 5), np.uint8)
        erode = cv.erode(mask, kernel, iterations=erode_iter)
        dilate = cv.dilate(erode, kernel, iterations=dilate_iter)
        contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnt_five = sorted(contours, key=cv.contourArea, reverse=True)[:5]
        cnts = []
        for c in cnt_five:
            x, y, w, h = cv.boundingRect(c)
            if 1.0 < w / h < 1.8 or 1.0 < h / w < 1.8:
                cnts.append(c)
        if len(cnts) == 0:
            return img
        max = cnts[0]
        arc = cv.arcLength(max, True)
        approx = cv.approxPolyDP(max, 0.02 * arc, True)
        if len(approx) == 4:
            dis_cnt = approx
            result = Pt.__transform(img, dis_cnt)
            return result
        else:
            if self.second:
                return img
            else:
                self.second = True
                result = self.__perspective(img, mask, erode_iter=25, dilate_iter=31)
                return result

    def perspective_transform(self, img, mask):
        result = self.__perspective(img, mask)
        return result

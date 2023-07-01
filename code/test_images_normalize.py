# 导入所需模块
import os
import cv2
import numpy as np


colors = [[255, 182, 193], [128, 0, 128], [0, 0, 255], [135, 206, 235], [0, 128, 0],
          [255, 255, 0], [255, 215, 0], [255, 165, 0], [255, 0, 0], [128, 128, 128],
          [173, 255, 47], [0, 255, 255], [165, 42, 42], [192, 192, 192], [255, 0, 255],
          [255, 182, 193], [255, 20, 147], [0, 0, 139], [135, 206, 250], [0, 191, 255],
          [225, 255, 255], [0, 139, 139], [144, 238, 144], [0, 100, 0], [255, 140, 0]]

# 提取物品轮廓
k = 0
images_path = os.listdir('../data/test_change2/images')
for image_path in images_path:
    img = cv2.imread('../data/test_change2/images/'+image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度化
    # 二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 3333, 1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda i: i[0][0][0])
    img_rects = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            img_rect = img[y: y + h, x: x + w]
            img_rects.append(img_rect)
            
    # 改变图像形状与背景
    for i, img_rect in enumerate(img_rects):
        h, w, _ = img_rect.shape
        if h > w:
            img_rect = cv2.resize(img_rect, (int(w*458/h), 458)) #使用训练集图像高度均值
            if img_rect.shape[1] > 489:
                img_rect = cv2.resize(img_rect, (489, 458)) #压缩宽度
            elif img_rect.shape[1] == 489:
                img_rect = img_rect
            else:
                R = 489 - img_rect.shape[1]
                img_rect = cv2.copyMakeBorder(img_rect, 0, 0, int(R/2), R - int(R/2),
                                              cv2.BORDER_CONSTANT) #填充宽度
        else:
            img_rect = cv2.resize(img_rect, (489, int(h*489/w))) #使用训练集图像宽度均值
            if img_rect.shape[0] > 458:
                img_rect = cv2.resize(img_rect, (489, 458)) #压缩高度
            elif img_rect.shape[0] == 458:
                img_rect = img_rect
            else:
                R = 458 - img_rect.shape[0]
                img_rect = cv2.copyMakeBorder(img_rect, int(R/2), R - int(R/2),
                                              0, 0, cv2.BORDER_CONSTANT) #填充高度
        for m in range(img_rect.shape[0]):
            for n in range(img_rect.shape[1]):
                if all(img_rect[m, n] == [0, 0, 0]) or all(img_rect[m, n] == [255, 255, 255]):
                    img_rect[m, n] = colors[k]
        if i == 0:
            img = img_rect
        else:
            img = np.hstack((img, img_rect))
        k += 1
    cv2.imwrite('../data/test/images/'+image_path, img)
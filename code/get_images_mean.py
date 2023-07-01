# 导入所需模块
import os
import cv2
import numpy as np


# 提取物品轮廓
images_path = os.listdir('../data/train_first/images')
# 读取图片，查看训练集图片大小均值
width = []
height = []
for image_path in images_path:
    img = cv2.imdecode(np.fromfile('../data/train_first/images/'+image_path,
                                   dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    width.append(w)
    height.append(h)
print(np.mean(width))
print(np.mean(height))
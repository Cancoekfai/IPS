# 导入所需模块
import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split


# 读取图片名
images = os.listdir('../data/all/images')
names = list(map(lambda i: i.rstrip('.jpg'), images))
# 读取标签字典
labels_first_dict = np.load('../data/all/labels/labels_first_dict.npy',
                            allow_pickle=True).item()
labels_second_dict = np.load('../data/all/labels/labels_second_dict.npy',
                             allow_pickle=True).item()
labels_first_dict = {v: k for k, v in labels_first_dict.items()}
labels_second_dict = {v: k for k, v in labels_second_dict.items()}

# 随机打乱
train_names, val_names = train_test_split(names, test_size=0.2, random_state=0)
train_names.append(train_names[0])
val_names.append(val_names[0])

def cv2ImgAddText(img, text, left, top, textColor, textSize):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype('simfang.ttf', textSize, encoding='utf-8')
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 训练集
k = 0
index = 0
num = random.randint(2, 3)
while k < len(train_names):
    try:
        with open('../data/train_combine1/labels/%s.txt'%index, 'w') as f1:
            with open('../data/train_combine2/labels/%s.txt'%index, 'w') as f2:
                labels = []
                for j in range(num):
                    # 组合图片
                    if j == 0:
                        img1 = cv2.imdecode(np.fromfile('../data/all/images/'+train_names[k]+'.jpg',
                                                        dtype=np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.resize(img1, (489, 458))
                    else:
                        img1 = cv2.imdecode(np.fromfile('../data/all/images/'+train_names[k]+'.jpg',
                                                        dtype=np.uint8), cv2.IMREAD_COLOR)
                        imgi = cv2.resize(img1, (489, 458))
                        img = np.hstack((img, imgi))
                        
                    # 读取标签信息
                    df_first = pd.read_table('../data/all/labels/TXT/'+train_names[k]+'_first.txt',
                                             header=None, sep=' ')
                    df_second = pd.read_table('../data/all/labels/TXT/'+train_names[k]+'_second.txt',
                                              header=None, sep=' ')
                    # 改变位置信息
                    w_ratio = img1.shape[1] / 489
                    h_ratio = img1.shape[0] / 458
                    x_min = (np.ceil(df_first.iloc[0, 1] / w_ratio) + j*489).astype('int')
                    x_max = (np.floor(df_first.iloc[0, 3] / w_ratio) + j*489).astype('int')
                    y_min = (np.ceil(df_first.iloc[0, 2] / h_ratio)).astype('int')
                    y_max = (np.floor(df_first.iloc[0, 4] / h_ratio)).astype('int')
                    # 标签信息标准化
                    if x_max < x_min:
                        x_min = x_max
                    if y_max < y_min:
                        y_min = y_max
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = (x_max - 1/2*width) / (489*num)
                    y_center = (y_max - 1/2*height) / 458
                    width = width / (489*num)
                    height = height / 458
                    # 写入标签信息
                    label_first = df_first.iloc[0, 0]
                    label_second = df_second.iloc[0, 0]
                    labels.append([label_first, label_second, x_min, y_min, x_max, y_max])
                    f1.write(str(label_first)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+\
                             ' '+str(height)+'\n')
                    f2.write(str(label_second)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+\
                             ' '+str(height)+'\n')
                        
                    k += 1
                    
        # 绘制框
        img_box = img.copy()
        h, w, _ = img_box.shape
        bbox_thick = int(0.6 * (h + w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        for i in labels:
            cv2.rectangle(img_box, (i[2], i[3]), (i[4], i[5]), (255, 0, 0), bbox_thick*2)
            # 获取文本大小
            label = labels_first_dict[i[0]] + '-' + labels_second_dict[i[1]]
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, thickness=bbox_thick)
            # 将文字放入框中
            if i[3] < 17:
                i[3] = 17
            # 绘制文本框
            cv2.rectangle(img_box, (i[2], i[3]), (i[2]+int(0.75*text_width), i[3]-2*text_height),
                          (255, 0, 0), thickness=cv2.FILLED)
            img_box = cv2ImgAddText(img_box, label, i[2], i[3]-17, (255, 255, 255), 17)
            
        # 写入图片
        cv2.imwrite('../data/train_combine1/images/%s.jpg'%index, img)
        cv2.imwrite('../data/train_combine2/images/%s.jpg'%index, img)
        # 写入带框图片
        cv2.imwrite('../data/train_combine_box/images/%s.jpg'%index, img_box)
        
        index += 1
        num = random.randint(2, 3)
        
    except:
        print('训练集图像组合完成')
        
        
# 验证集
k = 0
index = 0
num = random.randint(2, 3)
while k < len(val_names):
    try:
        with open('../data/val_combine1/labels/%s.txt'%index, 'w') as f1:
            with open('../data/val_combine2/labels/%s.txt'%index, 'w') as f2:
                labels = []
                for j in range(num):
                    # 组合图片
                    if j == 0:
                        img1 = cv2.imdecode(np.fromfile('../data/all/images/'+val_names[k]+'.jpg',
                                                        dtype=np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.resize(img1, (489, 458))
                    else:
                        img1 = cv2.imdecode(np.fromfile('../data/all/images/'+val_names[k]+'.jpg',
                                                        dtype=np.uint8), cv2.IMREAD_COLOR)
                        imgi = cv2.resize(img1, (489, 458))
                        img = np.hstack((img, imgi))
                        
                    # 读取标签信息
                    df_first = pd.read_table('../data/all/labels/TXT/'+val_names[k]+'_first.txt',
                                             header=None, sep=' ')
                    df_second = pd.read_table('../data/all/labels/TXT/'+val_names[k]+'_second.txt',
                                              header=None, sep=' ')
                    # 改变位置信息
                    w_ratio = img1.shape[1] / 489
                    h_ratio = img1.shape[0] / 458
                    x_min = (np.ceil(df_first.iloc[0, 1] / w_ratio) + j*489).astype('int')
                    x_max = (np.floor(df_first.iloc[0, 3] / w_ratio) + j*489).astype('int')
                    y_min = (np.ceil(df_first.iloc[0, 2] / h_ratio)).astype('int')
                    y_max = (np.floor(df_first.iloc[0, 4] / h_ratio)).astype('int')
                    # 标签信息标准化
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = (x_max - 1/2*width) / (489*num)
                    y_center = (y_max - 1/2*height) / 458
                    width = width / (489*num)
                    height = height / 458
                    # 写入标签信息
                    label_first = df_first.iloc[0, 0]
                    label_second = df_second.iloc[0, 0]
                    labels.append([label_first, label_second, x_min, y_min, x_max, y_max])
                    f1.write(str(label_first)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+\
                             ' '+str(height)+'\n')
                    f2.write(str(label_second)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+\
                             ' '+str(height)+'\n')
                        
                    k += 1
                    
        # 绘制框
        img_box = img.copy()
        h, w, _ = img_box.shape
        bbox_thick = int(0.6 * (h + w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        for i in labels:
            cv2.rectangle(img_box, (i[2], i[3]), (i[4], i[5]), (255, 0, 0), bbox_thick*2)
            # 获取文本大小
            label = labels_first_dict[i[0]] + '-' + labels_second_dict[i[1]]
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, thickness=bbox_thick)
            # 将文字放入框中
            if i[3] < 17:
                i[3] = 17
            # 绘制文本框
            cv2.rectangle(img_box, (i[2], i[3]), (i[2]+int(0.75*text_width), i[3]-2*text_height),
                          (255, 0, 0), thickness=cv2.FILLED)
            img_box = cv2ImgAddText(img_box, label, i[2], i[3]-17, (255, 255, 255), 17)
            
        # 写入图片
        cv2.imwrite('../data/val_combine1/images/%s.jpg'%index, img)
        cv2.imwrite('../data/val_combine2/images/%s.jpg'%index, img)
        # 写入带框图片
        cv2.imwrite('../data/val_combine_box/images/%s.jpg'%index, img_box)
        
        index += 1
        num = random.randint(2, 3)
        
    except:
        print('验证集图像组合完成')
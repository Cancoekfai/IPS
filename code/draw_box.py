# 导入所需模块
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


txts = os.listdir('../data/all/labels/TXT')
txts_first = list(filter(lambda i: 'first' in i, txts))
labels_first_dict = np.load('../data/all/labels/labels_first_dict.npy',
                            allow_pickle=True).item()
labels_second_dict = np.load('../data/all/labels/labels_second_dict.npy',
                             allow_pickle=True).item()
labels_first_dict = {v: k for k, v in labels_first_dict.items()}
labels_second_dict = {v: k for k, v in labels_second_dict.items()}

def cv2ImgAddText(img, text, left, top, textColor, textSize):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype('simfang.ttf', textSize, encoding='utf-8')
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

k = 0
for txt in txts_first:
    try:
        df_first = pd.read_table('../data/all/labels/TXT/'+txt, header=None, sep=' ')
        df_second = pd.read_table('../data/all/labels/TXT/'+'_second'.join(txt.split('_first')),
                                  header=None, sep=' ')
        img = cv2.imdecode(np.fromfile('../data/all/images/'+txt.rstrip('_first.txt')+'.jpg',
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        
        # 获取位置
        x1 = df_first.loc[0, 1]
        y1 = df_first.loc[0, 2]
        x2 = df_first.loc[0, 3]
        y2 = df_first.loc[0, 4]
        
        # 绘制框
        bbox_thick = int(0.6 * (h + w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), bbox_thick*2)
        
        # 获取文本大小
        label = labels_first_dict[df_first.loc[0, 0]] + '-' + labels_second_dict[df_second.loc[0, 0]]
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                              0.6, thickness=bbox_thick)
        # 绘制文本框
        cv2.rectangle(img, (x1, y1), (x1+int(0.75*text_width), y1-2*text_height), (255, 0, 0), thickness=cv2.FILLED)
        
        # 将文字放入框中
        if y1 < 17:
            y1 = 17
        img = cv2ImgAddText(img, label, x1, y1-17, (255, 255, 255), 17)
        cv2.imencode('.jpg', img)[1].tofile('../data/all/images_box/'+txt.rstrip('_first.txt')+'.jpg')
    except:
        k += 1
        os.remove('../data/all/images/'+txt.rstrip('_first.txt')+'.jpg')
        os.remove('../data/all/labels/TXT/'+txt)
        os.remove('../data/all/labels/TXT/'+'_second'.join(txt.split('_first')))
        print('读取图片失败')
print('共删除%s张图片'%k) #2
#%% 进行数据标准化
# 导入所需模块
import os
import cv2
import numpy as np
import pandas as pd


txts = os.listdir('../data/all/labels/TXT')
txts_first = list(filter(lambda i: 'first' in i, txts))

# 转换为`YOLOv5`格式的标注信息
for txt in txts_first:
    df_first = pd.read_table('../data/all/labels/TXT/'+txt, header=None, sep=' ')
    df_second = pd.read_table('../data/all/labels/TXT/'+'_second'.join(txt.split('_first')),
                              header=None, sep=' ')
    img = cv2.imdecode(np.fromfile('../data/all/images/'+txt.rstrip('_first.txt')+'.jpg',
                                   dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # 获取信息
    label_first = df_first.loc[0, 0]
    label_second = df_second.loc[0, 0]
    x1 = df_first.loc[0, 1]
    y1 = df_first.loc[0, 2]
    x2 = df_first.loc[0, 3]
    y2 = df_first.loc[0, 4]
    width = df_first.loc[0, 5]
    height = df_first.loc[0, 6]
    
    x_center = (x2 - 1/2*width) / w
    y_center = (y2 - 1/2*height) / h
    width = width / w
    height = height / h
    
    with open('../data/all/labels_first/'+txt.rstrip('_first.txt')+'.txt', 'w') as f:
        f.write(str(label_first)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+' '+str(height))
    with open('../data/all/labels_second/'+txt.rstrip('_first.txt')+'.txt', 'w') as f:
        f.write(str(label_second)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+' '+str(height))
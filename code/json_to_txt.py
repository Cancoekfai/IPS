#%% 将标注信息由`JSON`文件转为`TXT`文本文件，并进行数据合理化及规范化
# 导入所需模块
import os
import re
import json
import shutil
import numpy as np


labels_first = set()
labels_second = set()
jsons = os.listdir('../data/all/labels/JSON')
for js in jsons:
    try:
        # 读取`json`文件
        with open('../data/all/labels/JSON/'+js, encoding='utf-8') as f:
            json_data = json.load(f)
        label_first = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[0].strip(' ')
        label_second = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[1]
        if label_first == '有害物质':
            label_first = '有害垃圾'
        labels_first.add(label_first)
        labels_second.add(label_second)
    except:
        print('缺少标注信息')
        
labels_first_dict = dict(zip(labels_first, range(len(labels_first))))
labels_second_dict = dict(zip(labels_second, range(len(labels_second))))
# 保存字典
np.save('../data/all/labels/labels_first_dict.npy', labels_first_dict)
np.save('../data/all/labels/labels_second_dict.npy', labels_second_dict)

k = 0
# 将标注信息写入`.txt`文本文件
for js in jsons:
    try:
        # 读取`json`文件
        with open('../data/all/labels/JSON/'+js, encoding='utf-8') as f:
            json_data = json.load(f)
        bndbox = json_data['outputs']['object'][0]['bndbox']
        if bndbox['xmin'] < 0:
            bndbox['xmin'] = 0
        if bndbox['xmin'] > json_data['size']['width']:
            bndbox['xmin'] = json_data['size']['width']
        if bndbox['ymin'] < 0:
            bndbox['ymin'] = 0
        if bndbox['ymin'] > json_data['size']['height']:
            bndbox['ymin'] = json_data['size']['height']
        if bndbox['xmax'] < 0:
            bndbox['xmax'] = json_data['size']['width']
        if bndbox['ymax'] < 0:
            bndbox['ymax'] = json_data['size']['height']
        if bndbox['xmax'] > json_data['size']['width']:
            bndbox['xmax'] = json_data['size']['width']
        if bndbox['ymax'] > json_data['size']['height']:
            bndbox['ymax'] = json_data['size']['height']
        label_first = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[0].strip(' ')
        label_second = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[1]
        if label_first == '有害物质':
            label_first = '有害垃圾'
        with open('../data/all/labels/TXT/'+js.rstrip('.json')+'_first.txt', 'w') as f:
            f.write(str(labels_first_dict[label_first])+' '+str(bndbox['xmin'])+' '+str(bndbox['ymin'])+\
                    ' '+str(bndbox['xmax'])+' '+str(bndbox['ymax'])+' '+str(bndbox['xmax']-bndbox['xmin'])+\
                    ' '+str(bndbox['ymax']-bndbox['ymin']))
        with open('../data/all/labels/TXT/'+js.rstrip('.json')+'_second.txt', 'w') as f:
            f.write(str(labels_second_dict[label_second])+' '+str(bndbox['xmin'])+' '+str(bndbox['ymin'])+\
                    ' '+str(bndbox['xmax'])+' '+str(bndbox['ymax'])+' '+str(bndbox['xmax']-bndbox['xmin'])+\
                    ' '+str(bndbox['ymax']-bndbox['ymin']))
        # 复制图片
        if os.path.exists('../data/all/images_original/'+js.rstrip('.json')+'.jpg'):
            shutil.copy('../data/all/images_original/'+js.rstrip('.json')+'.jpg',
                        '../data/all/images/'+js.rstrip('.json')+'.jpg')
        if os.path.exists('../data/all/images_original/'+js.rstrip('.json')+'.jpeg'):
            shutil.copy('../data/all/images_original/'+js.rstrip('.json')+'.jpeg',
                        '../data/all/images/'+js.rstrip('.json')+'.jpg')
    except:
        k += 1
        print('缺少标注信息')
print('共删除%s张图片'%k) #共删除`28`张图片
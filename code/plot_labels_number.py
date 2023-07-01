# 导入所需模块
import os
import re
import json
import shutil
import numpy as np
import pandas as pd
from collections import Counter

labels_first = []
labels_second = []
jsons = os.listdir('../data/all/labels/JSON')
for js in jsons:
    try:
        # 读取json文件
        with open('../data/all/labels/JSON/'+js, encoding='utf-8') as f:
            json_data = json.load(f)
        label_first = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[0].strip(' ')
        label_second = re.split(r'-|/|=', json_data['outputs']['object'][0]['name'])[1]
        if label_first == '有害物质':
            label_first = '有害垃圾'
        labels_first.append(label_first)
        labels_second.append(label_second)
    except:
        print('缺少标注信息')
        
labels_first_dict = dict(Counter(labels_first))
labels_second_dict = dict(Counter(labels_second))
df_first = pd.DataFrame({'类别名称': labels_first_dict.keys(),
                         '数量': labels_first_dict.values()})
df_first.to_excel('../article/figures/plot_first_labels_number.xlsx', index=False)
df_second = pd.DataFrame({'类别名称': labels_second_dict.keys(),
                          '数量': labels_second_dict.values()})
df_second.to_excel('../article/figures/plot_second_labels_number.xlsx', index=False)
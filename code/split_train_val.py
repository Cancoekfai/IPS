# 导入所需模块
import os
import shutil
from sklearn.model_selection import train_test_split


images = os.listdir('../data/all/images')
images_name = list(map(lambda i: i.rstrip('.jpeg'), images))

train, val = train_test_split(images_name, test_size=0.2, random_state=0)

# first
for i in train:
    shutil.copy('../data/all/images/'+i+'.jpg', '../data/train_first/images/'+i+'.jpg')
    shutil.copy('../data/all/labels_first/'+i+'.txt', '../data/train_first/labels/'+i+'.txt')
for i in val:
    shutil.copy('../data/all/images/'+i+'.jpg', '../data/val_first/images/'+i+'.jpg')
    shutil.copy('../data/all/labels_first/'+i+'.txt', '../data/val_first/labels/'+i+'.txt')
# second
for i in train:
    shutil.copy('../data/all/images/'+i+'.jpg', '../data/train_second/images/'+i+'.jpg')
    shutil.copy('../data/all/labels_second/'+i+'.txt', '../data/train_second/labels/'+i+'.txt')
for i in val:
    shutil.copy('../data/all/images/'+i+'.jpg', '../data/val_second/images/'+i+'.jpg')
    shutil.copy('../data/all/labels_second/'+i+'.txt', '../data/val_second/labels/'+i+'.txt')
# 导入所需模块
import os
import numpy as np
import pandas as pd
from collections import Counter


files = os.listdir('../data/test_first_finally/labels')

labels = np.array([])
for file in files:
    label = pd.read_table('../data/test_first_finally/labels/'+file, header=None, sep=' ')[0]
    labels = np.append(labels, label)
Counter(labels)
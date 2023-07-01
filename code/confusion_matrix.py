# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:13:09 2021

@author: Zhuohui Chen
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman'] #允许仿宋字体绘图


def plot_confusion_matrix(cm, x_classes, y_classes=None, normalize=False, cmap=plt.cm.Blues):
    plt.figure(dpi=1000)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(x_classes))
    if y_classes == None:
        plt.xticks(tick_marks, x_classes, fontsize=8)
        plt.yticks(tick_marks, x_classes, fontsize=8)
    else:
        plt.xticks(tick_marks, x_classes, fontsize=8)
        plt.yticks(tick_marks, y_classes, fontsize=8)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=12,
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('Real value', fontsize=12)
    plt.xlabel('Predictive value', fontsize=12)
    
cm = np.array([[0.25, 0.12, 0, 0, 0.12], [0, 0.25, 0, 0, 0.25],
               [0.75, 0.25, 0.6, 0.37, 0.62], [0, 0, 0.2, 0.12, 0],
               [0, 0.37, 0.2, 0.5, 0]])
plot_confusion_matrix(cm, ['Harmful waste', 'Kitchen waste', 'Recyclables', 'Other waste', 'background FP'],
                      y_classes=['Harmful waste', 'Kitchen waste', 'Recyclables', 'Other waste', 'background FN'])
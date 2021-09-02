# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:06:07 2021

@author: cocob
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

fileList = [
'D:\\ShareDIR\\abla_dataset\\loss_tri01_w01\\events.out.tfevents.1623612130.wei-Linux', 
'D:\\ShareDIR\\abla_dataset\\dataset_dp\\events.out.tfevents.1624181469.wei-Linux', 
'D:\\ShareDIR\\abla_dataset\\dataset_g_dp\\events.out.tfevents.1624204021.wei-Linux', 
'D:\\ShareDIR\\abla_dataset\\dataset_s_dp\\events.out.tfevents.1624227813.wei-Linux', 
'D:\\ShareDIR\\abla_dataset\\dataset_gs_dp\\events.out.tfevents.1624250367.wei-Linux'
]

nameList = [
'loss_tri01_w01', 
'loss_tri01_w01_dp', 
'g_dp', 
's_dp', 
'gs_dp'
]

eventList = []

for filePath in fileList:
    ls = []
    tfBoardFile = tf.train.summary_iterator(filePath)
    for summary in tfBoardFile:
        for content in summary.summary.value:
            if (content.tag != 'train/loss') : continue
            ls.append(content.simple_value)
    print('size:', len(ls))
    eventList.append(ls)

x = 0
for event in eventList:
    x = len(event) if len(event) > x else x
    
x = np.arange(len(ls))

fig, ax = plt.subplots()

for i, ls in enumerate(eventList):
    ax.plot(x, ls, label = nameList[i])

ax.set_ylim([0, 0.5])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
fig.tight_layout()
plt.show()
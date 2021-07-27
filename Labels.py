# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 00:00:47 2021

@author: cocob
"""

sunrgbd = {
    0 : 'bed', 
    1 : 'table', 
    2 : 'sofa', 
    3 : 'chair', 
    4 : 'toilet', 
    5 : 'desk', 
    6 : 'dresser', 
    7 : 'night_stand', 
    8 : 'bookshelf', 
    9 : 'bathtub'}

scannet = {
    0 : 'cabinet', 
    1 : 'bed', 
    2 : 'chair', 
    3 : 'sofa', 
    4 : 'table', 
    5 : 'door', 
    6 : 'window', 
    7 : 'bookshelf', 
    8 : 'picture', 
    9 : 'counter', 
    10 : 'desk', 
    11 : 'curtain', 
    12 : 'refridgerator', 
    13 : 'shower curtain', 
    14 : 'toilet', 
    15 : 'sink', 
    16 : 'bathtub', 
    17 : 'garbage bin'}

# Abandoned categories
scannet[5] = ''
scannet[6] = ''
scannet[8] = ''

nyu2ModelNet40 = dict()
nyu2ModelNet40[''] = ''

import csv

with open('utils/scannetv2-labels.combined.tsv', 'r', encoding = 'utf-8') as fr:
    csvReader = csv.reader(fr, delimiter='\t')
    for i, row in enumerate(csvReader):
        if (i == 0) : continue
        nyu2ModelNet40[row[6]] = row[9]
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 04:01:05 2021

@author: cocob
"""


import os
import copy as cp
import numpy as np
import open3d as o3d
from operator import itemgetter

import torch

from Parsers import Parser_ModelSelector, initDevice
from Utils import SaveDict, ReadDict, GetModelByName, GetUnitModel


def GetObjectData(FILE_DIR : str = 'objects', dictName = 'objects.pkl', ext = '.pcd'):
    objectsDict = ReadDict(os.path.join(FILE_DIR, dictName))
    for objName in objectsDict:
        pcd = o3d.io.read_point_cloud(os.path.join(FILE_DIR, objName + ext))
        objectsDict[objName]['obj'] = pcd
    return objectsDict


def GetObjectTemplate(objDict, catDict, args):
    from Module_PointNetSeries import PointNet2Comp2
    net = PointNet2Comp2(0, 40, 'cls')
    net.to(args.device)
    net.load_state_dict(torch.load(args.modelPath, map_location=args.device))
    net.eval()
    
    for objName in objDict:
        label = objDict[objName]['label']
        if (not label):
            objDict[objName]['template'] = None
            continue
        
        pcd = objDict[objName]['obj']
        pcd = GetUnitModel(pcd)
        
        pts = np.asarray(cp.deepcopy(pcd.points)).astype('float32')
        pts = torch.tensor(pts).view(1, -1, 3)
        if (args.cuda) : pts = pts.cuda()
        _, feat = net(pts)
        if (args.cuda): feat = feat.cpu()
        feat = feat.detach().numpy().squeeze()
        
        modelNameList = []
        modelFeatList = []
        for modelName in catDict[label]:
            modelNameList.append(modelName)
            modelFeatList.append(catDict[label][modelName])
        
        modelFeats = np.asarray(modelFeatList)# N x 1024
        feat = np.tile(feat, (len(modelFeatList), 1))
        
        dists = np.mean(np.abs(feat - modelFeats), axis=1)
        
        ranking = [[modelName, dist] for modelName, dist in zip(modelNameList, dists)]
        ranking = sorted(ranking, key=itemgetter(1))
        
        objRankDict = dict()
        objRankDict['rank1'] = ranking[0][0]
        objRankDict['rank2'] = ranking[1][0]
        objRankDict['rank3'] = ranking[2][0]
        
        objDict[objName]['template'] = objRankDict
        
    return objDict


def DelObjectPC(objDict):
    for objName in objDict : objDict[objName].pop('obj', None)
    
    return objDict


def ShowTemplate(objDict, modelBasePath, specIdList = []):
    rowCnt = 0
    objCnt = 0
    showList = []
    for objName in objDict:
        if (specIdList and (objCnt not in specIdList)):
            objCnt += 1
            continue
        pcd = objDict[objName]['obj']
        pcd = GetUnitModel(pcd).translate([0, rowCnt, 0])
        if (objDict[objName]['template']):
            tmp1 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank1'], 
                                  modelBasePath, 'mesh')
            tmp2 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank2'], 
                                  modelBasePath, 'mesh')
            tmp3 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank3'], 
                                  modelBasePath, 'mesh')
            tmp1 = GetUnitModel(tmp1).translate([1, rowCnt, 0])
            tmp2 = GetUnitModel(tmp2).translate([2, rowCnt, 0])
            tmp3 = GetUnitModel(tmp3).translate([3, rowCnt, 0])
            showList.extend([tmp1, tmp2, tmp3])
        showList.append(pcd)
        rowCnt += 1
        objCnt += 1
    o3d.visualization.draw_geometries([*showList], mesh_show_wireframe=True)


def ShowAllTemplates(objDict, modelBasePath):
    rowCnt = 0
    showList = []
    for objName in objDict:
        pcd = objDict[objName]['obj']
        pcd = GetUnitModel(pcd).translate([0, rowCnt, 0])
        if (objDict[objName]['template']):
            tmp1 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank1'], 
                                  modelBasePath, 'mesh')
            tmp2 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank2'], 
                                  modelBasePath, 'mesh')
            tmp3 = GetModelByName(objDict[objName]['label'], 
                                  objDict[objName]['template']['rank3'], 
                                  modelBasePath, 'mesh')
            tmp1 = GetUnitModel(tmp1).translate([1, rowCnt, 0])
            tmp2 = GetUnitModel(tmp2).translate([2, rowCnt, 0])
            tmp3 = GetUnitModel(tmp3).translate([3, rowCnt, 0])
            showList.extend([tmp1, tmp2, tmp3])
        showList.append(pcd)
        rowCnt += 1
    o3d.visualization.draw_geometries([*showList], mesh_show_wireframe=True)


if __name__ == '__main__':
    args = Parser_ModelSelector()
    args = initDevice(args)
    OBJECT_DIR = args.objectDIR
    MODEL_FEAT_PATH = args.modelFeature
    if (not args.test):
        objDict = GetObjectData(OBJECT_DIR)
        catDict = ReadDict(MODEL_FEAT_PATH)
        objDict = GetObjectTemplate(objDict, catDict, args)
        objDict = DelObjectPC(objDict)
        SaveDict(os.path.join(args.objectDIR, 'templates.pkl'), objDict)
    else:
        objDict = GetObjectData(OBJECT_DIR, args.template)
        ShowAllTemplates(objDict, args.modelBasePath)
        # ShowTemplate(objDict, args.modelBasePath, [])

    
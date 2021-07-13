# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:04:30 2021

@author: cocob
"""

import os
import copy as cp
import numpy as np
import open3d as o3d
import pickle as pkl


class ObjectProp:
    def __init__(self, objName = None, objOBB = None, label = None, tmpName = None, rigid = None):
        self.objName = objName
        self.obb = objOBB# Dict{'center', 'extent', 'R'}
        self.label = label
        self.tmpName = tmpName
        self.rigid = rigid# Dict{'R', 'T'}
        
        self.obj = None
        self.tmp = None
        self.obj_unit_scale = None
        self.tmp_unit_scale = None


#############################################################
#               Dictionary File Processing
#############################################################

def SaveDict(FILE_NAME, obj):
    with open(FILE_NAME, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def ReadDict(FILE_NAME):
    with open(FILE_NAME, 'rb') as f:
        return pkl.load(f)


#############################################################
#               ModelNet40 Dataset Implementation
#############################################################

def WalkModelNet40ByCatName(DIR_PATH : str, CAT_PATH : str, extName : str = '.off', retFile : str = 'path'):
    assert retFile == 'all' or retFile == 'path' or retFile == 'name'
    filePathList = []
    fileNameList = []
    for dirpath, dirnames, filename in os.walk(os.path.join(DIR_PATH, CAT_PATH)):
        for modelName in filename:
            if (modelName[-len(extName):] == extName):
                filePathList.append(os.path.join(dirpath, modelName))
                fileNameList.append(modelName)
    if (retFile == 'all') : return filePathList, fileNameList
    if (retFile == 'path') : return filePathList
    if (retFile == 'name') : return fileNameList


def WalkModelNet40CatDIR(DIR_PATH : str):
    catList = []
    for dirpath, dirnames, filename in os.walk(DIR_PATH):
        for catdir in dirnames:
            catList.append(catdir)
        break
    return catList



def GetModelByName(modelCat : str, modelName : str, ModelBase_DIR : str, retType : str = 'mesh'):
    assert retType == 'mesh' or retType == 'pcd', "retType error ('mesh', 'pcd' or 'all')"
    meshF = False
    pcdF = False
    if (retType == 'mesh' or retType == 'all') : meshF = True
    if (retType == 'pcd' or retType == 'all') : pcdF = True
    
    if (meshF) : meshPath = os.path.join(ModelBase_DIR, 'mesh', modelCat, modelName + '.off')
    if (pcdF) : pcdPath = os.path.join(ModelBase_DIR, 'pcd', modelCat, modelName + '.pcd')
    if (meshF) : mesh = o3d.io.read_triangle_mesh(meshPath)
    if (pcdF) : pcd = o3d.io.read_point_cloud(pcdPath)
    
    if (meshF and pcdF) : return mesh, pcd
    elif (meshF) : return mesh
    elif (pcdF) : return pcd
    else : raise 'Unexpected error'


def BatchReadPoindCloud(DIR_PATH, pattern = 'vote_cluster', EXP = '.pcd', uniColor = None):
    assert uniColor == None or type(uniColor) == type(list()), 'Color must formed as [R, G, B] in list'
    PCDList = []
    pathList = []
    for dirpath, dirnames, filename in os.walk(DIR_PATH):
        for file in filename:
            if (pattern in file and file[-4:] == EXP):
                pathList.append(os.path.join(dirpath, file))
        break
    pathList = sorted(pathList)
    for _path in pathList:
        pcd = o3d.io.read_point_cloud(_path)
        if (uniColor) : pcd.paint_uniform_color(uniColor)
        PCDList.append(pcd)
    return PCDList


def GetUnitModel(model, deepCopy = True, retTrans = False):
    if (deepCopy) : model = cp.deepcopy(model)
    maxBound = model.get_max_bound()
    minBound = model.get_min_bound()
    length = np.linalg.norm(maxBound - minBound, 2)
    model.scale(1 / length, center = model.get_center())
    trans = -model.get_center()
    model.translate(trans)
    if (retTrans) : return model, 1 / length, trans# model, scale, translate
    return model

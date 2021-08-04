# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 02:45:46 2021

@author: cocob
"""


import os
import copy as cp
import numpy as np
import open3d as o3d

from Utils import SaveDict, ReadDict, DrawAxis


def GetSceneSemanticInfo(SCENE_DIR : str, outName : str = 'sceneLabel.pkl'):
    print('Source directory: %s' %SCENE_DIR)
    print("Reading mesh_semantic.ply...")
    
    from plyfile import PlyData
    file_in = PlyData.read(os.path.join(SCENE_DIR, 'habitat\\mesh_semantic.ply'))
    vertices_in = file_in.elements[0]
    faces_in = file_in.elements[1]
    pointSize = len(vertices_in)
    print("Filtering data...")
    objects = {}
    for f in faces_in:
        object_id = f[1]
        if (not object_id in objects) : objects[object_id] = []
        objects[object_id].extend(f[0])# Vertices
    
    import json
    with open(os.path.join(SCENE_DIR, 'habitat\\info_semantic.json')) as f:
        jFile = json.load(f)
        print("Reading info_semantic.json...")
        classes = jFile['classes']
        IDToClassDict = dict()
        classToIDDict = dict()
        for _class in classes:
            IDToClassDict[_class['id']] = _class['name']
            classToIDDict[_class['name']] = _class['id']
        objList = jFile['objects']
        objIDToLabelDict = dict()
        for obj in objList : objIDToLabelDict[obj['id']] = obj['class_id']
    
    print("making sceneLabel.pkl...")
    insPCD = [-1 for i in range(pointSize)]
    semPCD = [-1 for i in range(pointSize)]
    validObjCnt = len(objects)
    for objID in objects:
        objects[objID] = list(set(objects[objID]))
        try:
            label = objIDToLabelDict[objID]
            for pcdIdx in objects[objID]:
                semPCD[pcdIdx] = label
                insPCD[pcdIdx] = objID
        except:
            validObjCnt -= 1
            continue
    
    outDict = dict()
    outDict['instance_num'] = validObjCnt
    outDict['instance'] = insPCD# Ignore points when id < 0
    outDict['semantic'] = semPCD# Ignore points when id < 0
    outDict['class_id_to_name'] = IDToClassDict
    outDict['class_name_to_id'] = classToIDDict
    SaveDict(os.path.join(SCENE_DIR, outName), outDict)


def SceneAlignment(pcd : o3d.geometry.PointCloud, xyzAng = [0, 0, 0], showF = False):
    from scipy.spatial.transform import Rotation as R
    rMat = R.from_euler('xyz', xyzAng, True).as_matrix()
    tVec = -pcd.get_center()
    transPCD = cp.deepcopy(pcd).translate(tVec).rotate(rMat)
    transPCD = transPCD.translate(-transPCD.get_min_bound())
    if (showF) : o3d.visualization.draw_geometries([transPCD, DrawAxis(8)])
    return transPCD


def CropScene(pcd : o3d.geometry.PointCloud, semList, insList, minBoundVar, maxBoundVar, showF = False):
    assert len(semList) == len(insList)
    minBound = pcd.get_min_bound() + minBoundVar
    maxBound = pcd.get_max_bound() + maxBoundVar
    
    box = o3d.geometry.AxisAlignedBoundingBox(minBound, maxBound)
    ptsList = box.get_point_indices_within_bounding_box(pcd.points)
    ptsList = sorted(ptsList)
    newSemList = []
    newInsList = []
    
    for idx in ptsList:
        newSemList.append(semList[idx])
        newInsList.append(insList[idx])
    newPCD = pcd.select_by_index(ptsList)
    if (showF) : o3d.visualization.draw_geometries([newPCD, DrawAxis(8)])
    return newPCD, newSemList, newInsList


testF = True
if __name__ == '__main__':
    SCENE_DIR = 'D:\\OneDrive - ntut.edu.tw\\office_4'
    if (not testF):
        GetSceneSemanticInfo(SCENE_DIR)
    else:
        readDict = ReadDict(os.path.join(SCENE_DIR, 'sceneLabel.pkl'))
        semPts = readDict['semantic']
        insPts = readDict['instance']
        classNameToIDDict = readDict['class_name_to_id']
        
        partPtsList = np.where(np.asarray(semPts) == classNameToIDDict['chair'])[0]
        
        pcd = o3d.io.read_point_cloud(os.path.join(SCENE_DIR, 'mesh.ply'))
        part = pcd.select_by_index(partPtsList)
        o3d.visualization.draw_geometries([part])
        
        pcd = SceneAlignment(pcd, [0, 0, 30])
        minBoundVar = [0, 0, 0]
        maxBoundVar = [0, 0, -0.1]
        pcd, semPts, insPts = CropScene(pcd, semPts, insPts, minBoundVar, maxBoundVar)
        
        partPtsList = np.where(np.asarray(semPts) == classNameToIDDict['chair'])[0]
        part = pcd.select_by_index(partPtsList)
        o3d.visualization.draw_geometries([part])
    
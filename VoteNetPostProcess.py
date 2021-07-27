# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:14:22 2021

@author: cocob
"""


import os
import csv
import copy as cp
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from Utils import SaveDict
import Parsers
import Labels

uniSeedColor = [0, 0, 1]
uniVoteColor = [1, 0, 0]
uniOBBColor = [1, 0, 1]
eps = 0.05
minPts = 30
combThresh = 0.14

floorZ = None

labelType = None

################ Read and Write Data Process ################

def GetDataFromCSV(DIR_PATH, OBBName = 'OBB.csv', VOTE_DIR = 'votes', SEED_DIR = 'seeds'):
    global labelType
    if ('sunrgbd' in DIR_PATH) : labelType = 'sunrgbd'
    if ('scannet' in DIR_PATH) : labelType = 'scannet'
    
    obbParams = []
    votePCDPathList = []
    seedPCDPathList = []
    with open(os.path.join(DIR_PATH, OBBName), 'r', encoding = 'utf-8') as fr:
        csvReader = csv.reader(fr)
        for row in csvReader:
            obbParams.append([float(row[cnt]) for cnt in range(8)])
            votePCDPathList.append(os.path.join(DIR_PATH, VOTE_DIR, row[8]))
            seedPCDPathList.append(os.path.join(DIR_PATH, SEED_DIR, row[9]))

    obbs = []
    labels = []
    for obbParam in obbParams:
        center = np.array([obbParam[0], obbParam[1], obbParam[2]])
        eulerAng = np.asarray([0, 0, obbParam[6]])
        rotation = R.from_euler('xyz', eulerAng).as_matrix()
        extent = np.array([obbParam[3], obbParam[4], obbParam[5]])
        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        obb.color = uniOBBColor
        obbs.append(obb)
        labels.append(obbParam[7])
    
    votePCDList = []
    seedPCDList = []
    for votePCDPath, seedPCDPath in zip(votePCDPathList, seedPCDPathList):
        votePCD = o3d.io.read_point_cloud(votePCDPath)
        seedPCD = o3d.io.read_point_cloud(seedPCDPath)
        votePCD.paint_uniform_color(uniVoteColor)
        seedPCD.paint_uniform_color(uniSeedColor)
        votePCDList.append(votePCD)
        seedPCDList.append(seedPCD)
    
    return obbs, votePCDList, seedPCDList, labels
    

def GetLabel2CategoryTable():
    table = dict()
    if (labelType == 'scannet'):
        for _id in Labels.scannet : table[_id] = Labels.nyu2ModelNet40[Labels.scannet[_id]]
    elif (labelType == 'sunrgbd'):
        table = Labels.sunrgbd
    return table


################ Read and Write Data Process ################


################ Show Process ################

def GetSceneProp(OBJList, src, showF = True):
    scenePropDict = dict()
    scenePropDict['points'] = len(np.asarray(src.points))
    scenePropDict['sceneDensity'] = np.mean(np.asarray(src.compute_nearest_neighbor_distance()))
    scenePropDict['proposals'] = len(OBJList)
    
    objDensities = 0
    for obj in OBJList : objDensities += np.mean(np.asarray(obj.compute_nearest_neighbor_distance()))
    scenePropDict['objectDensity'] = objDensities / len(OBJList)
    scenePropDict['floor'] = floorZ
    
    if (showF):
        print('Points:', scenePropDict['points'])
        print('PC Density:', scenePropDict['sceneDensity'])
        print('Num of Proposals:', scenePropDict['proposals'])
        print('Object Densities:', scenePropDict['objectDensity'])
        print('floor:', scenePropDict['floor'])
    
    return scenePropDict


def ShowAllObjects(seedList, voteList, OBBList, OBJList, specIdList = []):
    assert len(seedList) == len(voteList) == len(OBBList), 'Lists length error'
    objSize = len(OBBList) if (not specIdList) else len(specIdList)
    colSize = np.sqrt(objSize)
    colSize = int(colSize) + 1 if colSize > int(colSize) else colSize
    colStep = 5
    rowStep = 5
    
    colCnt = 0
    rowCnt = 0
    showList = []
    
    for cnt, pack in enumerate(zip(seedList, voteList, OBBList, OBJList)):
        seeds, votes, obb, obj = pack
        if (specIdList and cnt not in specIdList) : continue
        
        x = colCnt * colStep
        y = rowCnt * rowStep
        translation = -obb.get_center()
        translation[0] = translation[0] + x
        translation[1] = translation[1] + y
        
        colCnt += 1
        if (colCnt >= colSize):
            colCnt = 0
            rowCnt += 1
        
        showList.append(cp.deepcopy(seeds).translate(translation))
        showList.append(cp.deepcopy(votes).translate(translation))
        showList.append(cp.deepcopy(obb).translate(translation))
        showList.append(cp.deepcopy(obj).translate(translation))
    
    o3d.visualization.draw_geometries([*showList])

################ Show Process ################


def GetIdxColorTable(groupIds):
    groupIds = set(groupIds)
    colorTable = dict()
    for idx in groupIds:
        colorTable[idx] = np.random.rand(3)
    return colorTable


def OBBCorrection(OBBList, src):# (OBB with z-axis up)
    global floorZ
    srcPts = np.asarray(src.points)
    floorZ = np.percentile(srcPts[:, 2], 20)
    limitZ = floorZ + 0.1
    newOBBList = []
    for obb in OBBList:
        if (obb.get_min_bound()[2] <= floorZ):
            
            diff = limitZ - obb.get_min_bound()[2]
            newCenterZ = (obb.get_max_bound()[2] + limitZ) / 2
            newCenter = cp.deepcopy(obb.center)
            newCenter[2] = newCenterZ
            newExtZ = obb.extent[2] - diff
            newExt = cp.deepcopy(obb.extent)
            newExt[2] = newExtZ
            newOBB = o3d.geometry.OrientedBoundingBox(newCenter, obb.R, newExt)
            newOBB.color = obb.color
            newOBBList.append(newOBB)
        else:
            newOBBList.append(obb)
    return newOBBList


def GetObjectPCD(seedList, OBBList, src, dbscan = True, refineObj = True, refineOBB = False):
    assert len(seedList) == len(OBBList), 'Lists length error'
    newOBBList = []
    newObjList = []
    for seeds, obb in zip(seedList, OBBList):
        obj = obb.get_point_indices_within_bounding_box(src.points)
        obj = src.select_by_index(obj)
        
        if (not dbscan):# Object = points in OBB
            newObjList.append(obj)
            newOBBList.append(obb)
            continue
        
        # DBSCAN clustering
        groupIds = np.asarray(obj.cluster_dbscan(eps, minPts))# clusters indices in obj
        maxIdx = np.max(groupIds)
        
        if (not refineObj):# Object = points in OBB with clustered colors
            colorTable = GetIdxColorTable(groupIds)
            colorTable[-1] = [0, 0, 0]
            colorIds = []
            for idx in groupIds : colorIds.append(colorTable[idx])
            colorIds = np.asarray(colorIds)
            obj.colors = o3d.utility.Vector3dVector(colorIds)
            newObjList.append(obj)
            newOBBList.append(obb)
            continue
        
        # Check seeds within OBB
        newSeeds = obb.get_point_indices_within_bounding_box(seeds.points)
        if (np.asarray(newSeeds).shape[0] < 1):# Object = points in OBB
            newObjList.append(obj)
            newOBBList.append(obb)
            continue
        newSeeds = seeds.select_by_index(newSeeds)
        
        newSeedDist = obj.compute_point_cloud_distance(newSeeds)
        newSeedDist = np.asarray(newSeedDist)
        seedIds = np.where(newSeedDist <= 0.001)[0]# seed indices in obj
        
        # Aggregate groups that insersect with seeds
        objIds = []
        for group in range(maxIdx + 1):
            ids = np.where(groupIds == group)[0]
            if (set(seedIds) & set(ids)) : objIds.extend(ids)

        if (not objIds):# Object = points in OBB
            newObjList.append(obj)
            newOBBList.append(obb)
            continue
        
        # Object = points in groups that insersect with seeds
        newObjList.append(obj.select_by_index(objIds))
        
        if (not refineOBB):# OBB = input OBB
            newOBBList.append(obb)
            continue
        
        # Refine objects OBB with bound (OBB with z-axis up and 0 head)
        newMinBound = newObjList[-1].get_min_bound()
        newMaxBound = newObjList[-1].get_max_bound()
        newCenter = (newMinBound + newMaxBound) / 2
        newExtent = newMaxBound - newMinBound
        newOBB = o3d.geometry.OrientedBoundingBox(newCenter, np.eye(3), newExtent)
        newOBB.color = uniOBBColor
        newOBBList.append(newOBB)
        
    return newObjList, newOBBList


def ObjectIntersectionRate(OBJList, specIdList = None, retType = 'rate'):
    assert retType == 'rate' or retType == 'id', "retType error: support 'rate' or 'id'"
    objSize = len(OBJList) if (not specIdList) else len(specIdList)
    
    intersectionRateList = [[-1 for j in range(objSize)] for i in range(objSize)]# Distance matrix
    
    rowCnt = 0
    for i in range(len(OBJList) - 1):
        if (specIdList and i not in specIdList) : continue
        colCnt = rowCnt + 1
        for j in range(i + 1, len(OBJList)):
            if (specIdList and j not in specIdList) : continue
            intersectionDist = np.asarray(OBJList[i].compute_point_cloud_distance(OBJList[j]))
            intersectionIds = np.where(intersectionDist <= 0.001)[0]
            intersectionRateList[rowCnt][colCnt] = len(intersectionIds) / len(np.asarray(OBJList[i].points))
            intersectionRateList[colCnt][rowCnt] = len(intersectionIds) / len(np.asarray(OBJList[j].points))
            colCnt += 1
        rowCnt += 1
    
    
    if (retType == 'rate'):
        print(intersectionRateList)
        return intersectionRateList
    if (retType == 'id'):
        combSet = set()
        for i in range(objSize):
            ls = np.where(np.asarray(intersectionRateList[i]) > combThresh)[0]
            if (ls.shape[0]):
                for e in ls : combSet.add(e) if (not specIdList) else combSet.add(specIdList[e])
                combSet.add(i) if (not specIdList) else combSet.add(specIdList[i])
        print(combSet)
        return list(combSet)


def ObjectCombination(seedList, voteList, OBBList, OBJList, labels, combineIds = [], newList = None, visualize = False):
    assert len(seedList) == len(voteList) == len(OBBList) == len(OBJList), 'Lists length error'
    assert len(combineIds) > 1, 'No enough combination indices for objects:\n{} < 2'.format(combineIds)
    assert newList == None or newList == 'remove' or newList == 'append', 'newList Error'
    
    combSeedList = []
    combVoteList = []
    combPtsList = []
    combColorsList = []
    
    for idx in combineIds:
        combSeedList.append(np.asarray(cp.deepcopy(seedList[idx].points)))
        combVoteList.append(np.asarray(cp.deepcopy(voteList[idx].points)))
        combPtsList.append(np.asarray(cp.deepcopy(OBJList[idx].points)))
        combColorsList.append(np.asarray(cp.deepcopy(OBJList[idx].colors)))
    
    combSeeds = np.concatenate(combSeedList, axis=0)
    combVotes = np.concatenate(combVoteList, axis=0)
    combPts = np.concatenate(combPtsList, axis=0)
    combColors = np.concatenate(combColorsList, axis=0)
    
    newSeeds = o3d.geometry.PointCloud()
    newSeeds.points = o3d.utility.Vector3dVector(combSeeds)
    newSeeds.paint_uniform_color(uniSeedColor)
    
    newVotes = o3d.geometry.PointCloud()
    newVotes.points = o3d.utility.Vector3dVector(combVotes)
    newVotes.paint_uniform_color(uniVoteColor)
    
    newObj = o3d.geometry.PointCloud()
    newObj.points = o3d.utility.Vector3dVector(combPts)
    newObj.colors = o3d.utility.Vector3dVector(combColors)
    
    newMinBound = newObj.get_min_bound()
    newMaxBound = newObj.get_max_bound()
    newCenter = (newMinBound + newMaxBound) / 2
    newExtent = newMaxBound - newMinBound
    
    newOBB = o3d.geometry.OrientedBoundingBox(newCenter, np.eye(3), newExtent)
    newOBB.color = uniOBBColor
    
    newLabel = labels[combineIds[0]]
    
    if (visualize) : o3d.visualization.draw_geometries([newSeeds, newVotes, newObj, newOBB])
    
    if (not newList) : return newSeeds, newVotes, newOBB, newObj, newLabel
    
    if (newList == 'remove'):
        combineIds = sorted(combineIds, reverse=True)
        for i in combineIds:
            seedList.pop(i)
            voteList.pop(i)
            OBBList.pop(i)
            OBJList.pop(i)
            labels.pop(i)
    seedList.append(newSeeds)
    voteList.append(newVotes)
    OBBList.append(newOBB)
    OBJList.append(newObj)
    labels.append(newLabel)
    
    return seedList, voteList, OBBList, OBJList, labels


def RemoveCombineIdsObject(seedList, voteList, OBBList, OBJList, labels, combineIds = []):
    if (not combineIds) : return seedPCDList, votePCDList, OBBList, OBJList, labels
    combineIds = list(set(combineIds))# Prevent ID switch
    combineIds = sorted(combineIds, reverse=True)# Prevent ID switch
    for _id in combineIds:
        seedList.pop(_id)
        voteList.pop(_id)
        OBBList.pop(_id)
        OBJList.pop(_id)
        labels.pop(_id)
    
    return seedList, voteList, OBBList, OBJList, labels


def GenResultFile(src, OBBList, OBJList, labels, FILE_DIR : str = 'objects', ext : str = '.pcd'):
    assert len(OBBList) == len(OBJList) ==len(labels), 'Lists length error'
    if (not os.path.exists(FILE_DIR)) : os.mkdir(FILE_DIR)
    
    label2Cat = GetLabel2CategoryTable()
    
    outDict = dict()
    for i, pack in enumerate(zip(OBBList, OBJList, labels), 1):
        obb, obj, label = pack
        
        objDict = dict()
        objName = 'obj%03d' %i
        o3d.io.write_point_cloud(os.path.join(FILE_DIR, objName + ext), obj)
        
        obbDict = dict()
        obbDict['center'] = cp.deepcopy(obb.center)
        obbDict['extent'] = cp.deepcopy(obb.extent)
        obbDict['R'] = cp.deepcopy(obb.R)
        
        objDict['obb'] = obbDict
        objDict['label'] = label2Cat[label]
        
        outDict[objName] = objDict
    
    SaveDict(os.path.join(FILE_DIR, 'objects.pkl'), outDict)
    
    sourceDict = dict()
    sourceDict['VoteNetDIR'] = Parsers.VOTE_PATH
    sourceDict['sceneDIR'] = Parsers.SCENE_PATH
    scenePropDict = GetSceneProp(OBJList, src, True)
    
    sceneDict = dict()
    sceneDict['source'] = sourceDict
    sceneDict['prop'] = scenePropDict
    SaveDict(os.path.join(FILE_DIR, 'scene.pkl'), sceneDict)


if __name__ == '__main__':

    src = o3d.io.read_point_cloud(Parsers.SCENE_PATH)
    
    OBBList, votePCDList, seedPCDList, labels = GetDataFromCSV(Parsers.VOTE_PATH)
    OBBList = OBBCorrection(OBBList, src)
    OBJList, OBBList = GetObjectPCD(seedPCDList, OBBList, src, dbscan=True, refineObj=True, refineOBB=False)
    
    # ShowSceneProp(OBJList, src)
    ShowAllObjects(seedPCDList, votePCDList, OBBList, OBJList, specIdList=[])
    
    combineIdsList = Parsers.COMB_LIST# ID respected to original detection
    removeIds = []
    for combineIds in combineIdsList:
        combineIds_th = ObjectIntersectionRate(OBJList, specIdList=combineIds, retType='id')
        ObjectIntersectionRate(OBJList, specIdList=combineIds, retType='rate')
        seedPCDList, votePCDList, OBBList, OBJList, labels = ObjectCombination(seedPCDList, votePCDList, OBBList, OBJList, labels, combineIds_th, newList='append')
        removeIds.extend(combineIds_th)
    seedPCDList, votePCDList, OBBList, OBJList, labels = RemoveCombineIdsObject(seedPCDList, votePCDList, OBBList, OBJList, labels, combineIds=removeIds)
    ShowAllObjects(seedPCDList, votePCDList, OBBList, OBJList, specIdList=[])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1920, height=1080)
    
    # objNum = 11
    # objCen = OBJList[objNum].get_center()
    # visualizer.add_geometry(seedPCDList[objNum])
    # visualizer.add_geometry(votePCDList[objNum])
    # visualizer.add_geometry(OBBList[objNum])
    # visualizer.add_geometry(OBJList[objNum])
    
    objCen = src.get_center()
    visualizer.add_geometry(src)
    for obb in OBBList : visualizer.add_geometry(obb)
    
    view_ctl = visualizer.get_view_control()
    view_ctl.set_front((0, 0, 1))
    view_ctl.set_up((0, 1, 0))
    view_ctl.set_lookat(objCen)
    view_ctl.set_zoom(0.5)
    
    visualizer.run()
    visualizer.destroy_window()
    if (not Parsers.TEST_FLAG) : GenResultFile(src, OBBList, OBJList, labels, Parsers.OBJ_DIR)
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 04:07:34 2021

@author: cocob
"""


import os
import copy as cp
import numpy as np
import open3d as o3d
from operator import itemgetter
from Utils import SaveDict, ReadDict, text_3d, GetObjectData
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

import PCR

################ Class Definitions ################

class ObjFeature:
    def __init__(self, label : str, extent : np.asarray):
        self.label = label
        self.extent = extent
    
    def distance(self, objFeat, sep = False):
        score_label = ScoreCalculation.label(self.label, objFeat.label)
        score_extent = ScoreCalculation.extent(self.extent, objFeat.extent)
        score_total = score_label * 2 + score_extent
        if (sep) : return {'score_label' : score_label, 'score_extent' : score_extent, 'score_total' : score_total}
        else : return score_total
    
    def _getOutputStr(self):
        return f'[ObjFeature] <(label){self.label:12s}(extent){self.extent}>'
    
    def __repr__(self):
        return self._getOutputStr()
    
    def __str__(self):
        return self._getOutputStr()


class LocObjFeature():
    def __init__(self, objFeat, N_featList = [], N_vecList = []):
        self.objFeat = objFeat
        self.N_featList = N_featList
        self.N_vecList = N_vecList
        assert len(N_featList) == len(N_vecList), 'Neighbor list length error'
        self.N_size = 0 if (not N_featList) else len(N_featList)
        self.N_ang = self.calAng(N_vecList[0], N_vecList[1]) if (self.N_size == 2) else None
    
    def calAng(self, vec1, vec2):# Degrees
        u1 = vec1 / np.linalg.norm(vec1)
        u2 = vec2 / np.linalg.norm(vec2)
        return np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
    
    def distance(self, locObjFeat, sep = False):
        score_obj = self.objFeat.distance(locObjFeat.objFeat, sep)
        score_ang = ScoreCalculation.angle(self.N_ang, locObjFeat.N_ang)
        if (self.N_featList and locObjFeat.N_featList):
            score_N_obj = np.min([[loc.distance(_loc)['score_total'] if (sep) else score_obj for _loc in locObjFeat.N_featList] for loc in self.N_featList])
        else:
            score_N_obj = 3
        score_N_vec = ScoreCalculation.N_vecDist(self.N_vecList, locObjFeat.N_vecList, True)
        score_total = (score_obj['score_total'] if (sep) else score_obj) + score_ang * 2 + score_N_obj + score_N_vec * 2
        if (sep) : return {'score_obj' : score_obj, 'score_ang' : score_ang, 
                           'score_N_obj' : score_N_obj, 'score_N_vec' : score_N_vec, 
                           'score_total' : score_total}
        return score_total
    
    def _getOutputStr(self):
        return f'[LocalObjFeature] <(ObjFeature){self.objFeat}\n\
                (N_featList){self.N_featList}\n\
                (N_vecList){self.N_vecList} (N_size){self.N_size} (N_ang){self.N_ang}>'
    
    def __repr__(self):
        return self._getOutputStr()
    
    def __str__(self):
        return self._getOutputStr()
            

class ScoreCalculation():# Lower better
    @staticmethod
    def extent(ext1, ext2):
        return 1 / (1 + np.exp(-0.5 * (np.mean((ext1 - ext2)**2) - 10)))
    
    @staticmethod
    def label(label1, label2):
        return 0 if (label1 == label2) else 1
    
    @staticmethod
    def angle(ang1, ang2):
        if (not(ang1 and ang2)) : return 1
        return 1 / (1 + np.exp(-0.2 * (np.abs(ang1 - ang2) - 30)))
    
    @staticmethod
    def N_vecDist(vecList1, vecList2, crossF = True):
        if (not(vecList1 and vecList2)) : return 1
        sz = min(len(vecList1), len(vecList2))
        if (crossF):
            orderSign1 = np.cross(vecList1[0], vecList1[1])
            orderSign2 = np.cross(vecList2[0], vecList2[1])
            vecList1 = np.linalg.norm(vecList1,axis=1)[:sz]
            vecList2 = np.linalg.norm(vecList2,axis=1)[:sz]
            if (orderSign1[2] < 0) : vecList1 = vecList1[::-1]
            if (orderSign2[2] < 0) : vecList2 = vecList2[::-1]
        else:
            vecList1 = np.asarray(sorted(np.linalg.norm(vecList1,axis=1))[:sz])
            vecList2 = np.asarray(sorted(np.linalg.norm(vecList2,axis=1))[:sz])
        
        return 1 / (1 + np.exp(-0.5 * (np.mean((vecList1 - vecList2)**2) - 10)))

################ Class Definitions ################

def GetObjFeatDict(objDict : dict):
    '''
    Parameters
    ----------
    objDict : dict
        Object dictionary.

    Returns
    -------
    objFeatDict : dict
        Object feature dictionary.\n
        Content: {'objName' : ObjFeature,...}
    '''
    objFeatDict = dict()
    for objName in objDict:
        label = objDict[objName]['label']
        extent = objDict[objName]['obb']['extent']
        objFeatDict[objName] = ObjFeature(label, extent)
    return objFeatDict


def GetLocObjFeatDict(objDict : dict, objFeatDict : dict, nSize : int = 2, nRadius : float = 4, distDim : int = 3):
    '''
    Parameters
    ----------
    objDict : dict
        Object dictionary.
    objFeatDict : dict
        Object feature dictionary.
    nSize : int, optional
        Number of neighbors. The default is 2.
    nRadius : float, optional
        Limit distance of neighbor. The default is 4.
    distDim : int, optional
        Dimension of distance vector. Valid range: 1 to 3. The default is 3.\n
        If 3: use (x,y,z) to calculate distance.\n
        If 2: use (x,y) to calculate distance.\n
        If 1: use (x) to calculate distance.

    Returns
    -------
    LocObjFeatDict : dict
        Object local feature dictionary.\n
        Content: {'objName' : LocObjFeature,...}
    '''
    centerList = [[objName, objDict[objName]['obb']['center']] for objName in objDict]
    assert len(centerList) > nSize, 'objects less then neighbors'
    
    LocObjFeatDict = dict()
    for i, pack in enumerate(centerList):
        rowName = pack[0]
        rowCenter = pack[1]
        distList = []
        for j, pack in enumerate(centerList):
            if (i == j) : continue
            colName = pack[0]
            colCenter = pack[1]
            vec = colCenter[:distDim] - rowCenter[:distDim]
            distList.append([colName, vec, np.linalg.norm(vec)])
        distList = sorted(distList, key=itemgetter(2))
        
        N_featList = []
        N_vecList = []
        for j in range(nSize):
            if (distList[j][2] < nRadius):
                N_featList.append(objFeatDict[distList[j][0]])
                N_vecList.append(distList[j][1])
        
        LocObjFeatDict[rowName] = LocObjFeature(objFeatDict[rowName], N_featList, N_vecList)
    
    return LocObjFeatDict


def ObjectMatching(realObjFeatDict : dict, mapObjFeatDict : dict, thresh : float = None, retType : str = 'ids', retDistMat : bool = False, printF : bool = False):
    '''
    Parameters
    ----------
    realObjFeatDict : dict
        Real scene object dictionary.
    mapObjFeatDict : dict
        Virtual map object dictionary.
    thresh : float, optional
        If assigned, remove matches that greater then thresh. The default is None.
    retType : str, optional
        ids: dict of matching obj; all: score included. The default is 'ids'.
    retDistMat : bool, optional
        Return feature distance matrix. The default is False.
    printF : bool, optional
        Show matching result. The default is False.

    Returns
    -------
    outDict : dict
        While retDistMat=False and retType='ids'.\n
        Content: {'realObjName' : 'mapObjName',...}
    outDict : dict
        While retDistMat=False and retType='all'.\n
        Content: {'realObjName' : {'target' : 'mapObjName', 'score' : score},...}
    outDict, distDict : dict, dict
        While retDistMat=True.\n
        outDict is same as former description.\n
        distDict: {'nameMat' : featNameMat, 'distMat' : featDistMat}
    '''
    assert retType == 'ids' or retType == 'all'
    featDistMat = []
    featNameMat = []
    for realObjName in realObjFeatDict:
        scoreList = []
        tarNameList = []
        for mapObjName in mapObjFeatDict:
            scoreList.append(mapObjFeatDict[mapObjName].distance(realObjFeatDict[realObjName]))
            tarNameList.append([realObjName, mapObjName])
        featDistMat.append(scoreList)
        featNameMat.append(tarNameList)
    
    featDistMat = np.asarray(featDistMat)
    rowIds, colIds = linear_sum_assignment(featDistMat)
    
    outDict = dict()
    for i, j in zip(rowIds, colIds):
        if (printF) : print(featNameMat[i][j], featDistMat[i][j])
        if (thresh and featDistMat[i][j] > thresh) : continue
        if (retType == 'all'):
            propDict = dict()
            propDict['target'] = featNameMat[i][j][1]
            propDict['score'] = featDistMat[i][j]
            outDict[featNameMat[i][j][0]] = propDict
        elif (retType == 'ids'):
            outDict[featNameMat[i][j][0]] = featNameMat[i][j][1]
    
    if (retDistMat):
        distDict = dict()
        distDict['nameMat'] = featNameMat
        distDict['distMat'] = featDistMat
        return outDict, distDict
    return outDict


def ObjTransMatEstimation(realObjDict : dict, mapObjDict : dict, matchDict : dict, ptDim = 3, showF = False):
    def GenPCD(pts, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        return pcd
    
    def GenPairLines(pts1, pts2, color):
        assert pts1.shape == pts2.shape
        ptSize = pts1.shape[0]
        pts = np.concatenate((pts1, pts2), axis=0)
        lines = [[i, i + ptSize]for i in range(ptSize)]
        colors = [color for i in range(ptSize)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    def GenPairLinesGT(pts1, pts2):
        assert pts1.shape == pts2.shape
        # gtMask = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]# global
        # gtMask = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]# local_NO_THRESH
        # gtMask = [0, 0, 0, 1, 1, 1, 1, 1]# local_THRESH=1
        gtMask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]# GT
        ptSize = pts1.shape[0]
        pts = np.concatenate((pts1, pts2), axis=0)
        lines = [[i, i + ptSize]for i in range(ptSize)]
        colors = [[0, 0, 0] if (i) else [1, 0, 0] for i in gtMask]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    
    realPointList = []
    mapPointList = []
    for realName in matchDict:
        mapName = matchDict[realName]
        rPt = np.zeros(3)
        rPt[:ptDim] = realObjDict[realName]['obb']['center'][:ptDim]
        realPointList.append(rPt)
        mPt = np.zeros(3)
        mPt[:ptDim] = mapObjDict[mapName]['obb']['center'][:ptDim]
        mapPointList.append(mPt)
    
    realPointList = np.asarray(realPointList)
    mapPointList = np.asarray(mapPointList)
    realTrans = -np.mean(realPointList, axis=0)
    mapTrans = -np.mean(mapPointList, axis=0)
    
    rPts_zm = realPointList + realTrans
    mPts_zm = mapPointList + mapTrans
    est_rotMat = R.align_vectors(mPts_zm, rPts_zm)[0].as_matrix()
    
    if (showF):
        est_mPts = (est_rotMat @ rPts_zm.transpose()).transpose()
        
        rPCD = GenPCD(rPts_zm, [1, 0, 0])
        mPCD = GenPCD(mPts_zm, [0, 0.7, 0])
        est_mPCD = GenPCD(est_mPts, [0, 0, 1])
        
        est_pairLines = GenPairLines(est_mPts, mPts_zm, [0, 0, 0])
        # est_pairLines = GenPairLinesGT(est_mPts, mPts_zm)
        
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=600, height=600)
        visualizer.add_geometry(rPCD)
        visualizer.add_geometry(mPCD)
        visualizer.add_geometry(est_mPCD)
        visualizer.add_geometry(est_pairLines)
        
        view_ctl = visualizer.get_view_control()
        view_ctl.set_front((0, 0, 1))
        view_ctl.set_up((0, 1, 0))
        view_ctl.set_lookat([0, 0, 0])
        view_ctl.set_zoom(0.7)
        
        view_rend = visualizer.get_render_option()
        view_rend.line_width = 1
        view_rend.point_size = 10
        
        visualizer.run()
    
    from_realTransMat = np.block([[np.eye(3), realTrans.reshape(3, -1)], [0, 0, 0, 1]])
    to_mapTransMat = np.block([[np.eye(3), -mapTrans.reshape(3, -1)], [0, 0, 0, 1]])
    rigidTransMat = np.block([[est_rotMat, np.zeros((3, 1))], [0, 0, 0, 1]])
    finalTransMat = to_mapTransMat @ rigidTransMat @ from_realTransMat
    
    return finalTransMat


def ObjPCRemove(objDict, scene, removeType = 'obb', extentRange = 0.1):
    assert removeType == 'obb' or removeType == 'obj'
    scenePts = cp.deepcopy(scene.points)
    objIdxList = []
    if (removeType == 'obb'):
        for objName in objDict:
            obbPara = objDict[objName]['obb']
            obb = o3d.geometry.OrientedBoundingBox(obbPara['center'], obbPara['R'], obbPara['extent'])
            objIds = obb.get_point_indices_within_bounding_box(scenePts)
            objIdxList.extend(objIds)
        objIdxList = list(set(objIdxList))
        objPCD = scene.select_by_index(objIdxList)
    elif (removeType == 'obj'):
        objPCD = o3d.geometry.PointCloud()
        for objName in objDict:
            objPCD += objDict[objName]['obj']

    
    objIds = np.asarray(scene.compute_point_cloud_distance(objPCD))
    sceneIds = np.where(objIds > extentRange)[0]
    scenePCD = scene.select_by_index(sceneIds)
    
    return scenePCD


if __name__ == '__main__':
    MAP_SCENE_PATH = 'D:/ShareDIR/Replica/room_office2_Aligned.ply'
    REAL_SCENE_PATH = 'D:/ShareDIR/Replica/room_office2_Aligned_viewR.ply'
    MAP_OBJ_DIR = 'objects_room_office2_Aligned'
    REAL_OBJ_DIR = 'objects_room_office2_Aligned_viewR'
    MAP_OBJ_NAME = 'objects.pkl'
    REAL_OBJ_NAME = 'objects.pkl'
    
    mapObjDict = GetObjectData(MAP_OBJ_DIR, MAP_OBJ_NAME)
    realObjDict = GetObjectData(REAL_OBJ_DIR, REAL_OBJ_NAME)
    
    vmap = o3d.io.read_point_cloud(MAP_SCENE_PATH)
    real = o3d.io.read_point_cloud(REAL_SCENE_PATH)
    
    # Create global features
    mapObjFeatDict = GetObjFeatDict(mapObjDict)
    realObjFeatDict = GetObjFeatDict(realObjDict)
    
    # globalMatch = ObjectMatching(realObjFeatDict, mapObjFeatDict, printF=True)# Global feature distances
    # initTransMat = ObjTransMatEstimation(realObjDict, mapObjDict, globalMatch)# Estimate real to map transform
    
    
    # Create local features
    mapLocObjFeatDict = GetLocObjFeatDict(mapObjDict, mapObjFeatDict)
    realLocObjFeatDict = GetLocObjFeatDict(realObjDict, realObjFeatDict)
    
    localMatch = ObjectMatching(realLocObjFeatDict, mapLocObjFeatDict, printF=True)# Local feature distances
    initTransMat = ObjTransMatEstimation(realObjDict, mapObjDict, localMatch)# Estimate real to map transform
    
    # gtMatch = {'obj001' : 'obj014', 'obj002' : 'obj003', 'obj003' : 'obj016', 'obj004' : 'obj001'}
    # initTransMat = ObjTransMatEstimation(realObjDict, mapObjDict, gtMatch)
    
    vmap_nObj = ObjPCRemove(mapObjDict, vmap, 'obj', 0.2)
    real_nObj = ObjPCRemove(realObjDict, real, 'obj', 0.2)
    
    iterTransMat = PCR.ICP(real_nObj, vmap_nObj, initTransMat, True, 50)
    real_tmp = cp.deepcopy(real).transform(iterTransMat)
    real_tmp.paint_uniform_color([1, 0, 0])
    
    
    
    # Visualization

    
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1920, height=1080)
    
    # visualizer.add_geometry(vmap)
    # visualizer.add_geometry(real_tmp)
    # objCen = vmap.get_center()
    
    src1 = o3d.io.read_point_cloud(MAP_SCENE_PATH)
    src2 = o3d.io.read_point_cloud(REAL_SCENE_PATH)
    objCen = src1.get_center()
    
    trans1 = -src1.get_center()
    src1.translate(trans1)
    visualizer.add_geometry(src1)
    for objName in mapObjDict:
        obbPara = mapObjDict[objName]['obb']
        obb = o3d.geometry.OrientedBoundingBox(obbPara['center'], 
                                                obbPara['R'], 
                                                obbPara['extent'])
        obb.color = [1, 0, 1]
        obb.translate(trans1)
        visualizer.add_geometry(obb)
        # text options
        textPos = obb.get_center()
        textPos[2] = obb.get_max_bound()[2] + 0.1
        visualizer.add_geometry(text_3d(objName[-3:], textPos, density=10, font_size=48))
    
    trans2 = -src2.get_center() + np.asarray([8, 0, 0])
    src2.translate(trans2)
    visualizer.add_geometry(src2)
    for objName in realObjDict:
        obbPara = realObjDict[objName]['obb']
        obb = o3d.geometry.OrientedBoundingBox(obbPara['center'], 
                                                obbPara['R'], 
                                                obbPara['extent'])
        obb.color = [1, 0, 1]
        obb.translate(trans2)
        visualizer.add_geometry(obb)
        # text options
        textPos = obb.get_center()
        textPos[2] = obb.get_max_bound()[2] + 0.1
        visualizer.add_geometry(text_3d(objName[-3:], textPos, density=10, font_size=48))
    
    view_ctl = visualizer.get_view_control()
    view_ctl.set_front((0, 0, 1))
    view_ctl.set_up((0, 1, 0))
    view_ctl.set_lookat(objCen)
    view_ctl.set_zoom(0.6)
    
    view_rend = visualizer.get_render_option()
    view_rend.line_width = 1
    
    visualizer.run()
    visualizer.destroy_window()

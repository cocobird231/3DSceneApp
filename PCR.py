# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:34:57 2021

@author: cocob
"""


import os
import copy as cp
import numpy as np
import open3d as o3d
from operator import itemgetter
from scipy.spatial.transform import Rotation as R

import torch

from Module_DCP import DCP, DCPProp
from Parsers import Parser_PCR, initDevice
from Utils import SaveDict, ReadDict, GetModelByName, GetUnitModel


def ICP(pc_from, pc_to, initT, deepCopy = True, iterSize = 50, iterStep = 0.2):
    ICP_TRANSFORM = ICPIter(cp.deepcopy(pc_from), cp.deepcopy(pc_to), initT, iterSize, iterStep)
    ICP_TRANSFORM = ICP_TRANSFORM.transformation
    return ICP_TRANSFORM


def ICPIter(templatePC, targetPC, initTransform, iterSize = 50, iterStep = 0.2):
    ICP_TRANSFORM = o3d.pipelines.registration.registration_icp(
        templatePC, targetPC, iterStep, initTransform, 
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iterSize))
    return ICP_TRANSFORM


def GetTemplateData(TMP_DIR, dictName, BASE_DIR, tmpType = 'pcd', normalizedObj = True, normalizedTmp = False):
    objDict = ReadDict(os.path.join(TMP_DIR, dictName))
    for objName in objDict:
        pcd = o3d.io.read_point_cloud(os.path.join(TMP_DIR, objName + '.pcd'))
        if (normalizedObj):
            objDict[objName]['obj'], objDict[objName]['obj_s'], objDict[objName]['obj_t'] = GetUnitModel(pcd, True, True)
        else:
            objDict[objName]['obj'] = pcd
        
        if (not objDict[objName]['template']) : continue
        tmp = GetModelByName(modelCat=      objDict[objName]['label'], 
                             modelName=     objDict[objName]['template']['rank1'], 
                             ModelBase_DIR= BASE_DIR, 
                             retType=       tmpType)
        if (objDict[objName]['label'] == 'chair') : tmp.paint_uniform_color([0.4, 0.4, 0.9])
        if (objDict[objName]['label'] == 'table') : tmp.paint_uniform_color([0.4, 0.9, 0.4])
        if (normalizedTmp):
            objDict[objName]['tmp'], objDict[objName]['tmp_s'], objDict[objName]['tmp_t'] = GetUnitModel(tmp, True, True)
        else:
            objDict[objName]['tmp'] = tmp# tmp already normalized in pcd type (not normalized in mesh type)

    return objDict# Delete obj, obj_s, obj_t and tmp while saving objDict


def GetRigidTransform(objDict, method = 'dcp_icp', args = None, net = None):
    assert method == 'dcp_icp' or method == 'icp_4', 'Rigid prediction method error'
    if (method == 'dcp_icp'):
        for objName in objDict:
            if (not objDict[objName]['template']): 
                objDict[objName]['transform'] = None
                continue
            
            tmpPCD = objDict[objName]['tmp']
            objPCD = objDict[objName]['obj']
            
            tmpPCD.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            objPCD.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            
            tmp = np.asarray(cp.deepcopy(tmpPCD.points)).astype('float32')
            obj = np.asarray(cp.deepcopy(objPCD.points))[:2048].astype('float32')
            
            tmp = torch.tensor(tmp).view(1, -1, 3)
            obj = torch.tensor(obj).view(1, -1, 3)
            
            if (args.cuda) : tmp = tmp.cuda()
            if (args.cuda) : obj = obj.cuda()
            
            rot_ab_pred, trans_ab_pred, _, _ = net(obj, tmp)
            if (args.cuda) : rot_ab_pred = rot_ab_pred.cpu()
            if (args.cuda) : trans_ab_pred = trans_ab_pred.cpu()
            
            outRot = rot_ab_pred.detach().numpy().squeeze()
            outTrans = trans_ab_pred.detach().numpy().squeeze()
            outTransMat = np.block([[outRot, outTrans.reshape(3, -1)], [np.eye(4)[-1]]])
            
            ICP_TRANSFORM = ICP(objPCD, tmpPCD, outTransMat)
            
            outRot = ICP_TRANSFORM[:3, :3].T
            outTrans = -outRot.dot(ICP_TRANSFORM[:3, 3])
            outTransMat = ICP_TRANSFORM

            transDict = dict()
            transDict['R'] = outRot
            transDict['t'] = outTrans
            transDict['T'] = np.block([[outRot, outTrans.reshape(3, -1)], [0, 0, 0, 1]])
            objDict[objName]['transform'] = transDict
            
    elif (method == 'icp_4'):
        for objName in objDict:
            if (not objDict[objName]['template']): 
                objDict[objName]['transform'] = None
                continue
            
            tmpPCD = objDict[objName]['tmp']
            objPCD = objDict[objName]['obj']
            
            tmpPCD.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            objPCD.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            
            scoreList = []
            angList = [0, 90, 180, 270]# z-axis rotation angle
            for ang in angList:
                rotMat = R.from_euler('xyz', [0, 0, ang], degrees=True).as_matrix()
                initTrans = np.eye(4)
                initTrans[:3,:3] = rotMat
                
                ICP_TRANSFORM = ICP(objPCD, tmpPCD, initTrans)
                ICP_PC = cp.deepcopy(objPCD).transform(ICP_TRANSFORM)
                ChDist = np.sum(np.asarray(ICP_PC.compute_point_cloud_distance(tmpPCD))) + \
                        np.sum(np.asarray(tmpPCD.compute_point_cloud_distance(ICP_PC)))
                scoreList.append([ICP_TRANSFORM, ChDist])
                
            scoreList = sorted(scoreList, key=itemgetter(1))
            outRot = scoreList[0][0][:3, :3].T
            outTrans = -outRot.dot(scoreList[0][0][:3, 3])
            outTransMat = np.block([[outRot, outTrans.reshape(3, -1)], [0, 0, 0, 1]])
            
            transDict = dict()
            transDict['R'] = outRot
            transDict['t'] = outTrans
            transDict['T'] = outTransMat
            objDict[objName]['transform'] = transDict
    
    return objDict


def GetTemplateTransform(objDict, args = None):
    for objName in objDict:
        if (not objDict[objName]['template']) : continue
        
        TransMatObj = np.block([[np.eye(3), -objDict[objName]['obj_t'].reshape(3, -1)], [0, 0, 0, 1]])
        TransMatRig = objDict[objName]['transform']['T']
        
        transDict_pcd = dict()
        transDict_pcd['T'] = TransMatObj @ TransMatRig
        transDict_pcd['s'] = 1 / objDict[objName]['obj_s']
        
        tmpMesh = GetModelByName(objDict[objName]['label'], 
                                 objDict[objName]['template']['rank1'], 
                                 args.modelBasePath, 'mesh')
        tmpMesh, tmp_s, tmp_t = GetUnitModel(tmpMesh, True, True)
        
        TransMatTmp = np.block([[np.eye(3), tmp_t.reshape(3, -1)], [0, 0, 0, 1]])
        
        transDict_mesh = dict()
        transDict_mesh['T'] = TransMatObj @ TransMatRig @ TransMatTmp
        transDict_mesh['s'] = tmp_s / objDict[objName]['obj_s']
        
        transDict = dict()
        transDict['pcd'] = transDict_pcd
        transDict['mesh'] = transDict_mesh
        transDict['unit'] = TransMatRig
        objDict[objName]['transform'] = transDict
        
        
    return objDict


def DelObjectPC(objDict):
    for objName in objDict:
        objDict[objName].pop('obj', None)
        objDict[objName].pop('obj_s', None)
        objDict[objName].pop('obj_t', None)
        objDict[objName].pop('tmp', None)
        objDict[objName].pop('tmp_s', None)
        objDict[objName].pop('tmp_t', None)
    
    return objDict


def ShowTemplate(objDict, modelBasePath, specIdList = []):
    rowCnt = 0
    objCnt = 0
    showList = []
    for objName in objDict:
        if (specIdList and (objCnt not in specIdList)):
            objCnt += 1
            continue
        pcd = GetUnitModel(objDict[objName]['obj'], True)
        pcd.translate([0, rowCnt, 0])
        
        if (objDict[objName]['template']):
            transMat = objDict[objName]['transform']['unit']
            
            tmp = GetModelByName(objDict[objName]['label'], 
                                 objDict[objName]['template']['rank1'], 
                                 modelBasePath, 'pcd').transform(transMat)
            
            tmpMesh = GetModelByName(objDict[objName]['label'], 
                                     objDict[objName]['template']['rank1'], 
                                     modelBasePath, 'mesh')
            tmpMesh = GetUnitModel(tmpMesh, True).transform(transMat)
            
            tmp.translate([1, rowCnt, 0])
            tmpMesh.translate([2, rowCnt, 0])
            showList.extend([tmp, tmpMesh])
        showList.append(pcd)
        rowCnt += 1
        objCnt += 1
    o3d.visualization.draw_geometries([*showList], mesh_show_wireframe=True)


def ShowAllTemplates(objDict, modelBasePath):
    rowCnt = 0
    showList = []
    for objName in objDict:
        pcd = GetUnitModel(objDict[objName]['obj'], True)
        pcd.translate([0, rowCnt, 0])
        
        if (objDict[objName]['template']):
            transMat = objDict[objName]['transform']['unit']
            
            tmp = GetModelByName(objDict[objName]['label'], 
                                 objDict[objName]['template']['rank1'], 
                                 modelBasePath, 'pcd').transform(transMat)
            
            tmpMesh = GetModelByName(objDict[objName]['label'], 
                                     objDict[objName]['template']['rank1'], 
                                     modelBasePath, 'mesh')
            tmpMesh = GetUnitModel(tmpMesh, True).transform(transMat)
            
            tmp.translate([1, rowCnt, 0])
            tmpMesh.translate([2, rowCnt, 0])
            showList.extend([tmp, tmpMesh])
        showList.append(pcd)
        rowCnt += 1
    o3d.visualization.draw_geometries([*showList], mesh_show_wireframe=True)


if __name__ == '__main__':
    args = Parser_PCR()
    args = initDevice(args)
    TMP_DIR = args.objectDIR
    TMP_NAME = args.template
    BASE_DIR = args.modelBasePath
    if (not args.test):
        objDict = GetTemplateData(TMP_DIR, TMP_NAME, BASE_DIR, 'pcd', True, False)# All model normalized
        
        
        if (args.method == 'dcp_icp'):
            net = DCP(DCPProp())
            net.load_state_dict(torch.load(args.modelPath, map_location='cpu'))
            net.to(args.device)
            net.eval()
        else : net = None
        
        import time
        print('start')
        t1 = time.time()
        objDict = GetRigidTransform(objDict, args.method, args, net)
        print(time.time() - t1)
        
        objDict = GetTemplateTransform(objDict, args)
        DelObjectPC(objDict)
        SaveDict(os.path.join(TMP_DIR, 'transforms_%s.pkl' %args.method), objDict)
    else:
        TRANS_NAME = args.transform
        objDict = GetTemplateData(TMP_DIR, TRANS_NAME, BASE_DIR, 'mesh', False, False)
        
        ShowAllTemplates(objDict, BASE_DIR)
        # ShowTemplate(objDict, BASE_DIR, [])# 0, 4, 15, 17
        
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=1920, height=1080)
        
        scenePropDict = ReadDict(os.path.join(TMP_DIR, 'scene.pkl'))
        
        pcd = o3d.io.read_point_cloud(scenePropDict['source']['sceneDIR'])
        visualizer.add_geometry(pcd)
        
        for objName in objDict:
            if (not objDict[objName]['template']) : continue
            tmpMesh = cp.deepcopy(objDict[objName]['tmp'])
            tmpMesh.scale(objDict[objName]['transform']['mesh']['s'], center = tmpMesh.get_center())
            tmpMesh.transform(objDict[objName]['transform']['mesh']['T'])
            if (tmpMesh) : visualizer.add_geometry(tmpMesh)
        
        view_ctl = visualizer.get_view_control()
        view_ctl.set_front((0, 1, 0.5))
        view_ctl.set_up((0, -1, 0))
        view_ctl.set_lookat(pcd.get_center())
        view_ctl.set_zoom(0.5)
        
        view_rend = visualizer.get_render_option()
        view_rend.mesh_show_wireframe = True
        
        visualizer.run()
        visualizer.destroy_window()
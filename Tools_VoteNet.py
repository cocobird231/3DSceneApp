# -*- coding: utf-8 -*-
"""
Created on Sun May 16 01:52:13 2021

@author: cocob
"""

import os
import time
import copy as cp
import numpy as np
import open3d as o3d
import pickle as pkl
from Utils import DrawAxis, jitter_pointcloud, SaveDict, ReadDict
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

map_deg = interp1d([0, 1], [0, 180])


def CvtCamParaToDict(camPara):
    intDict = dict()
    intDict['width'] = camPara.intrinsic.width
    intDict['height'] = camPara.intrinsic.height
    intMat = camPara.intrinsic.intrinsic_matrix
    intDict['fx'] = intMat[0][0]
    intDict['fy'] = intMat[1][1]
    intDict['cx'] = intMat[0][2]
    intDict['cy'] = intMat[1][2]
    
    extDict = dict()
    extDict['extrinsic'] = camPara.extrinsic
    
    camParaDict = dict()
    camParaDict['intrinsic'] = intDict
    camParaDict['extrinsic'] = extDict
    
    return camParaDict


def DrawBBox(minBound = np.array([0, 0, 0]), maxBound = np.array([1, 1, 1]), color = (1, 0, 0)):
    bbox = o3d.geometry.AxisAlignedBoundingBox(minBound, maxBound)
    bbox.color = color
    return bbox


def AlignScene(PCD_PATH):
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    
    rotAng = [0, 0, -6]
    rotMat = R.from_euler('xyz', rotAng, degrees=True).as_matrix()
    pcd.rotate(rotMat)
    
    
    minBound = pcd.get_min_bound()
    maxBound = pcd.get_max_bound()
    maxBound[2] -= 0.2
    box = DrawBBox(minBound, maxBound)
    pcd = pcd.crop(box)
    
    
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1920, height=1080)
    
    visualizer.add_geometry(DrawAxis(5))
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(box)
    
    view_ctl = visualizer.get_view_control()
    
    view_ctl.set_front((0, 0, 1))
    view_ctl.set_up((0, 1, 0))
    view_ctl.set_lookat(pcd.get_center())
    view_ctl.set_zoom(1)
    
    visualizer.run()
    

    
    visualizer.destroy_window()
    
    # o3d.visualization.draw_geometries([pcd, DrawAxis(), box], 
    #                                   zoom=0.5,
    #                                   front=[1, 0, 0],
    #                                   lookat=[0, 0, 0],
    #                                   up=[0, 1, 0])
    # o3d.visualization.draw_geometries([pcd, DrawAxis(), box], 
    #                                   zoom=0.5,
    #                                   front=[0, 0, 1],
    #                                   lookat=[0, 0, 0],
    #                                   up=[0, 1, 0])
    PCD_NAME = PCD_PATH[:-4] + '_Aligned'
    # o3d.io.write_point_cloud(PCD_NAME + PCD_PATH[-4:], pcd)
    # o3d.io.write_point_cloud(PCD_PATH, pcd)


def ShowSceneWithCamPara(PCD_PATH : str, CAM_PATH : str):
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    
    camParaDict = ReadDict(CAM_PATH)
    camExtDict = camParaDict['extrinsic']
    camIntDict = camParaDict['intrinsic']
    
    camPara = o3d.camera.PinholeCameraParameters()
    camPara.extrinsic = camExtDict['extrinsic']
    camPara.intrinsic = o3d.camera.PinholeCameraIntrinsic(camIntDict['width'], 
                                                          camIntDict['height'], 
                                                          camIntDict['fx'], 
                                                          camIntDict['fy'], 
                                                          camIntDict['cx'], 
                                                          camIntDict['cy'])
    
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(width=1920, height=1080)
    visualizer.add_geometry(pcd)
    view_ctl = visualizer.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(camPara)
    
    
    visualizer.run()
    visualizer.destroy_window()


def GetCamViewPointCloud(PCD_PATH : str, VIEW_DIR : str = 'view'):
    if (not os.path.exists(VIEW_DIR)) : os.mkdir(VIEW_DIR)
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    # o3d.visualization.draw_geometries_with_editing([pcd])

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(width=1920, height=1080)
    
    visualizer.add_geometry(pcd)
    
    view_ctl = visualizer.get_view_control()
    view_ctl.set_front((0, 0, 1))
    view_ctl.set_up((0, 1, 0))
    view_ctl.set_lookat(pcd.get_center())
    view_ctl.set_zoom(1)
    visualizer.run()
    visualizer.destroy_window()
    
    obj = CvtCamParaToDict(view_ctl.convert_to_pinhole_camera_parameters())
    SaveDict(os.path.join(VIEW_DIR, 'camPara.pkl'), obj)



def CombineCamViewPointCloud_HPR(pcdPathList, camPathList, useHPR : bool = True, saveDIR = 'view'):
    assert len(pcdPathList) == len(camPathList)
    pcdSize = len(pcdPathList)
    
    outputDict = dict()
    outputCamDict = dict()
    
    varList = [[[0, 0, 0] for j in range(2)] for i in range(pcdSize)]
    varList[0][0] = [0.2, 0.2, 0]
    
    combList = []
    for i, pack in enumerate(zip(pcdPathList, camPathList, varList)):
        pcdPath, camPath, var = pack
        ShowSceneWithCamPara(pcdPath, camPath)
        
        pcd = o3d.io.read_point_cloud(pcdPath)
        camParaDict = ReadDict(camPath)
        outputCamDict[i] = camParaDict
        
        minBound = pcd.get_min_bound()
        maxBound = pcd.get_max_bound()
        minBound += var[0]
        maxBound += var[1]
        
        box = DrawBBox(minBound, maxBound)
        pcd = pcd.crop(box)
        
        o3d.visualization.draw_geometries([pcd, DrawAxis(3), box])
        
        if (useHPR):
            camPos = camParaDict['extrinsic']['extrinsic'][:3,3]
            _, sub_points_map = pcd.hidden_point_removal(camPos, 30000)
            hprPC = pcd.select_by_index(sub_points_map)
            
            o3d.visualization.draw_geometries([hprPC])
            combList.append(hprPC)
        else : combList.append(pcd)
    
    # Combine multi-view PC
    basePC = combList[0]
    for i in range(1, pcdSize) : basePC += combList[i]
    
    basePts = jitter_pointcloud(np.asarray(cp.deepcopy(basePC.points)))
    basePC.points = o3d.utility.Vector3dVector(basePts)
    
    randRotMat = R.from_euler('xyz', [0, 0, map_deg(np.random.rand())], degrees=True).as_matrix()
    print(randRotMat)
    randTransVec = np.random.rand(3) * 10
    randTransMat = np.block([[randRotMat, randTransVec.reshape(3, 1)], [0, 0, 0, 1]])
    basePC.transform(randTransMat)
    
    o3d.visualization.draw_geometries([basePC, DrawAxis(5, localCenter=list(basePC.get_center())), DrawAxis()])
    o3d.io.write_point_cloud(os.path.join(saveDIR, 'combine.ply'), basePC)
    outputDict['rigid'] = randTransMat
    outputDict['camera'] = outputCamDict
    SaveDict(os.path.join(saveDIR, 'scene.pkl'), outputDict)


if (__name__ == '__main__'):
    PCD_PATH = 'D:\\ShareDIR\\Replica\\room_office1_Aligned.ply'
    VIEW_DIR = 'view_room_office1_Aligned'
    # GetCamViewPointCloud(PCD_PATH, VIEW_DIR)
    CombineCamViewPointCloud_HPR([os.path.join(VIEW_DIR, 'cropped_1.ply')], 
                                  [os.path.join(VIEW_DIR, 'camPara.pkl')], saveDIR = VIEW_DIR)
    # ShowSceneWithCamPara('view/cropped_2.ply', 'view/camPara_2.pkl')
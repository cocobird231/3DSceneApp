# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:08:21 2021

@author: cocob
"""

import os
import copy as cp
from shutil import copyfile

import numpy as np
import open3d as o3d
import torch

from Utils import WalkModelNet40CatDIR, WalkModelNet40ByCatName, SaveDict, ReadDict
from Parsers import Parser_ModelBase, initDevice


def FixMeshFileHeader(modelPath : str, exportPath : str):
    content = []
    with open(modelPath, 'r') as f:
        content = f.readlines()
    # Fix off file header with seperate first line into two lines, 
    # which contents OFF header and model info respectively.
    content[0] = content[0][3:]
    with open(exportPath, 'w') as f:
        f.write('OFF\n')
        for c in content : f.write(c)
    return


def GenUnitSpherePCDFromMesh(modelPath : str, pointSize : int = 0):
    try:
        mesh = o3d.io.read_triangle_mesh(modelPath)
        if (mesh.is_empty()) : raise 'Empty geometry'
        maxBound = mesh.get_max_bound()
        minBound = mesh.get_min_bound()
        length = np.linalg.norm(maxBound - minBound, 2)
        mesh = mesh.scale(1 / length, center = mesh.get_center())
        mesh = mesh.translate(-mesh.get_center())
        return mesh.sample_points_uniformly(pointSize)
    except:
        raise 'Geometry error'


def GenModelBaseDir(ModelNet40_DIR : str, ModelBase_DIR : str):
    # Generate ModelBase directory
    if (not os.path.exists(ModelBase_DIR)) : os.mkdir(ModelBase_DIR)
    mesh_DIR = os.path.join(ModelBase_DIR, 'mesh')
    if (not os.path.exists(mesh_DIR)) : os.mkdir(mesh_DIR)
    pcd_DIR = os.path.join(ModelBase_DIR, 'pcd')
    if (not os.path.exists(pcd_DIR)) : os.mkdir(pcd_DIR)
    # Search model category
    catList = WalkModelNet40CatDIR(ModelNet40_DIR)
    # Search model in each category
    for cat in catList:
        # Generate ModelBase category directory
        mesh_cat_DIR = os.path.join(mesh_DIR, cat)
        if (not os.path.exists(mesh_cat_DIR)) : os.mkdir(mesh_cat_DIR)
        pcd_cat_DIR = os.path.join(pcd_DIR, cat)
        if (not os.path.exists(pcd_cat_DIR)) : os.mkdir(pcd_cat_DIR)
        
        print('Processing category:', mesh_cat_DIR)
        filePathList, fileNameList = WalkModelNet40ByCatName(ModelNet40_DIR, cat, '.off', 'all')
        for path, name in zip(filePathList, fileNameList):
            try:
                pcd = GenUnitSpherePCDFromMesh(path, 2048)
                copyfile(path, os.path.join(mesh_cat_DIR, name))
                o3d.io.write_point_cloud(os.path.join(pcd_cat_DIR, '%s.pcd' %name[:-4]), pcd)
            except:
                FixMeshFileHeader(path, os.path.join(mesh_cat_DIR, name))
                pcd = GenUnitSpherePCDFromMesh(path, 2048)
                o3d.io.write_point_cloud(os.path.join(pcd_cat_DIR, '%s.pcd' %name[:-4]), pcd)


def GenModelBaseFeature():
    from Module_PointNetSeries import PointNet2Comp2
    args = Parser_ModelBase()
    args = initDevice(args)
    net = PointNet2Comp2(0, 40, 'cls')
    net.to(args.device)
    net.load_state_dict(torch.load(args.modelPath, map_location=args.device))
    net.eval()
    
    pcd_DIR = os.path.join(args.dataset, 'pcd')
    catList = WalkModelNet40CatDIR(pcd_DIR)
    catDict = dict()
    for cat in catList:
        print('Processing category:', cat)
        filePathList, fileNameList = WalkModelNet40ByCatName(pcd_DIR, cat, '.pcd', 'all')
        featDict = dict()
        for path, name in zip(filePathList, fileNameList):
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(cp.deepcopy(pcd.points)).astype('float32')
            pts = torch.tensor(pts).view(1, -1, 3)
            if (args.cuda) : pts = pts.cuda()
            _, feat = net(pts)
            if (args.cuda): feat = feat.cpu()
            feat = feat.detach().numpy().squeeze()
            featDict[name[:-4]] = feat
        catDict[cat] = featDict
    
    SaveDict(os.path.join(args.dataset, 'features.pkl'), catDict)
    

def TestModelBaseFeature(FEAT_PATH : str):
    features = ReadDict(FEAT_PATH)
    valueSize = 0
    for i, cat in enumerate(features):
        valueSize += len(features[cat])
        print(f'{cat:12}: {len(features[cat])}')
    print('Total values:', valueSize)


if __name__ == '__main__':
    # ModelNet40_DIR = 'D:\\Datasets\\ModelNet40'
    # ModelBase_DIR = 'D:\\Datasets\\ModelNet40_Base'
    # GenModelBaseDir(ModelNet40_DIR, ModelBase_DIR)
    GenModelBaseFeature()# ubuntu env (gpu support)
    # TestModelBaseFeature('features_pn2_scale.pkl')
    
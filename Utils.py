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


#############################################################
#               Dictionary File Processing
#############################################################

def SaveDict(FILE_NAME, obj):
    with open(FILE_NAME, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def ReadDict(FILE_NAME):
    with open(FILE_NAME, 'rb') as f:
        return pkl.load(f)


def GetObjectData(FILE_DIR : str = 'objects', dictName = 'objects.pkl', ext = '.pcd'):
    objectsDict = ReadDict(os.path.join(FILE_DIR, dictName))
    for objName in objectsDict:
        pcd = o3d.io.read_point_cloud(os.path.join(FILE_DIR, objName + ext))
        objectsDict[objName]['obj'] = pcd
    return objectsDict


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


#############################################################
#                   Visualization Implementation
#############################################################

def DrawAxis(length = 10, localCenter = None):
    o = [0, 0, 0] if (not localCenter) else localCenter
    points = [o, [o[0] + length, o[1], o[2]], [o[0], o[1] + length, o[2]], [o[0], o[1], o[2] + length]]
    lines = [[0, 1], [0, 2], [0, 3]]# x, y, z
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]# r, g, b
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def DrawBox(center, extent, rotMat = np.eye(3), color = None, axis = False, axis_length = 0.5):
    color = [0, 0, 0] if (not color) else color
    obb = o3d.geometry.OrientedBoundingBox(center, rotMat, extent)
    obb.color = color
    if (not axis) : return obb
    
    obb = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_points = np.asarray(cp.deepcopy(obb.points))
    obb_colors = np.asarray(cp.deepcopy(obb.colors))
    obb_lines = np.asarray(cp.deepcopy(obb.lines))
    
    print(obb_points.shape, obb_colors, obb_lines)
    
    axis = DrawAxis(axis_length, center)
    axis_points = np.asarray(cp.deepcopy(axis.points))
    axis_colors = np.asarray(cp.deepcopy(axis.colors))
    axis_lines = np.asarray(cp.deepcopy(axis.lines)) + obb_points.shape[0]
    
    print(axis_points.shape, axis_colors, axis_lines)
    
    obb.points = o3d.utility.Vector3dVector(np.concatenate((obb_points, axis_points), axis=0))
    obb.colors = o3d.utility.Vector3dVector(np.concatenate((obb_colors, axis_colors), axis=0))
    obb.lines = o3d.utility.Vector2iVector(np.concatenate((obb_lines, axis_lines), axis=0))
    
    print(obb)
    
    
    return obb


#https://github.com/intel-isl/Open3D/issues/2#issuecomment-610683341
def text_3d(text, pos, direction=None, degree=0.0, density=1, font='utils/times.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param density: https://github.com/intel-isl/Open3D/issues/2#issuecomment-620387385
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


#############################################################
#                   Point Cloud Implementation
#############################################################

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


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def scaling_pointCloud(pointcloud, scalingScalar = 0.2):
    coeff = np.random.uniform(1 - scalingScalar, 1 + scalingScalar)
    pointcloud = pointcloud * coeff
    return pointcloud


#############################################################
#                   Rigid Implementation
#############################################################

def invRigid(transMat):
    inv_randRigid = cp.deepcopy(transMat)
    inv_randRigid[:3,:3] = transMat[:3,:3].transpose()
    inv_randRigid[:3,3] = (-inv_randRigid[:3,:3] @ inv_randRigid[:3,3]).reshape(1, 3)
    return inv_randRigid


# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:18:01 2021

@author: cocob
"""


import argparse
import torch

# ModelBase
MODEL_BASE_DIR = 'D:/Datasets/ModelNet40_Base'
MODEL_FEAT_PATH = 'features/features_g.pkl'

# VoteNetPoseProcess
VOTE_PATH = 'office_4/scannet_mesh_Aligned_results'
SCENE_DIR = 'office_4'
SCENE_PATH = 'office_4/mesh_Aligned.ply'
OBJ_DIR = 'office_4/obj'
COMB_LIST = [[2, 3, 8, 11, 14, 17, 22]]

# ModelSelector
MS_MODEL_PATH = 'models/ModelSelector_g.pth'
#-- test --
TMP_NAME = 'templates_pn2_scale.pkl'

# PCR
PCR_METHOD = 'dcp_icp'
DCP_MODEL_PATH = 'models/DCP_PN_V_E1000_D2.t7'
#-- test --
TRANS_NAME = 'transforms_pn2_scale_icp_4.pkl'

TEST_FLAG = True


def Parser_ModelBase():
    parser = argparse.ArgumentParser(description='ModelBaseTools')
    # Required arguments
    parser.add_argument('-d', '--dataset',      required=False, type=str, metavar='PATH', 
                        default=MODEL_BASE_DIR, help='ModelBase dataset path')
    parser.add_argument('-m', '--modelPath',    required=False, type=str, metavar='PATH', 
                        default=MS_MODEL_PATH, help='Pre-trained model path for ModelSelector')
    # Device settings
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device')
    # Program flag, do not implement
    parser.add_argument('--device', type=str, 
                        default=None, help='torch network device')# do not implement
    
    return parser.parse_args()


def Parser_ModelSelector():
    parser = argparse.ArgumentParser(description='ModelSelector')
    # Required arguments
    parser.add_argument('-o', '--objectDIR',        required=False, type=str, metavar='PATH', 
                        default=OBJ_DIR, help='objects directory')
    parser.add_argument('-f', '--modelFeature',     required=False, type=str, metavar='PATH', 
                        default=MODEL_FEAT_PATH, help='ModelBase feature path')
    parser.add_argument('-m', '--modelPath',        required=False, type=str, metavar='PATH', 
                        default=MS_MODEL_PATH, help='Pre-trained model path for ModelSelector')
    # Test options
    parser.add_argument('--test', action='store_true', 
                        default=TEST_FLAG, help='Run test mode')
    parser.add_argument('--modelBasePath', required=False, type=str, metavar='PATH', 
                        default=MODEL_BASE_DIR, help='ModelBase directory path')
    parser.add_argument('--template', required=False, type=str, metavar='PATH', 
                        default=TMP_NAME, help='Object template file name')
    # Device settings
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device')
    # Program flag, do not implement
    parser.add_argument('--device', type=str, 
                        default=None, help='torch network device')# do not implement
    
    return parser.parse_args()


def Parser_PCR():
    parser = argparse.ArgumentParser(description='PCR')
    # Required arguments
    parser.add_argument('-o', '--objectDIR',    required=False, type=str, metavar='PATH', 
                        default=OBJ_DIR, help='objects directory')
    parser.add_argument('-t', '--template',     required=False, type=str, metavar='PATH', 
                        default=TMP_NAME, help='Object template file name')
    parser.add_argument('-b', '--modelBasePath',required=False, type=str, metavar='PATH', 
                        default=MODEL_BASE_DIR, help='ModelBase directory path')
    parser.add_argument('-m', '--modelPath',    required=False, type=str, metavar='PATH', 
                        default=DCP_MODEL_PATH, help='Pre-trained model path for DCP')
    parser.add_argument('-p', '--method',    required=False, type=str, metavar='METHOD', 
                        default=PCR_METHOD, help='PCR method (dcp_icp or icp_4)')
    # Test options
    parser.add_argument('--test', action='store_true', 
                        default=TEST_FLAG, help='Run test mode')
    parser.add_argument('--transform', required=False, type=str, metavar='PATH', 
                        default=TRANS_NAME, help='Object transformed file name')
    # Device settings
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device')
    # Program flag, do not implement
    parser.add_argument('--device', type=str, 
                        default=None, help='torch network device')# do not implement
    
    return parser.parse_args()


def initDevice(args):
    if (not args.cuda or not torch.cuda.is_available()):
        device = torch.device('cpu')
        args.cuda = False
    elif (torch.device(args.cudaDevice)):
        device = torch.device(args.cudaDevice)
        torch.cuda.set_device(device.index)
    else:
        device = torch.device('cpu')
        args.cuda = False
    args.device = device
    return args

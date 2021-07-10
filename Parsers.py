# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:18:01 2021

@author: cocob
"""


import argparse
import torch


def Parser_ModelBase():
    parser = argparse.ArgumentParser(description='ModelBaseTools')
    # Required arguments
    parser.add_argument('-d', '--dataset',      required=False, type=str, metavar='PATH', 
                        default='/home/wei/Desktop/datasets/ModelNet40_Base', help='ModelBase dataset path')
    parser.add_argument('-m', '--modelPath',    required=False, type=str, metavar='PATH', 
                        default='/home/wei/Desktop/pointnet2Comp2_best.pth', help='Pre-trained model path for ModelSelector')
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
    parser = argparse.ArgumentParser(description='ModelBaseTools')
    # Required arguments
    parser.add_argument('-o', '--objectDIR',        required=False, type=str, metavar='PATH', 
                        default='objects', help='objects directory')
    parser.add_argument('-f', '--modelFeature',     required=False, type=str, metavar='PATH', 
                        default='features/features_g.pkl', help='ModelBase feature path')
    parser.add_argument('-b', '--modelBasePath',    required=False, type=str, metavar='PATH', 
                        default='D:/Datasets/ModelNet40_Base', help='ModelBase directory path')
    parser.add_argument('-m', '--modelPath',        required=False, type=str, metavar='PATH', 
                        default='/home/wei/Desktop/pointnet2Comp2_best.pth', help='Pre-trained model path for ModelSelector')
    parser.add_argument('--test', action='store_true', 
                        default=False, help='Run test mode')
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

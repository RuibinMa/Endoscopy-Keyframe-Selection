#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
from shutil import rmtree
import cv2
from DoubleList import DoubleList
from numpy.matlib import repmat
from AnalyzeScore_subsequence_function import analyze_score

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
def warpFlow(img, flow): # this assumes the two images are of the same size; warping 1 -> 0
    warpped = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            nx = int(x + flow[x,y,1])
            if(nx < 0 or nx >= img.shape[0]):
                continue
            ny = int(y + flow[x,y,0])
            if(ny < 0 or ny >= img.shape[1]):
                continue
            
            if(len(warpped.shape) < 3):
                warpped[x,y] = img[nx, ny]
            else:
                warpped[x,y,:] = img[nx,ny,:]
    return warpped
def warp_NCC(img0, img1, flow):
    patch0 = []
    patch1 = []
    for x in range(img0.shape[0]):
        for y in range(img0.shape[1]):
            nx = int(x + flow[x,y,1])
            if(nx < 0 or nx >= img1.shape[0]):
                continue
            ny = int(y + flow[x,y,0])
            if(ny < 0 or ny >= img1.shape[1]):
                continue
            
            patch0.append(img0[x,y])
            patch1.append(img1[nx,ny])
    
    return correlation_coefficient(np.asarray(patch0), np.asarray(patch1))           
    

def motion_analysis_flownet2(args):
    data_base_dir = args.dir
    folder = data_base_dir + 'classifiedgood/'
    outputfolder = data_base_dir + 'opticalflow2/'
    if os.path.isdir(outputfolder):
        rmtree(outputfolder)
    os.mkdir(outputfolder)
    
    # sample two images to initialize the blob dimensions
    imglist = os.listdir(folder)
    imglist.sort()
    img0 = misc.imread(folder + imglist[0]);
    img1 = misc.imread(folder + imglist[1]);
    # prepare x axis and y axis for faster computation
    x_axis = np.arange(0, img0.shape[0], dtype=np.int)
    x_axis = repmat(x_axis[:, np.newaxis], 1, img0.shape[1])
    y_axis = np.arange(0, img0.shape[1], dtype=np.int)
    y_axis = repmat(y_axis, img0.shape[0], 1)
    
    def warp_NCC_opt(img0, img1, flow):
        nx = (x_axis + flow[:,:,1]).astype(int)
        ny = (y_axis + flow[:,:,0]).astype(int)
        pos0 = (nx <= img1.shape[0]-1) & (nx >= 0) & (ny <= img1.shape[1]-1) & (ny >= 0)
        patch0 = img0[pos0]
        pos1x = nx[pos0]
        pos1y = ny[pos0]
        nn = [pos1x, pos1y]
        patch1 = img1[nn]
        return correlation_coefficient(patch0, patch1)
    
    
    input_data = []
    if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    print ('input blob 0 :'),
    print (input_data[0].shape)
    if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    print ('input blob 1 :'),
    print (input_data[1].shape)
    
    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height
    
    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
    
    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
    
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
    
    proto = open(args.deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
    
        tmp.write(line)
        
    tmp.flush()
    
    if not args.verbose:
        caffe.set_logging_disabled()
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
    
    # calculate sharpness and sort by sharpness
    sharpness = []
    print ('Calculating sharpness ...')
    dl = DoubleList()
    D = {}
    for f in imglist:
        dl.append(f)
        D[f] = dl.tail
    for i in range(len(imglist)):
        print ('%d / %d'%(i, len(imglist)))
        img = misc.imread(folder + imglist[i], flatten=True)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        sharpness.append((np.sum(sobelx ** 2) + np.sum(sobely ** 2)) / img.size)
    order = np.argsort(sharpness)
    ## load img0 and img1 iteratively from folder
    
    score = []
    keyframes = []
    for j in range(len(imglist)):
        print ('Homography estimation: %d / %d   sharpness = %.2f'%(j, len(imglist), sharpness[order[j]]))
        if order[j] == 0 or order[j] == len(imglist)-1:
            keyframes.append(imglist[order[j]])
            score.append(0.9)
            continue
        ID = order[j]
        PrevName = D[imglist[ID]].prev.data
        NextName = D[imglist[ID]].next.data
        Prev = misc.imread(folder + PrevName, mode='RGB')
        Next = misc.imread(folder + NextName, mode='RGB')
        input_dict = {}
        input_data = []
        if len(Prev.shape) < 3: input_data.append(Prev[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(Prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        if len(Next.shape) < 3: input_data.append(Next[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(Next[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        input_dict[net.inputs[0]] = input_data[0]
        input_dict[net.inputs[1]] = input_data[1]
        
        
        print ('Network forward pass using %s and %s' % (PrevName, NextName))
        i = 1
        while i<=5:
            i+=1
        
            net.forward(**input_dict)
        
            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()
        
                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True
        
            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')
    
        flow = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        #s = warp_NCC(img0=rgb2gray(Prev), img1=rgb2gray(Next), flow=flow)
        s = warp_NCC_opt(img0=rgb2gray(Prev), img1=rgb2gray(Next), flow=flow)
        #warpped = warpFlow(rgb2gray(Next), flow)
        #s = correlation_coefficient(warpped, rgb2gray(Prev))
        print ('correlation coefficient = %.4f' % s)
        if s > 0.98:
            dl.remove_byaddress(D[imglist[ID]])
            print ('Redundant: ' + imglist[ID])
        else:
            keyframes.append(imglist[order[j]])
            score.append(s)
            print ('Keyframe:  ' + imglist[ID])
    
    ordkeyframes = np.argsort(keyframes)
    keyframes.sort()
    score = np.asarray(score)
    score = score[np.asarray(ordkeyframes)]
     
    offilename = data_base_dir + 'opticalflowscore2.txt'
    if os.path.exists(offilename):
        os.remove(offilename)
    offile = open(offilename, 'w')
    for i in range(len(keyframes)):
        offile.write('%s %.4f\n' % (keyframes[i], score[i]))
        os.system('cp ' + folder + keyframes[i] + ' ' + outputfolder + keyframes[i])
    
    analyze_score(data_base_dir=data_base_dir, fname='opticalflowscore2', imgnames=keyframes, score=score)
    
    del net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffemodel', help='path to model', 
                        default = '/playpen/software/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5')
    parser.add_argument('--deployproto', help='path to deploy prototxt template',
                        default = '/playpen/software/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template')
    parser.add_argument('--dir', help='path to files', default = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/')
    
    parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
    #parser.add_argument('--warp',  help='whether create warpped images', default=False, type=bool)
    parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
    
    args = parser.parse_args()
    
    if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
    if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
    
    motion_analysis_flownet2(args)
#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import time

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('img0', help='image 0 path')
parser.add_argument('img1', help='image 1 path')
parser.add_argument('out',  help='output filename')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
if(not os.path.exists(args.img0)): raise BaseException('img0 does not exist: '+args.img0)
if(not os.path.exists(args.img1)): raise BaseException('img1 does not exist: '+args.img1)

num_blobs = 2
input_data = []


img0 = misc.imread(args.img0)
img0 = img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
img1 = misc.imread(args.img1)
img1 = img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
batchsize = 1;
batch0 = img0;
batch1 = img1;
for i in range(batchsize-1):
    batch0 = np.concatenate((batch0, img0))
    batch1 = np.concatenate((batch1, img1))
input_data.append(batch0)
print(input_data[-1].shape)
input_data.append(batch1)
print(input_data[-1].shape)

width = input_data[0].shape[3]
height = input_data[0].shape[2]
vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height
vars['BATCH_SIZE'] = batchsize

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

input_dict = {}
for blob_idx in range(num_blobs):
    input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

#
# There is some non-deterministic nan-bug in caffe
# it seems to be a race-condition 
#
print('Network forward pass using %s.' % args.caffemodel)
i = 1
while i<=5:
    i+=1
    
    start_time = time.time()
    net.forward(**input_dict)
    print ("One Forward time = %.2f second"%(time.time()- start_time))
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

flow0 = net.blobs['predict_flow_final'].data[0, ...].transpose(1,2,0)

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


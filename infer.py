# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:57:39 2016

@author: wrlife
"""

import numpy as np
from PIL import Image

import os
from shutil import rmtree
import caffe
import time
from pickimagenet import *
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe


#======================
#Load net parames
#======================
style_weights="./model/weights.pretrained.caffemodel"
test_net = caffe.Net(style_net(train=False,learn_all=False),style_weights, caffe.TEST)
test_net.forward()


MEAN_FILE=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto'
Mean_blob = caffe.proto.caffe_pb2.BlobProto ()
Mean_blob.ParseFromString (open (MEAN_FILE,'rb').read ())
# will mean blob to numpy.array
Mean_npy = np.array(caffe.io.blobproto_to_array (Mean_blob))[0]

Mean_npy = Mean_npy.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', Mean_npy)


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', Mean_npy)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

atup=('bad','good')
style_labels=list(atup)

data_base_dir = "/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_004/"
data_dir= data_base_dir + "/images-raw/"

if os.path.isdir(data_base_dir + 'classifiedgood'):
    rmtree(data_base_dir + 'classifiedgood')
os.mkdir(data_base_dir + 'classifiedgood')

if os.path.isdir(data_base_dir + 'classifiedbad'):
    rmtree(data_base_dir + 'classifiedbad')
os.mkdir(data_base_dir + 'classifiedbad')

count=1

computingtime=np.zeros(9617)

for imfile in os.listdir(data_dir):
    if imfile.endswith(".jpg"):
        im = Image.open(data_dir+imfile)
        im = np.array(im, dtype=np.float32)
        print im.shape
        
        
        transformed_image = transformer.preprocess('data', im)
        t=disp_preds(test_net, transformed_image, style_labels, k=2, name='style')
        
        if t[0]==1:
            os.system("cp "+data_dir+imfile+" "+data_base_dir+"classifiedgood")
        else:
            os.system("cp "+data_dir+imfile+" "+data_base_dir+"classifiedbad")

        print "Processed file:", count; count=count+1



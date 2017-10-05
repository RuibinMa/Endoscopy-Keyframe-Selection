'''
Created on Oct 5, 2017

@author: ruibinma
'''
from os import listdir
import os
import cv2
import numpy as np
from Homography import readImage
from shutil import rmtree
from pyclustering.cluster.xmeans import xmeans
from pyclustering.utils import draw_clusters

folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
#folder = '/home/ruibinma/Desktop/keyframes/'
fname = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes.txt'
imglist = []
with open(fname, 'r') as ins:
    for line in ins:
        imglist.append(line.rstrip('\n'))

features = []
cm0 = 0.
cm1 = 0.
cm2 = 0.
for imgname in imglist:
    img = cv2.imread(folder + imgname, 1)
    #print img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #print img.shape
    c0 = np.mean(img[:,:,0])
    c1 = np.mean(img[:,:,1])
    c2 = np.mean(img[:,:,2])
    
    features.append([c0])
    #features.append([c0, c1, c2])
    cm0 = cm0 + c0
    cm1 = cm1 + c1
    cm2 = cm2 + c2

cm0 = cm0 / len(features)
cm1 = cm1 / len(features)
cm2 = cm2 / len(features)
cm = [cm0]
#cm = [cm0, cm1, cm2]
print cm

#features = [[0.,0.],[1.,1.]]
#cm = [0.5, 0.5]
xmeans_instance = xmeans(features, [cm], 5, ccore=False)
xmeans_instance.process()
draw_clusters(features, xmeans_instance.get_clusters())
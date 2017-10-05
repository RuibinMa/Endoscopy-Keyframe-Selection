from os import listdir
import os
import cv2
import numpy as np
from Homography import readImage
from shutil import rmtree
from pyclustering.cluster.xmeans import xmeans
from pyclustering.utils import draw_clusters
import matplotlib.pyplot as plt
from pylab import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
#folder = '/home/ruibinma/Desktop/keyframes/'
fname = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes.txt'
imglist = []
with open(fname, 'r') as ins:
    for line in ins:
        imglist.append(line.rstrip('\n'))

imgname1 = imglist[100]
img1 = imread(folder + imgname1)
imgname2 = imglist[200]
img2 = imread(folder + imgname2)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.imshow(img1)
ax2.imshow(img2)
flow = cv2.calcOpticalFlowFarneback(rgb2gray(img1), rgb2gray(img2), None, 0.5, 3, 15, 3, 5, 1.2, 0)
warpped = zeros_like(img1)
for x in range(img1.shape[0]):
    for y in range(img2.shape[1]):
        nx = int(x + flow[x,y,0])
        if(nx < 0 or nx >= img2.shape[0]):
            continue
        ny = int(y + flow[x,y,0])
        if(ny < 0 or ny >= img2.shape[1]):
            continue
        
        warpped[x,y,:] = img2[nx,ny,:]
ax3.imshow(warpped)
plt.show()
        
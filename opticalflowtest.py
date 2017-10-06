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

#folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
base_folder = '/home/ruibinma/throat/004/'
folder = base_folder + 'images-raw/'
fname = base_folder + 'keyframes.txt'
imglist = []
with open(fname, 'r') as ins:
    for line in ins:
        imglist.append(line.rstrip('\n'))

imglist.sort()
#imgname1 = imglist[100]
imgname1 = 'frame3485.jpg'
imgname2 = 'frame3490.jpg'
img1 = imread(folder + imgname1)
#imgname2 = imglist[150]
img2 = imread(folder + imgname2)
print imgname1
print imgname2
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
ax1.imshow(img1)
ax2.imshow(img2)
flow = cv2.calcOpticalFlowFarneback(rgb2gray(img1), rgb2gray(img2), None, 0.5, 3, 15, 3, 5, 1.2, 0)
score = (np.sum(np.absolute(flow[..., 0])) + np.sum(np.absolute(flow[..., 1]))) / rgb2gray(img1).size

print score
warpped = zeros_like(img1)
for x in range(img2.shape[0]):
    for y in range(img2.shape[1]):
        nx = int(x - flow[x,y,1])
        if(nx < 0 or nx >= img1.shape[0]):
            continue
        ny = int(y - flow[x,y,0])
        if(ny < 0 or ny >= img1.shape[1]):
            continue
        
        warpped[x,y,:] = img1[nx,ny,:]
ax3.imshow(warpped)

mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros_like(img1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
ax4.imshow(rgb)
plt.show()
        
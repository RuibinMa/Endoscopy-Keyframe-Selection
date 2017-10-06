'''
Created on Sep 26, 2017

@author: ruibinma
'''
import cv2
import numpy as np
from os import listdir
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import minimum_filter1d
import os
from shutil import rmtree

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_004/classifiedgood/'
folder = '/home/ruibinma/throat/004/classifiedgood/'
#outputfolder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_004/keyframes/'
outputfolder = '/home/ruibinma/throat/004/keyframes/'
filelist = listdir(folder)
filelist.sort()

if os.path.isdir(outputfolder):
    rmtree(outputfolder)
os.mkdir(outputfolder)

prev = rgb2gray(mpimg.imread(folder+filelist[0]))
score = np.empty([len(filelist)])
score[0] = 0;
for i in range(1, len(filelist)):
    print ('image  %d / %d'%(i,len(filelist)))
    cur = rgb2gray(mpimg.imread(folder+filelist[i]))
    flow = cv2.calcOpticalFlowFarneback(prev, cur, 0.5, 3, 15, 3, 5, 1.2, 0)
    score[i] = np.sum(np.absolute(flow[..., 0])) + np.sum(np.absolute(flow[..., 1]))
    prev = cur
    
#plt.figure()
#plt.plot(score)
#plt.show()

localminima = []
localmin = minimum_filter1d(score, size=3)
for i in range(len(score)):
    if localmin[i] == score[i]:
        print 'keyframe  '+filelist[i]
        #os.system('cp '+ folder + filelist[i] + ' ' + outputfolder + filelist[i])
        localminima.append(filelist[i])
   
thefile = open('opticalflowlocalminima.txt', 'w')
for i in localminima:
    thefile.write("%s\n" % i)     
    
    
    
    

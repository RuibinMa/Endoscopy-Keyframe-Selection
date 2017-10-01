'''
Created on Sep 27, 2017

@author: ruibinma
'''
import cv2
import numpy as np
from os import listdir
#from scipy.ndimage.filters import minimum_filter1d
import os
from Homography import EstimateHomography
from Homography import readImage
from DoubleList import DoubleList
from shutil import rmtree

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

#folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/classifiedgood/'
folder = '/home/ruibinma/Desktop/classifiedgood/'
#outputfolder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
outputfolder = '/home/ruibinma/Desktop/keyframes/'

use_optical_flow = False
if use_optical_flow:
    filelist = []
    with open('./opticalflowlocalminima.txt', 'r') as ins:
        for line in ins:
            filelist.append(line.rstrip('\n'))
else:
    filelist = listdir(folder)

filelist.sort()

if os.path.isdir(outputfolder):
    rmtree(outputfolder)
os.mkdir(outputfolder)

#create double link to store file names

dl = DoubleList()
D = {}
for f in filelist:
    dl.append(f)
    D[f] = dl.tail

sharpness = np.zeros([len(filelist)])
score = np.ones([len(filelist)])

for i in range(len(filelist)):
    img = readImage(folder + filelist[i])
    print img.shape
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    
    sharpness[i] = (np.sum(sobelx ** 2) + np.sum(sobely ** 2)) / img.size
    print filelist[i] + ' Sharpness = %.4f'%sharpness[i]
order = np.argsort(sharpness)

keyframes = []

for i in range(len(order)):
    print '%d / %d   '%(i, len(filelist))
    if order[i] == 0 or order[i] == len(filelist)-1:
        keyframes.append(filelist[order[i]])
    else:
        ID = order[i]
        PrevName = D[filelist[ID]].prev.data
        NextName = D[filelist[ID]].next.data
        print 'Processing: ' + PrevName + ' and ' + NextName
        Prev = readImage(folder + PrevName)
        Next = readImage(folder + NextName)
        H = EstimateHomography(estimation_thresh=0.6, img1=Prev, img2=Next, use_builtin_ransac=True)
        warpped = cv2.warpPerspective(Prev, H, (Next.shape[1], Next.shape[0]))
        score[i] = correlation_coefficient(warpped, Next)
        if score[i] > 0.9:
            dl.remove_byaddress(D[filelist[ID]])
            print 'Redundant: ' + filelist[ID]
        else:
            keyframes.append(filelist[order[i]])
            print 'Keyframe:  ' + filelist[ID]

for i in keyframes:
    os.system('cp ' + folder + i + ' ' + outputfolder + i)

thefile = open('homographykeyframes.txt', 'w')
for i in keyframes:
    thefile.write("%s\n" % i)

    
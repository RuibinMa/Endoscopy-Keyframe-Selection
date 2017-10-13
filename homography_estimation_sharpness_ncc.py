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
from scipy import misc

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
def warp_homography_NCC(img0, img1, H):
    patch0 = []
    patch1 = []
    for x in range(img0.shape[0]):
        for y in range(img0.shape[1]):
            coord = np.array([[x],[y],[1]])
            ncoord = np.matmul(H, coord).flat
            nx = int(ncoord[0])
            ny = int(ncoord[1])
            if(nx < 0 or nx >= img1.shape[0] or ny < 0 or ny >= img1.shape[1]):
                continue
            
            patch0.append(img0[x,y])
            patch1.append(img1[nx,ny])
    
    return correlation_coefficient(np.asarray(patch0), np.asarray(patch1))  

folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/classifiedgood/'
#folder = '/home/ruibinma/Desktop/classifiedgood/'
outputfolder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
#outputfolder = '/home/ruibinma/Desktop/keyframes/'

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
print ('Calculating sharpness ...')
dl = DoubleList()
D = {}
for f in filelist:
    dl.append(f)
    D[f] = dl.tail

sharpness = np.zeros([len(filelist)])
score = np.ones([len(filelist)])

for i in range(len(filelist)):
    print ('%d / %d'%(i, len(filelist)))
    img = readImage(folder + filelist[i])
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    
    sharpness[i] = (np.sum(sobelx ** 2) + np.sum(sobely ** 2)) / img.size
order = np.argsort(sharpness)

keyframes = []

for i in range(len(order)):
    print ('Homography estimation: %d / %d   sharpness = %.2f'%(i, len(filelist), sharpness[order[i]]))
    if order[i] == 0 or order[i] == len(filelist)-1:
        keyframes.append(filelist[order[i]])
    else:
        ID = order[i]
        PrevName = D[filelist[ID]].prev.data
        NextName = D[filelist[ID]].next.data
        print 'Processing: ' + PrevName + ' and ' + NextName,
        Prev = readImage(folder + PrevName)
        Next = readImage(folder + NextName)
        H = EstimateHomography(img1=Prev, img2=Next, use_builtin_ransac=True)
        #score[i] = warp_homography_NCC(img0=Prev, img1=Next, H=H)
        warpped = cv2.warpPerspective(Prev, H, (Next.shape[1], Next.shape[0]))
        misc.imsave(outputfolder + filelist[ID], warpped)
        score[i] = correlation_coefficient(warpped, Next)
        print warpped.shape
        print '        score = %.4f'%score[i]
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

    
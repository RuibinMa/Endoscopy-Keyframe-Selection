'''
Created on Oct 5, 2017

@author: ruibinma
'''
from os import listdir
import os
import cv2
import random
import numpy as np
from Homography import readImage
from shutil import rmtree
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import draw_clusters

base_folder = '/home/ruibinma/throat/003/'
folder = base_folder + 'keyframes/'
fname = base_folder + 'keyframes.txt'
imglist = []
with open(fname, 'r') as ins:
    for line in ins:
        imglist.append(line.rstrip('\n'))

features = []
N = 10
D = 3
K = 10
for imgname in imglist:
    img = cv2.imread(folder + imgname, 1)
    #print img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #print img.shape
    feature = []
    for dim in range(D):
        for i in range(1, N):
            #print 100*float(i)/float(N)
            feature.append(np.percentile(img[:,:,dim].flat, 100*float(i)/float(N)))
    #features.append([c0])
    features.append(feature)


#cm = [[128.0 for i in range(D*(N-1))]]
cm = [[float(random.randint(0, 255)) for i in range(D * (N - 1))] for _ in range(K)]
#cm = [[128.0 for i in range(D * (N - 1))] for _ in range(K)]
print cm
print len(cm)

#features = [[0.,0.],[1.,1.]]
#cm = [0.5, 0.5]
#print features
kmeans_instance = kmeans(features, cm)
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
#xmeans_instance = xmeans(features, cm, 5, ccore=False)
#xmeans_instance.process()
#clusters = xmeans_instance.get_clusters()
print len(clusters)
#draw_clusters(features, xmeans_instance.get_clusters())

for i in range(len(clusters)):
    subfolder = base_folder + 'cluster' + str(i)
    if os.path.isdir(subfolder):
        rmtree(subfolder)
    os.mkdir(subfolder)
    for item in clusters[i]:
        os.system('cp ' + base_folder + 'images-raw/' + imglist[item] + ' ' + subfolder)

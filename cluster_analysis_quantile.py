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

#base_folder = '/home/ruibinma/throat/003/'
base_folder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/'
#os.system('rm -r ' + base_folder + 'cluster*')

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

features = np.asarray(features, dtype = np.float32)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(features, K, None, criteria, 10, flags)
labels = np.asarray(np.squeeze(labels), dtype=int)
ids = np.asarray(range(len(imglist)), dtype=int)
#kmeans_instance = kmeans(features, cm)
#kmeans_instance.process()
#clusters = kmeans_instance.get_clusters()
#xmeans_instance = xmeans(features, cm, 5, ccore=False)
#xmeans_instance.process()
#clusters = xmeans_instance.get_clusters()
#print len(clusters)
#draw_clusters(features, xmeans_instance.get_clusters())

count = 0
minLength = 50
for i in range(max(labels)+1):
    cluster = ids[labels == i]
    if len(cluster) < minLength:
        continue
    
    subfolder = base_folder + 'cluster' + str(count)
    if os.path.isdir(subfolder):
        rmtree(subfolder)
    os.mkdir(subfolder)
    
    for item in cluster:
        os.system('cp ' + base_folder + 'images-raw/' + imglist[item] + ' ' + subfolder)
    
    count = count + 1

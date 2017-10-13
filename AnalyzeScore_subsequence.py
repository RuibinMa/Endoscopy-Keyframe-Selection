'''
Created on Oct 3, 2017

@author: ruibinma
'''
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from shutil import rmtree
from scipy.ndimage.filters import minimum_filter1d
from scipy.ndimage.filters import maximum_filter1d

score = []
imgnames = []
data_base_dir = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_004/'
fname = data_base_dir + 'opticalflowscore2.txt'
#fname = '/home/ruibinma/throat/004/opticalflowscore.txt'
with open(fname, 'r') as ins:
    for line in ins:
        pair = line.split()
        imgname = pair[0]
        imgnames.append(imgname)
        score.append(float(pair[1]))


folder = data_base_dir + 'images-raw/'
outputfolder = data_base_dir + 'keyframes/'

if os.path.isdir(outputfolder):
    rmtree(outputfolder)
os.mkdir(outputfolder)

#for i in range(len(imgnames)):
#    os.system('cp ' + folder + imgnames[i] + ' ' + outputfolder + imgnames[i])
#exit()

scoreminima = minimum_filter1d(input=score, size=4)
localminima = []
keyframes = []
keyframeids = []
keyframesfilename = data_base_dir + 'keyframes.txt'
if os.path.exists(keyframesfilename):
    os.remove(keyframesfilename)
keyframesfile = open(keyframesfilename, 'w')

#for i in range(len(score)):
#    if score[i] == scoreminima[i]:
#        localminima.append(imgnames[i])
#        #os.system('cp ' + folder + imgnames[i] + ' ' + outputfolder + imgnames[i])
#        keyframesfile.write('%s\n' % imgnames[i])
#        keyframes.append(imgnames[i])
#        keyframeids.append(i)

for i in range(len(score)):
    #localminima.append(imgnames[i])
    #os.system('cp ' + folder + imgnames[i] + ' ' + outputfolder + imgnames[i])
    keyframesfile.write('%s\n' % imgnames[i])
    keyframes.append(imgnames[i])
    keyframeids.append(i)

#print('%d frames selected' % len(localminima))

# boundary detection

boundaries = []
ignore_ends = True
#plt.figure()
#------------------------------------ boundary detection method 1:
#threshold = np.percentile(score, 99)
#scoreextrema = minimum_filter1d(input=score, size= len(score) / 5)
#scoreextrema = maximum_filter1d(input=score, size = 301)
#localextrema = []
#localextremaids = []
#for i in range(len(score)):
#    if score[i] == scoreextrema[i] and score[i] > threshold and score[i] > 5:
#        boundaries.append(imgnames[i])
#        localextrema.append(score[i])
#        localextremaids.append(i)

#print boundaries
#print localextrema
#print len(localextrema)

#plt.plot(score)
#plt.show()

#------------------------------------ boundary detection method 2:
if(ignore_ends):
    score[:int(len(score)*0.1)] = np.ones_like(score[:int(len(score)*0.1)]) 
    score[int(len(score)*0.9):] = np.ones_like(score[int(len(score)*0.9):])
    
threshold = np.percentile(score, 1)
#scoreextrema = minimum_filter1d(input=score, size= len(score) / 5)
scoreextrema = minimum_filter1d(input=score, size = 301)
localextrema = []
localextremaids = []
for i in range(len(score)):
    if score[i] == scoreextrema[i] and score[i] < threshold and score[i] < 0.8:
        boundaries.append(imgnames[i])
        localextrema.append(score[i])
        localextremaids.append(i)

#plt.plot(score)
#plt.show()

#------------------------------------ boundary detection method 3:
#acc_motions = []
#for i in range(len(keyframeids)):
#    acc_motion = 0
#    if(i == 0):
#        acc_motion = score[keyframeids[i]]
#    else:
#        acc_motion = np.sum(score[keyframeids[i-1] + 1 : keyframeids[i] + 1])
#    acc_motions.append(acc_motion)
#if(ignore_ends):
#    acc_motions[:int(len(acc_motions)*0.1)] = np.zeros_like(acc_motions[:int(len(acc_motions)*0.1)]) 
#    acc_motions[int(len(acc_motions)*0.9):] = np.zeros_like(acc_motions[int(len(acc_motions)*0.9):])
#    
#threshold = np.percentile(acc_motions, 99)
#localextrema = []
#localextremaids = []
#acc_motion_score_extrema = maximum_filter1d(input=acc_motions, size=301)
#for i in range(len(acc_motions)):
#    #if(keyframes[i] == 'frame3773.jpg'):
#    #    print(acc_motions[i])
#    if acc_motions[i] == acc_motion_score_extrema[i] and acc_motions[i] > threshold:
#        boundaries.append(keyframes[i])
#        localextrema.append(acc_motions[i])
#        localextremaids.append(i)
#plt.plot(acc_motions)        

# ----------------------------------- boundary detection results:
#plt.plot(localextremaids, localextrema, 'ro')
print 'number of boundaries = %d' % len(localextrema)
print 'boundaries :',
print boundaries
print 'boundaries motion values: ',
print localextrema

#plt.show()

#------------------------------------ split video by boundaries
shots = []
boundaryid = 0
shot = []
for i in range(len(keyframes)):
    if boundaryid >= len(boundaries):
        shot.append(keyframes[i])
        continue
    
    if keyframes[i] < boundaries[boundaryid]:
        shot.append(keyframes[i])
    else:
        shots.append(shot)
        shot = []
        shot.append(keyframes[i])
        boundaryid = boundaryid + 1
shots.append(shot)

#print shots
print 'number of shots = %d' % len(shots)
subid = 0
min_shot_length = 40
print 'minimum shot length threshold = %d' % min_shot_length
for i in range(len(shots)):
    shot = shots[i]
    print 'shot %d: %d frames' %(i, len(shot)),
    if len(shot) > min_shot_length:
        subfolder = outputfolder + str(subid) + '/'
        os.mkdir(subfolder)
        for j in shot:
            os.system('cp ' + folder + j + ' ' + subfolder + j)
        print ' '
        subid = subid + 1
    else:
        print '     discarded'
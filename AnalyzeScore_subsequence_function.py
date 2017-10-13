'''
Created on Oct 3, 2017

@author: ruibinma
'''
import numpy as np
import os
from shutil import rmtree
from scipy.ndimage.filters import minimum_filter1d

def analyze_score(data_base_dir='/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_004/', fname='opticalflowscore2.txt',
                  imgnames=None, score=None):
    print 'Performing boundary detection ... '
    if imgnames is None:
        assert(score is None);
        score = []
        imgnames = []
        fname=data_base_dir + fname
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

    keyframes = []
    keyframeids = []
    keyframesfilename = data_base_dir + 'keyframes.txt'
    if os.path.exists(keyframesfilename):
        os.remove(keyframesfilename)
    keyframesfile = open(keyframesfilename, 'w')
    
    for i in range(len(score)):
        #os.system('cp ' + folder + imgnames[i] + ' ' + outputfolder + imgnames[i])
        keyframesfile.write('%s\n' % imgnames[i])
        keyframes.append(imgnames[i])
        keyframeids.append(i)
    
    # boundary detection
    
    boundaries = []
    ignore_ends = True  
    #------------------------------------ boundary detection method 2:
    if(ignore_ends):
        score[:int(len(score)*0.1)] = np.ones_like(score[:int(len(score)*0.1)]) 
        score[int(len(score)*0.9):] = np.ones_like(score[int(len(score)*0.9):])
        
    threshold = np.percentile(score, 1)
    scoreextrema = minimum_filter1d(input=score, size = 301)
    localextrema = []
    localextremaids = []
    for i in range(len(score)):
        if score[i] == scoreextrema[i] and score[i] < threshold and score[i] < 0.8:
            boundaries.append(imgnames[i])
            localextrema.append(score[i])
            localextremaids.append(i)  
    
    # ----------------------------------- boundary detection results:
    #plt.plot(localextremaids, localextrema, 'ro')
    print 'number of boundaries = %d' % len(localextrema)
    print 'boundaries :',
    print boundaries
    print 'boundaries motion values: ',
    print localextrema
    
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

if __name__ == '__main__':
    analyze_score()
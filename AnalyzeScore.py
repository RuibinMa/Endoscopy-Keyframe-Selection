'''
Created on Sep 27, 2017

@author: ruibinma
'''
from os import listdir
import os
from sets import Set
from shutil import rmtree

folder = '/home/ruibinma/Desktop/classifiedgood/'
#outputfolder = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/keyframes/'
outputfolder = '/home/ruibinma/Desktop/keyframes/'

if os.path.isdir(outputfolder):
    rmtree(outputfolder)
os.mkdir(outputfolder)
fname1 = '/home/ruibinma/Desktop/HomographyEstimation/opticalflowlocalminima.txt'
fname2 = '/home/ruibinma/Desktop/HomographyEstimation/keyframes.txt'

set1 = Set()
set2 = Set()
with open(fname1, 'r') as ins:
    for line in ins:
        set1.add(line.rstrip('\n'))
with open(fname2, 'r') as ins:
    for line in ins:
        set2.add(line.rstrip('\n'))

set3 = set1 & set2
set4 = set1 | set2
print len(set1)
print len(set2)
print len(set3)
print len(set4)

f = open('intersection.txt', 'w')
for k in set3:
    f.write('%s\n'%k)
    os.system('cp ' + folder + k + ' ' + outputfolder + k)
    




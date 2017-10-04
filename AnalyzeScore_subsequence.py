'''
Created on Oct 3, 2017

@author: ruibinma
'''
import numpy as np
from dask.array.ghost import boundaries
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import minimum_filter1d
from scipy.ndimage.filters import maximum_filter1d

score = []
imgnames = []
fname = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/opticalflowscore.txt'
with open(fname, 'r') as ins:
    for line in ins:
        pair = line.split()
        imgname = pair[0]
        imgnames.append(imgname)
        score.append(float(pair[1]))


threshold = np.percentile(score, 95)
#scoreextrema = minimum_filter1d(input=score, size= len(score) / 5)
scoreextrema = maximum_filter1d(input=score, size = 301)
boundaries = []
localextrema = []
localextremaids = []
for i in range(len(score)):
    if score[i] == scoreextrema[i] and score[i] > threshold and score[i] > 5:
        boundaries.append(imgnames[i])
        localextrema.append(score[i])
        localextremaids.append(i)

#print localextremaimg
#print localextrema
#print len(localextrema)



plt.figure()
plt.plot(score)
plt.plot(localextremaids, localextrema, 'ro')
plt.show()

from Homography import EstimateHomography
from Homography import readImage
imgname1 = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/images-raw/frame4164.jpg'
imgname2 = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/images-raw/frame4175.jpg'
img1 = readImage(imgname1)
img2 = readImage(imgname2)

H = EstimateHomography(estimation_thresh=0.6, img1=img1, img2=img2, use_builtin_ransac=True)
warpped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

score = correlation_coefficient(warpped, img2)
print score
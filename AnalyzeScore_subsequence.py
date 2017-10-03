'''
Created on Oct 3, 2017

@author: ruibinma
'''
import numpy as np
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

score = []
fname = '/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/subsequences.txt'
with open(fname, 'r') as ins:
    for line in ins:
        pair = line.split()
        imgname = pair[0]
        score.append(float(pair[1]))

plt.figure()
plt.plot(score)
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
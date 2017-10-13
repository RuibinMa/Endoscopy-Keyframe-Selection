# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:00:00 2017


@author: ruibinma
"""

from PIL import Image

import cv2
from shutil import rmtree
from pickimagenet import *

from Homography import EstimateHomography
from Homography import readImage
from DoubleList import DoubleList
from scipy.ndimage.filters import minimum_filter1d, maximum_filter1d

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

def main(args):
    data_base_dir = args.dir
    data_dir= data_base_dir + "images-raw/"
    outputfolder = data_base_dir + 'keyframes/'
    min_shot_length = args.min_shot_length
    nmins_window_size = args.nmins_window_size
    percentile_threshold = args.percentile_threshold
    absolute_threshold = args.absolute_threshold
    ignore_ends = args.ignore_ends
    
    if os.path.isdir(data_base_dir + 'classifiedgood'):
        rmtree(data_base_dir + 'classifiedgood')
    os.mkdir(data_base_dir + 'classifiedgood')
    if os.path.isdir(data_base_dir + 'classifiedbad'):
        rmtree(data_base_dir + 'classifiedbad')
    os.mkdir(data_base_dir + 'classifiedbad')
    if os.path.isdir(outputfolder):
        rmtree(outputfolder)
    os.mkdir(outputfolder)
    
    #======================
    #Load net parames
    #======================
    style_weights="./model/weights.pretrained.caffemodel"
    test_net = caffe.Net(style_net(train=False,learn_all=False),style_weights, caffe.TEST)
    test_net.forward()
    
    
    MEAN_FILE=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto'
    Mean_blob = caffe.proto.caffe_pb2.BlobProto ()
    Mean_blob.ParseFromString (open (MEAN_FILE,'rb').read ())
    # will mean blob to numpy.array
    Mean_npy = np.array(caffe.io.blobproto_to_array (Mean_blob))[0]
    
    Mean_npy = Mean_npy.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', Mean_npy)
    
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', Mean_npy)            # subtract the dataset-mean value in each channel
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    atup=('bad','good')
    style_labels=list(atup)
    
    classifiedgood = []
    opticalflowscore = []
    sharpness = []
    
    filelist = os.listdir(data_dir)
    filelist.sort()
    #filelist = filelist[3000:3100]
    count = 1
    prev_of = None
    
    # ImageNet classification and optical flow motion estimation
    for imfile in filelist:
        if imfile.endswith(".jpg"):
            print 'Classification and Motion Estimation: %d / %d'%(count, len(filelist))
            count = count + 1
            im = Image.open(data_dir+imfile)
            im = np.array(im, dtype=np.float32)
            transformed_image = transformer.preprocess('data', im)
            t=disp_preds(test_net, transformed_image, style_labels, k=2, name='style')
            
            if t[0]==1: # only do optical flow on frames classified as good
                classifiedgood.append(imfile)
                os.system("cp "+data_dir+imfile+" "+data_base_dir+"classifiedgood")
                print 'classified as good : ' + imfile
                
                if len(opticalflowscore) == 0:
                    opticalflowscore.append(0)
                    cur_of = rgb2gray(im)     
                else:
                    cur_of = rgb2gray(im)
                    flow = cv2.calcOpticalFlowFarneback(prev_of, cur_of, 0.5, 3, 15, 3, 5, 1.2, 0)
                    opticalflowscore.append((np.sum(np.absolute(flow[..., 0])) + np.sum(np.absolute(flow[..., 1]))) / cur_of.size)
                #if imfile == 'frame2967.jpg':
                #    print opticalflowscore[-1]
                #    temp = Image.open(data_dir + classifiedgood[-2])
                #    temp = np.array(temp, dtype=np.float32)
                #    temp = rgb2gray(temp)
                #    print temp - prev_of
                #    print imfile
                prev_of = cur_of
    # find the local minima of the optical flow motion estimation
    assert len(opticalflowscore) == len(classifiedgood)
    localminima = []
    localmin = minimum_filter1d(opticalflowscore, size=3)
    for i in range(len(opticalflowscore)):
        if localmin[i] == opticalflowscore[i]:
            print 'optical flow local minimum:  '+classifiedgood[i]
            #os.system('cp '+ folder + filelist[i] + ' ' + outputfolder + filelist[i])
            localminima.append(classifiedgood[i])
    
    # calculate sharpness
    print 'Calculating sharpness ...'
    dl = DoubleList()
    D = {}
    for f in localminima:
        dl.append(f)
        D[f] = dl.tail
    for i in range(len(localminima)):
        img = readImage(data_dir + localminima[i])
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        sharpness.append((np.sum(sobelx ** 2) + np.sum(sobely ** 2)) / img.size)
    order = np.argsort(sharpness)
    
    # use homography estimation to eliminate the redundant candidates
    keyframes = []
    score = []
    for i in range(len(order)):
        print 'Homography estimation: %d / %d   '%(i, len(localminima))
        if order[i] == 0 or order[i] == len(localminima)-1:
            keyframes.append(localminima[order[i]])
            score.append(0.9)
        else:
            ID = order[i]

            PrevName = D[localminima[ID]].prev.data
            NextName = D[localminima[ID]].next.data
            Prev = readImage(data_dir + PrevName)
            Next = readImage(data_dir + NextName)
            H = EstimateHomography(img1=Prev, img2=Next, use_builtin_ransac=True)
            warpped = cv2.warpPerspective(Prev, H, (Next.shape[1], Next.shape[0]))
            s = correlation_coefficient(warpped, Next)
            if s > 0.9:
                dl.remove_byaddress(D[localminima[ID]])
                print 'Redundant: ' + localminima[ID]
            else:
                keyframes.append(localminima[order[i]])
                score.append(s)
                print 'Keyframe:  ' + localminima[ID]
            #if localminima[ID] == 'frame3423.jpg':
            #    print PrevName
            #    print NextName
            #    print score[ID]
            #    print '-----------------------'
            
    
    ordkeyframes = np.argsort(keyframes)
    keyframes.sort()
    
    score = np.asarray(score)
    score = score[np.asarray(ordkeyframes)]   
    
    # key frame selection
    # optical flow boundary detection
    if ignore_ends:
        opticalflowscore[:int(len(opticalflowscore)*0.1)] = np.zeros_like(opticalflowscore[:int(len(opticalflowscore)*0.1)]) 
        opticalflowscore[int(len(opticalflowscore)*0.9):] = np.zeros_like(opticalflowscore[int(len(opticalflowscore)*0.9):])
    boundaries = []
    threshold = np.percentile(opticalflowscore, percentile_threshold)
    opticalextrema = maximum_filter1d(opticalflowscore, nmins_window_size)
    for i in range(len(opticalflowscore)):
        if opticalflowscore[i] == opticalextrema[i] and opticalflowscore[i] > threshold and opticalflowscore[i] > absolute_threshold:
            boundaries.append(classifiedgood[i])
    if len(boundaries) == 0 or boundaries[-1] is not classifiedgood[-1]:
        boundaries.append(classifiedgood[-1])
    
    
    shots = []
    boundaryid = 0
    shot = []
    for i in range(len(keyframes)):
        if keyframes[i] <= boundaries[boundaryid]:
            shot.append(keyframes[i])
        else:
            shots.append(shot)
            shot = []
            shot.append(keyframes[i])
            boundaryid = boundaryid + 1
    shots.append(shot)
    
    print shots
    print len(shots)
    
    final_valid_frames = 0
    subid = 0
    for i in range(len(shots)):
        shot = shots[i]
        print 'shot %d: %d frames' %(i, len(shot)),
        if 0 and len(shot) > min_shot_length:
            subfolder = outputfolder + str(subid) + '/'
            os.mkdir(subfolder)
            for j in shot:
                os.system('cp ' + data_dir + j + ' ' + subfolder + j)
            print ' '
            final_valid_frames = final_valid_frames + len(shot)
            subid = subid + 1
        else:
            print '     discarded'
        
    
    # output result to file    
    for i in keyframes:
        os.system('cp ' + data_dir + i + ' ' + outputfolder + i)
    resultfile = data_base_dir + 'keyframes.txt'
    if os.path.exists(resultfile):
        os.remove(resultfile)
    thefile = open(resultfile, 'w')
    for i in keyframes:
        thefile.write("%s\n" % i)
    
    # divide the sequence into shots
    #sub_sequences = data_base_dir + 'subsequences.txt'
    #if os.path.exists(sub_sequences):
    #    os.remove(sub_sequences)
    #subfile = open(sub_sequences, 'w')
    #for i in range(len(score)):
    #    subfile.write('%s %.4f\n' % (keyframes[i], score[i]))
    
    offilename = data_base_dir + 'opticalflowscore.txt'
    if os.path.exists(offilename):
        os.remove(offilename)
    offile = open(offilename, 'w')
    for i in range(len(opticalflowscore)):
        offile.write('%s %.4f\n' % (classifiedgood[i], opticalflowscore[i]))         
    
    print '%d optical flow local minima' % len(localminima)
    print 'Selected %d / %d (%.2f%%) frames as keyframes' % (final_valid_frames, len(filelist), 100.0*final_valid_frames/float(len(filelist)))
    print 'Split the sequence into %d sub-sequences' % subid
    
if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser(
        description="Select Keyframes from A Endoscopic Video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use_homography", type=bool, 
        default=True,
        help="whether use homography as a further step to extract frames")
    parser.add_argument("--dir", type=str, 
        default="/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/",
        #default="/home/ruibinma/throat/004/",
        help="data_base_dir: this folder should contain the folder images-raw")
    parser.add_argument("--min_shot_length", type=int, 
        default=50,
        help="minimum number of frames that a shot should contain")
    parser.add_argument("--nmins_window_size", type=int, 
        default=501,
        help="window size of non-minimum suppression")
    parser.add_argument("--percentile_threshold", type=int, 
        default=99,
        help="percentile threshold for optical flow boundary detection")
    parser.add_argument("--absolute_threshold", type=int, 
        default=5,
        help="absolute threshold for optical flow boundary detection (in pixel)")
    parser.add_argument("--ignore_ends", type=bool, 
        default=True,
        help="whether ignore the start the end 10 percent sequence")
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print "Elapsed time = %.2f second"%(time.time()- start_time)
    print "Done."

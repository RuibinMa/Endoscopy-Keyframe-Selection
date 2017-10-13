# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:00:00 2017


@author: ruibinma
"""

from PIL import Image
from shutil import rmtree
from pickimagenet import *
from compute_opticalflow2_function import motion_analysis_flownet2
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def main(args):
    data_base_dir = args.dir
    data_dir= data_base_dir + "images-raw/"
    
    if os.path.isdir(data_base_dir + 'classifiedgood'):
        rmtree(data_base_dir + 'classifiedgood')
    os.mkdir(data_base_dir + 'classifiedgood')
    if os.path.isdir(data_base_dir + 'classifiedbad'):
        rmtree(data_base_dir + 'classifiedbad')
    os.mkdir(data_base_dir + 'classifiedbad')
    
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
    
    filelist = os.listdir(data_dir)
    filelist.sort()
    #filelist = filelist[3000:3100]
    count = 1
    
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
    
    motion_analysis_flownet2(args)
    
if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser(
        description="Select Keyframes from A Endoscopic Video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", type=str, 
        default="/playpen/throat/Endoscope_Study/UNC_HN_Laryngoscopy_003/",
        #default="/home/ruibinma/throat/004/",
        help="data_base_dir: this folder should contain the folder images-raw")
    
    parser.add_argument('--caffemodel', help='path to model', 
                        default = '/playpen/software/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5')
    parser.add_argument('--deployproto', help='path to deploy prototxt template',
                        default = '/playpen/software/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template')
    parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
    #parser.add_argument('--warp',  help='whether create warpped images', default=False, type=bool)
    parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print "Elapsed time = %.2f second"%(time.time()- start_time)
    print "Done."

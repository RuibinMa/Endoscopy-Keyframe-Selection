# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:54:58 2016

@author: wrlife
"""

from pickimagenet import *
from defsolver import *

#Learning and testing

niter = 600  # number of iterations to train

# Reset style_solver as before.


model = style_net(train=True,learn_all=True)
style_solver_filename = solver(model, style_net(train=False, learn_all=True, subset='test'), 
                               base_lr = 0.001)
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', style_solver)]
#           ('scratch', scratch_style_solver)]
loss, weights = run_solvers(niter, solvers)
print 'Done.'

style_weights = weights['pretrained']#, weights['scratch']

del style_solver, solvers


import os

test_net, accuracy = eval_style_net(style_weights)

os.system("mv "+style_weights+" ./model")

os.system("mv "+model+" ./model")

print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )


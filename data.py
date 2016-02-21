#!/usr/bin/env python
# encoding: utf-8

import cPickle as pkl
import gzip
import numpy as np
from random import shuffle

train_val_set = pkl.load(open('../deepenhancer_orig/pkl/ubiquitous_trainmat.pkl', 'r'))
test_set = pkl.load(open('../deepenhancer_orig/pkl/ubiquitous_testmat.pkl', 'r'))

# train_set, val_set
N_train_val = len(train_val_set)
inds = range(N_train_val)
shuffle(inds)
N_val = N_train_val / 9
N_train = N_train_val - N_val
train_x = np.zeros((N_train, 4*400))
train_y = np.zeros((N_train, ))
val_x = np.zeros((N_val, 4*400))
val_y = np.zeros((N_val, ))
for row, ind in enumerate(inds):
    inputx = train_val_set[ind][0].reshape((1,-1))
    inputy = train_val_set[ind][1]
    if row < N_train:
        train_x[row, :] = inputx
        train_y[row] = inputy
    else:
        val_x[row-N_train, :] = inputx
        val_y[row-N_train] = inputy

print N_train, np.sum(train_y)
print N_val, np.sum(val_y)

# test_set
N_test = len(test_set)
inds = range(N_test)
shuffle(inds)
data_x = np.zeros((N_test, 4*400))
data_y = np.zeros((N_test, ))
for row, ind in enumerate(inds):
    inputx = test_set[ind][0].reshape((1,-1))
    inputy = test_set[ind][1]
    data_x[row, :] = inputx
    data_y[row] = inputy

print N_test, np.sum(data_y)

# train:val:test = 8:1:1
train_set = (train_x, train_y)
val_set = (val_x, val_y)
test_set = (data_x, data_y)

pkl.dump((train_set, val_set, test_set), open('./data/ubiquitous.pkl', 'w'))
# pkl.dump((train_set, val_set, test_set), gzip.open('./data/ubiquitous.pkl.gz', 'wb'))

#!/usr/bin/env python
# encoding: utf-8

import hickle as hkl
import numpy as np
from sklearn.cross_validation import StratifiedKFold

KFD = './data/ubiquitous_kfold.hkl'
POS = './data/ubiquitous_positive_aug.hkl'
NEG = './data/ubiquitous_negative_aug.hkl'
DATA = './data/ubiquitous_aug.hkl'

acgt2num = {'A': 0, 'C':1, 'G':2, 'T':3}

pos = hkl.load(POS)
neg = hkl.load(NEG)
assert len(pos) == len(neg)
n_samples = len(pos)

def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)
    mat = np.zeros((h, w))
    for i in xrange(w):
        mat[acgt2num[seq[i]], i] = 1
    return mat.reshape((1, -1))

labels = [1] * n_samples + [0] * n_samples
skf = StratifiedKFold(labels, n_folds=10, shuffle=True)
hkl.dump(skf, KFD, 'w')

y = np.array(labels)
Xpos = np.vstack([seq2mat(item[-1]) for item in pos])
Xneg = np.vstack([seq2mat(item[-1]) for item in neg])
X = np.vstack((Xpos, Xneg))

hkl.dump((X, y), DATA, mode='w', compression='gzip')


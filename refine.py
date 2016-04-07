#!/usr/bin/env python
# encoding: utf-8

import cPickle as pkl
import hickle as hkl
import numpy as np
from sklearn.cross_validation import StratifiedKFold

POS_OP = './data/ubiquitous_positive.op'
NEG_OP = './data/ubiquitous_negative.op'
POS_PKL = './data/ubiquitous_positive.pkl'
NEG_PKL = './data/ubiquitous_negative.pkl'
POS_REF = './data/ubiquitous_positive_refine.pkl'
NEG_REF = './data/ubiquitous_negative_refine.pkl'

DATA = './data/ubiquitous_refine.hkl'
KFD = './data/ubiquitous_kfold.pkl'

def readop(opfile):
    op = []
    f = open(opfile, 'r')
    contents = f.readlines()
    f.close()
    for row in contents:
        op.append(float(row[:-1]))
    return op

posop = readop(POS_OP)
negop = readop(NEG_OP)
assert len(posop) == len(negop)

posop_s = sorted(enumerate(posop), key=lambda x: x[1])
negop_s = sorted(enumerate(negop), key=lambda x: x[1])

[pos_idx, pos_ops] = zip(*posop_s)
[neg_idx, neg_ops] = zip(*negop_s)

# # find maximal subset that min(pos_ops) > max(neg_ops)
# maxsubset = len(pos_ops)
# while pos_ops[-maxsubset] <= neg_ops[maxsubset - 1]:
#     maxsubset -= 1
# print 'Maximal subsets contains {} samples statisfying pos_op > neg_op'.format(maxsubset)
# print 'The minimal #openness of pos is {}, the maximal #openness of neg is {}'.format(pos_ops[-maxsubset], neg_ops[maxsubset-1])

# omit 1000 misclassified samples
maxsubset = len(pos_ops) - 1000

pos = pkl.load(open(POS_PKL, 'r'))
neg = pkl.load(open(NEG_PKL, 'r'))

pos_idx = sorted(pos_idx[-maxsubset:])
neg_idx = sorted(neg_idx[:maxsubset])

pos_refine = [(i, pos[i][-1]) for i in pos_idx]
neg_refine = [(i, neg[i][-1]) for i in neg_idx]

with open(POS_REF, 'w') as f:
    pkl.dump(pos_refine, f)
with open(NEG_REF, 'w') as f:
    pkl.dump(neg_refine, f)

def acgt2num(s):
    if s in 'aA':
        return 0
    elif s in 'cC':
        return 1
    elif s in 'gG':
        return 2
    elif s in 'tT':
        return 3
    else:
        raise ValueError('Cannot convert %s into number' % s)

def seq2mat(seq):
    h = 4
    w = len(seq)
    mat = np.zeros((h, w))
    for i in xrange(w):
        mat[acgt2num(seq[i]), i] = 1
    return mat.reshape((1, -1))

labels = [1] * maxsubset + [0] * maxsubset
skf = StratifiedKFold(labels, n_folds=10, shuffle=True)
pkl.dump(skf, open(KFD, 'w'))

y = np.array(labels)
Xpos = np.vstack([seq2mat(item[-1]) for item in pos_refine])
Xneg = np.vstack([seq2mat(item[-1]) for item in neg_refine])
X = np.vstack((Xpos, Xneg))

hkl.dump((X, y), DATA, mode='w', compression='gzip')


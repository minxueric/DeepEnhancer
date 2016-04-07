#!/usr/bin/env python
# encoding: utf-8

import cPickle as pkl

POS = './data/ubiquitous_positive_refine.pkl'
NEG = './data/ubiquitous_negative_refine.pkl'
KFD = './data/ubiquitous_kfold.pkl'

POSFA_TRAIN = './data/ubiquitous_positive_lsgkm_train.fa'
NEGFA_TRAIN = './data/ubiquitous_negative_lsgkm_train.fa'
POSFA_TEST = './data/ubiquitous_positive_lsgkm_test.fa'
NEGFA_TEST = './data/ubiquitous_negative_lsgkm_test.fa'

pos_refine = pkl.load(open(POS, 'r'))
neg_refine = pkl.load(open(NEG, 'r'))
data_refine = pos_refine + neg_refine
labels = [1] * len(pos_refine) + [0] * len(neg_refine)

kfd = pkl.load(open(KFD, 'r'))
kfs = [(train, test) for train, test in kfd]
(train, test) = kfs[0]
assert len(pos_refine) * 2 == len(train) + len(test)

fp = open(POSFA_TRAIN, 'w')
fn = open(NEGFA_TRAIN, 'w')
for idx in train:
    if labels[idx] == 1:
        fp.write('>%05d\n' % data_refine[idx][0])
        fp.write('%s\n' % data_refine[idx][1])
    else:
        fn.write('>%05d\n' % data_refine[idx][0])
        fn.write('%s\n' % data_refine[idx][1])
fp.close()
fn.close()

fp = open(POSFA_TEST, 'w')
fn = open(NEGFA_TEST, 'w')
for idx in test:
    if labels[idx] == 1:
        fp.write('>%05d\n' % data_refine[idx][0])
        fp.write('%s\n' % data_refine[idx][1])
    else:
        fn.write('>%05d\n' % data_refine[idx][0])
        fn.write('%s\n' % data_refine[idx][1])
fp.close()
fn.close()




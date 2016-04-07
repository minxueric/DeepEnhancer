#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cPickle as pkl
import hickle as hkl
import os

POS_RGN = './data/ubiquitous_positive.rgn'
POS_PKL = './data/ubiquitous_positive_refine.pkl'
NEG_RGN = './data/ubiquitous_negative.rgn'
NEG_PKL = './data/ubiquitous_negative_refine.pkl'

MSA_046 = '/home/liuqiao/MSA/m046'
MSA_100 = '/home/liuqiao/MSA/m100'
NUM = {MSA_046: 46., MSA_100: 100.}

def genmat(loc, msa, lastname, lastcontent):
    (chr, start, end) = loc
    start = int(start)
    end = int(end)
    mat = np.zeros((4, end - start))
    if chr == 'chrX':
        chr = 'chr23'
    elif chr == 'chrY':
        chr = 'chr24'
    chrnum = int(chr[3:])
    filename = 'outcome_chr%02d.txt' % chrnum
    if filename == lastname:
        filecontent = lastcontent
    else:
        fullfile = os.path.join(msa, filename)
        f = open(fullfile, 'r')
        filecontent = f.readlines()
        f.close()
    for i in range(start, end):
        weights = filecontent[i + 1][:-1].split('\t')[-4:]
        mat[0, i - start] = int(weights[0])
        mat[1, i - start] = int(weights[2])
        mat[2, i - start] = int(weights[1])
        mat[3, i - start] = int(weights[3])
    mat /= NUM[msa]
    mat = mat.reshape((1, -1))
    return mat, filename, filecontent

def readmsa(REF, RGN, MSA):
    pos = pkl.load(open(REF))
    indice = [item[0] for item in pos]
    f = open(RGN)
    lines = f.readlines()
    f.close()
    loci = [lines[idx][:-1].split('\t') for idx in indice]
    lastname = None
    lastcontent = None
    mats = []
    for i, loc in enumerate(loci):
        mat, filename, filecontent = genmat(loc, MSA, lastname, lastcontent)
        lastname = filename
        lastcontent = filecontent
        mats.append(mat)
    return mats

if __name__ == '__main__':
    posmats = readmsa(POS_PKL, POS_RGN, MSA_100)
    negmats = readmsa(NEG_PKL, NEG_RGN, MSA_100)
    assert len(posmats) == len(negmats)
    labels = [1] * len(posmats) + [0] * len(posmats)
    y = np.array(labels)
    Xpos = np.vstack(posmats)
    Xneg = np.vstack(negmats)
    X = np.vstack((Xpos, Xneg))
    hkl.dump((X, y), './data/ubiquitous_msa100.hkl', mode='w', compression='gzip')


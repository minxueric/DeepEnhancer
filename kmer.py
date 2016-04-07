#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict
from math import log, sqrt
import numpy as np
import hickle as hkl

POS_PKL = './data/ubiquitous_positive_lsgkm_train.fa'
NEG_PKL = './data/ubiquitous_negative_lsgkm_train.fa'
KMER_HKL = './data/ubiquitous_kmer.hkl'
KMER_FA = './data/ubiquitous_kmer.fa'
KMER_SCORE = './data/ubiquitous_kmer_weight'

s2n = {'A':0, 'C':1, 'G':2, 'T':3}

def kmers(fastas, k=10):
    n = len(fastas)
    kmercounts = [defaultdict(int) for i in xrange(n)]
    for i in xrange(n):
        f = open(fastas[i], 'r')
        lines = f.readlines()
        f.close()
        ns = len(lines) / 2
        for j in xrange(ns):
            seq = lines[2 * j + 1][:-1].upper()
            l = len(seq)
            for pos in xrange(l - k + 1):
                key = seq[pos:pos + k]
                kmercounts[i][key] += 1
    return kmercounts

def convseq(seqs, labels):
    n = len(seqs)
    l = len(seqs[0])
    mat = -np.ones((n, 1, 4, l))
    for i in xrange(n):
        seq = seqs[i].upper()
        l = len(seq)
        for j, s in enumerate(seq):
            mat[i, 0, s2n[s], j] = 1
        if labels[i] == False:
            mat[i] = - mat[i]
    return mat

if __name__ == '__main__':
    kmercounts = kmers((POS_PKL, NEG_PKL), k=10)
    keys = kmercounts[0].keys() + kmercounts[1].keys()
    print len(keys)
    # f = open(KMER_FA, 'w')
    # for i, key in enumerate(keys):
    #     f.write('>%d\n' % i)
    #     f.write('%s\n' % key)
    # f.close()
    # kmercntslogratio = defaultdict(float)
    # for key in keys:
    #     kmercntslogratio[key] = sqrt(kmercounts[0][key] + kmercounts[1][key]) * abs(log((kmercounts[0][key] + 50.)/(kmercounts[1][key] + 50.)))
    # kmersorted = sorted(kmercntslogratio, key=lambda k: kmercntslogratio[k], reverse=True)
    # kmerscores = [(key, kmercntslogratio[key], kmercounts[0][key], kmercounts[1][key])
    #         for key in kmersorted]
    # n = 10000
    # for i, kmer in enumerate(kmerscores[n-1000:n]):
    #     print '{0:<10d}{1[0]:<15}{1[1]:<10.4f}{1[2]:<10d}{1[3]:<10d}'.format(i, kmer)
    # seqs = kmersorted[:n]
    # labels = [kmercounts[0][key] > kmercounts[1][key] for key in seqs]
    # mat = convseq(seqs, labels)
    # hkl.dump(mat, KMER_HKL, 'w')
    f = open(KMER_SCORE, 'r')
    contents = f.readlines()
    f.close()
    scores = dict()
    for line in contents:
        id = int(line.split('\t')[0])
        score = float(line[:-1].split('\t')[1])
        scores[id] = score
    idsorted = sorted(scores, key=lambda k: abs(scores[k]), reverse=True)
    topkeys = [keys[id] for id in idsorted[:1000]]
    labels = [scores[id]>0 for id in idsorted[:1000]]
    print labels
    mat = convseq(topkeys, labels)
    hkl.dump(mat, KMER_HKL, 'w')

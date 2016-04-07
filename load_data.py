import gzip
import cPickle as pkl
import hickle as hkl
import numpy as np
import bisect
import random
import os
import copy
from sklearn.cross_validation import StratifiedKFold

# ubiquitous enhancer dataset ~40k
INDEX = '/home/openness/human/data/info/enhancer/permissive_enhancers.bed.gz'
# gm12878 active enhancers
INDEX1 = '/home/openness/human/data/info/enhancer/gm12878-active-enhancers.bed'
# Genome sequence chr1-22,X,Y
GENOME = '/home/openness/common/igenomes/Homo_sapiens/UCSC/hg19/Sequence/Chromosomes/'
# random negative samples except the following regions
ENHANCER    = '/home/xumin/deepenhancer/temp/enhancer.txt'     # 800k enhancers data
PROMOTER    = '/home/xumin/deepenhancer/temp/promoter.txt'     # 33k promoters data
LNCPROMOTER = '/home/xumin/deepenhancer/temp/lncpromoter.txt'  # 153k lncpromoters data
LNCRNA      = '/home/xumin/deepenhancer/temp/lncrna.txt'       # 89k lncrnas data
EXON        = '/home/xumin/deepenhancer/temp/genes.gtf.gz'  # 963k exons(CDS) data

acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}

def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=bool)  # True or false in mat
    for i in xrange(w):
        mat[acgt2num[seq[i]], i] = 1.
    return mat.reshape((1, -1))

# load whole genome sequence
print 'Loading whole genome sequence...'
chrs = range(1, 23)
chrs.extend(['X', 'Y'])
keys = ['chr' + str(x) for x in chrs]
if os.path.isfile('./temp/sequences.hkl'):
    print 'Find corresponding hkl file'
    sequences = hkl.load('./temp/sequences.hkl')
else:
    sequences = dict()
    for i in range(24):
        fa = open('%s%s.fa' % (GENOME, keys[i]), 'r')
        sequence = fa.read().splitlines()[1:]
        fa.close()
        sequence = ''.join(sequence)
        sequences[keys[i]]= sequence
    hkl.dump(sequences, './temp/sequences.hkl', 'w')

def checkseq(chrkey, start, end):
    sequence = sequences[chrkey][start:end]
    legal = ('n' not in sequence) and ('N' not in sequence)
    return sequence, legal

def loadindex(name='ubiquitous'):
    """Load enhancers indexes (id, chr, start, end)"""
    print 'Loading %s enhancer indexes...' % name
    if name == 'ubiquitous':
        if os.path.isfile('./temp/ubiquitous_index.hkl'):
            print 'Find corresponding hkl file'
            indexes = hkl.load('./temp/ubiquitous_index.hkl')
            return indexes
        fr = gzip.open(INDEX, 'r')
        entries = fr.readlines()
        fr.close()
        n = len(entries)
        indexes = list()
        for i, entry in enumerate(entries):
            chrkey, start, end = entry.split('\t')[:3]
            start = int(start) - 1
            end = int(end) - 1
            seq, legal = checkseq(chrkey, start, end)
            if legal:
                indexes.append(['ubiquitous%05d' % i, chrkey, start, end, seq])
        print 'Totally {0} enhancers in {1}'.format(n, INDEX)
        hkl.dump(indexes, './temp/ubiquitous_index.hkl', 'w')
    if name == 'gm12878':
        f = open(INDEX1, 'r')
        entries = f.read().splitlines()
        f.close()
        n = len(entries)
        print 'Totally {0} enhancers in {1}'.format(n-1, INDEX1)
        indexes = list()
        for i, entry in enumerate(entries[1:]):
            indexes.append([i] + entry.split(' ')[:3])
    return indexes

def chunks(l, n, o):
    """Yield successive n-sized chunks with o-sized overlaps from l."""
    return [l[i: i + n] for i in range(0, len(l), n-o)]

def merge(a, b):
    """merge sort 2 sorted (a[:][0]) list a & b"""
    if a == []:
        return b
    if b == []:
        return a
    l = []
    while a and b:
        if a[0][0] < b[0][0]:
            l.append(a.pop(0))
        else:
            l.append(b.pop(0))
    return l + a + b

def merge_(lists):
    """merge all the sorted lists in lists, merge lists[0], lists[1], lists[2]..."""
    L = len(lists)
    temp = merge(lists[0], lists[1])
    for i in range(2, L):
        temp = merge(lists[i], temp)
    return temp

def stopreads(file, l):
    """read stop reads from file for negative sample generating"""
    nos = dict()
    for key in keys:
        nos.setdefault(key, list())
    if file in [ENHANCER, PROMOTER, LNCPROMOTER, LNCRNA]:
        f = open(file, 'r')
        entries = f.readlines()[2:]
        f.close()
        for i, entry in enumerate(entries):
            [chrkey, start, end] = [x for x in entry[:-1].split('\t')]
            start = int(start) - 1
            end = int(end) - 1
            if chrkey not in keys:
                continue
            if (nos[chrkey] != []) and (start - l < nos[chrkey][-1][-1]):
                nos[chrkey][-1][1] = max(end, nos[chrkey][-1][-1])
            else:
                nos[chrkey].append([start - l, end])
    elif file in [EXON]:
        f = gzip.open(EXON, 'r')
        entries = f.readlines()
        f.close()
        for i, entry in enumerate(entries):
            [chrkey, un, exon, start, end] = entry.split('\t')[:5]
            if exon != 'exon':
                continue
            start = int(start) - 1
            end = int(end) - 1
            if chrkey not in keys:
                continue
            if (nos[chrkey] != []) and (start - l < nos[chrkey][-1][-1]):
                nos[chrkey][-1][1] = max(end, nos[chrkey][-1][-1])
            else:
                nos[chrkey].append([start - l, end])
    return nos

def nosrgns(indexes, l):
    """negative sequences DO NOT lie in result regions
        parameters:
            indexes: regions not included
            l: length of sequence to be sampled"""
    if os.path.isfile('./temp/nos_bins_%d.hkl' % l):
        return hkl.load('./temp/nos_bins_%d.hkl' % l)

    chrlens = [len(sequences[keys[i]]) for i in range(24)]
    if os.path.isfile('./temp/prelens.pkl'):
        prelens = pkl.load(open('./temp/prelens.pkl', 'r'))
    else:
        prelens = [next(a for a in xrange(chrlens[i]) if sequences[keys[i]][a] not in 'nN') for i in xrange(24)]
        pkl.dump(prelens, open('./temp/prelens.pkl', 'w'))
    if os.path.isfile('./pkl/suflens.pkl'):
        suflens = pkl.load(open('./temp/suflens.pkl', 'r'))
    else:
        suflens = [next(a for a in xrange(chrlens[i]) if sequences[keys[i]][-a] not in 'nN') for i in xrange(24)]
        pkl.dump(suflens, open('./temp/suflens.pkl', 'w'))
    chrbins = np.add.accumulate(chrlens)

    nos00 = dict()
    for chrkey in keys:
        nos00.setdefault(chrkey, list())
    for entry in indexes:
        chrkey = entry[1]
        start = entry[2]
        end = entry[3]
        if (nos00[chrkey] != []) and (start - l < nos00[chrkey][-1][-1]):
            nos00[chrkey][-1][-1] = max(end, nos00[chrkey][-1][-1])
        else:
            nos00[chrkey].append([start - l, end])

    nos1 = stopreads(file=ENHANCER, l=l)
    nos2 = stopreads(file=PROMOTER, l=l)
    nos3 = stopreads(file=LNCPROMOTER, l=l)
    nos4 = stopreads(file=LNCRNA, l=l)
    nos5 = stopreads(file=EXON, l=l)

    # merge nos00, nos1~5
    nos0 = dict()
    for key in keys:
        nos0[key] = merge_([nos00[key], nos1[key], nos2[key], nos3[key], nos4[key], nos5[key]])

    # merge overlap regions
    nos = dict()
    for i, key in enumerate(keys):
        nos[key]=[[0, prelens[i]]]
        for no in nos0[key]:
            if key in nos.keys() and no[0] - l < nos[key][-1][1]:
                nos[key][-1][1] = max(no[1], nos[key][-1][1])
            else:
                nos[key].append(no)
        if chrlens[i] - suflens[i] - l < nos[key][-1][-1]:
            nos[key][-1][-1] = chrlens[i]
        else:
            nos[key].append([chrlens[i] - suflens[i] - l, chrlens[i]])

    hkl.dump((nos, chrbins), './temp/nos_bins_%d.hkl' % l, 'w')
    return nos, chrbins

def train_test_split(indexes, ratio):
    """split train test dataset
        parameters:
            indexes: indexes to be split
            ratio: test ratio"""
    print 'Splitting the indexes into train and test parts...'
    if os.path.isfile('./temp/ubiquitous_index_split_%.2f.hkl' % ratio):
        return hkl.load('./temp/ubiquitous_index_split_%.2f.hkl' % ratio)
    n_samles = len(indexes)
    indexes_bak = copy.copy(indexes)
    random.shuffle(indexes_bak)
    n_train = int(n_samles * (1-ratio))
    train_indexes = indexes_bak[:n_train]
    test_indexes = indexes_bak[n_train:]
    hkl.dump((train_indexes, test_indexes),
            './temp/ubiquitous_index_split_%.2f.hkl' % ratio)
    return train_indexes, test_indexes

def tofasta(indexes, fafile):
    """format indexes into fasta file
        parameters:
            indexes: indexes to be saved to fastas
            fafile: destination fasta file"""
    print 'Saving sequences into %s...' % fafile
    f = open(fafile, 'w')
    for index in indexes:
        if len(index) == 4:
            [id, chrkey, start, end] = index
            f.write('>{0[0]}.{0[1]}.{0[2]:010d}.{0[3]:010d}\n{1}\n'.format(
                index, sequences[chrkey][start:end]))
        elif len(index) == 5:
            [id, chrkey, start, end, seq] = index
            f.write('>{0[0]}.{0[1]}.{0[2]:010d}.{0[3]:010d}\n{1}\n'.format(index, seq))
        else:
            raise ValueError('index not in correct format!')
    f.close()

def genrand(num, indexes, l):
    """generate random negative sequences
        parameters:
            num: number of random seqs
            indexes: positive indexes
            l: length of a random seq"""
    print 'Generating random negative samples with length {} bps...'.format(l)
    nos, chrbins = nosrgns(indexes, l)
    oks = dict()
    bins = dict()
    noslens = dict()
    for z in range(24):
        no = nos[keys[z]]
        oks[keys[z]] = [second[0] - first[1]  for first, second in zip(no, no[1:])]
        bins[keys[z]] = np.add.accumulate(oks[keys[z]])
        noslens[keys[z]] = [n[1] - n[0] for n in no]

    # random chrom
    nonenhancers = list()
    ratio = list(chrbins[0:1]) + list(np.diff(chrbins))
    ratio = [x / float(sum(ratio)) for x in ratio]
    nums = [int(num * r) for r in ratio]
    residual = num - sum(nums)
    nums[-1] += residual
    for chrnum in xrange(24):
        # random site on the chrom #chrnum
        us = np.random.rand(int(nums[chrnum] * 10)) * bins[keys[chrnum]][-1]
        i = 0
        for u in us:
            if i == nums[chrnum]:
                break
            z = bisect.bisect_left(bins[keys[chrnum]], u)
            start = int(u + np.sum(noslens[keys[chrnum]][:(z + 1)]))
            end = start + l
            seq, legal = checkseq(keys[chrnum], start, end)
            if not legal:
                continue
            else:
                nonenhancers.append(('random%010d'%i, keys[chrnum], start, end, seq))
                i += 1
    # save nonenhancers
    print 'Generate {} random negative samples'.format(len(nonenhancers))
    return nonenhancers

def cropseq(indexes, l, stride):
    """generate chunked enhancer sequence according to loaded index"""
    print 'Generating cropped positive samples with length {} bps...'.format(l)
    enhancers = list()
    for index in indexes:
        [sampleid, chrkey, startpos, endpos, _] = index
        l_orig = endpos - startpos
        if l_orig < l:
            for shift in range(0, l - l_orig, stride):
                start = startpos - shift
                end = start + l
                seq, legal = checkseq(chrkey, start, end)
                if legal:
                    enhancers.append((sampleid, chrkey, start, end, seq))
        elif l_orig >= l:
            chunks_ = chunks(range(startpos, endpos), l, l - stride)
            for chunk in chunks_:
                start = chunk[0]
                end = chunk[-1] + 1
                if (end - start) == l:
                    seq, legal = checkseq(chrkey, start, end)
                    enhancers.append((sampleid, chrkey, start, end, seq))
                elif (end - start) < l:
                    break

    print 'Data augmentation: from {} indexes to {} samples'.format(len(indexes), len(enhancers))
    return enhancers

def main(name='ubiquitous', test_ratio=0.15, l=300, stride_train=1, stride_test=1):
    # Files where to save seqs NAME = 'ubiquitous' or 'gm12878'
    TRAIN_POS_FA = './data/%s_train_positive.fa' % name
    TEST_POS_FA = './data/%s_test_positive.fa' % name
    TRAIN_NEG_FA = './data/%s_train_negative.fa' % name
    TEST_NEG_FA = './data/%s_test_negative.fa' % name

    indexes = loadindex(name)
    train_indexes, test_indexes = train_test_split(indexes, ratio=test_ratio)
    train_pos = cropseq(train_indexes, l, stride_train)
    test_pos = cropseq(test_indexes, l, stride_test)
    tofasta(train_indexes, TRAIN_POS_FA)
    tofasta(test_indexes, TEST_POS_FA)

    train_rand = genrand(len(train_pos), indexes, l)
    test_rand = genrand(len(test_indexes), indexes, l)
    train_rand_bak = copy.copy(train_rand)
    random.shuffle(train_rand_bak)
    tofasta(train_rand_bak[:len(train_indexes)], TRAIN_NEG_FA)
    tofasta(test_rand, TEST_NEG_FA)

    train_y = [1] * len(train_pos) + [0] * len(train_rand)
    train_y = np.array(train_y, dtype=bool)
    train_X_pos = np.vstack([seq2mat(item[-1]) for item in train_pos])
    train_X_neg = np.vstack([seq2mat(item[-1]) for item in train_rand])
    train_X = np.vstack((train_X_pos, train_X_neg))

    test_y = [1] * len(test_pos) + [0] * len(test_rand)
    test_y = np.array(test_y, dtype=bool)
    test_pos_ids = [item[0] for item in test_pos]
    test_X_pos = np.vstack([seq2mat(item[-1]) for item in test_pos])
    test_X_neg = np.vstack([seq2mat(item[-1]) for item in test_rand])
    test_X = np.vstack((test_X_pos, test_X_neg))
    # dump train/test X/y into hickle file
    hkl.dump((train_X, train_y), './data/%s_train.hkl' % name, 'w')
    hkl.dump((test_X, test_y), './data/%s_test.hkl' % name, 'w')
    hkl.dump(test_pos_ids, './data/%s_test_pos_ids.hkl' % name, 'w')
    return

if __name__ == "__main__":
    main()

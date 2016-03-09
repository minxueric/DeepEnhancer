import gzip
import cPickle as pkl
from random import shuffle
import matplotlib.pyplot as plt
import heapq
import numpy as np
import bisect
import random
import os

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

# load whole genome sequence
print 'loading whole genome sequence...'
chrs = range(1,23)
chrs.extend(['X', 'Y'])
keys = ['chr'+str(x) for x in chrs]
sequences = dict()
for i in range(24):
    fa = open('%s%s.fa' % (GENOME, keys[i]), 'r')
    sequence = fa.read().splitlines()[1:]
    sequence = ''.join(sequence)
    sequences[keys[i]]= sequence

def loadindex(name='ubiquitous'):
    """load enhancers indexes"""
    print 'loading enhancer indexes...'
    if name == 'ubiquitous':
        fr = gzip.open(INDEX, 'r')
        entries = fr.readlines()
        fr.close()
        n = len(entries)
        print '...Totally {0} enhancers in {1}'.format(n, INDEX)
        indexes = list()
        for i, entry in enumerate(entries):
            indexes.append(entry.split('\t')[:3])
    if name == 'gm12878':
        f = open(INDEX1, 'r')
        entries = f.read().splitlines()
        f.close()
        n = len(entries)
        print '...Totally {0} enhancers in {1}'.format(n-1, INDEX1)
        indexes = list()
        for entry in entries[1:]:
            indexes.append(entry.split(' ')[:3])
    return indexes

def chunks(l, n, o):
    """Yield successive n-sized chunks with o-sized overlaps from l."""
    return [l[i: i+n] for i in range(0, len(l), n-o)]

def SeqLenDis(indexes):
    lengths = []
    for index in indexes:
        lengths.append(int(index[2]) - int(index[1]))
    print heapq.nlargest(30, lengths)
    plt.hist(lengths)
    plt.show()

def enhchr(indexes):
    """chromosome length vs. enhancer numbers on it"""
    lens = np.array([len(sequences[keys[i]]) for i in range(24)], dtype=np.float)
    nums = np.zeros((24,))
    for index in indexes:
        chrkey = index[0]
        nums[keys.index(chrkey)] += 1
    print "The length of 24 Chromosomes are \n{}".format(np.array(lens, dtype=np.uint64))
    print "The number of enhancers on each chromosome are \n{}".format(np.array(nums, dtype=np.uint64))

    ind = np.arange(24)
    w = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, lens / np.sum(lens), w, color='r')
    rects2 = ax.bar(ind + w, nums / np.sum(nums), w, color='y')
    ax.set_ylabel('Chrom Length & #Enhancers')
    ax.set_xticks(ind + w)
    ax.set_xticklabels(keys)
    ax.legend((rects1[0], rects2[0]), ('Chrom Length (%)', '#Enahncers (%)'))
    plt.show()

def checkseq(chrkey, start, end):
    sequence = sequences[chrkey][start:end]
    return ('n' not in sequence) and ('N' not in sequence)

def genseq(indexes, pklfile, bedfile, rgnfile, l=400, overlap=10):
    """generate chunked enhancer sequence according to loaded index"""
    # shuffle(indexes)    # shuffle chroms orders
    print 'generating chunked enhancer sequence samples...'
    enhancers = list()
    for index in indexes:
        chrkey = index[0]
        startpos = int(index[1]) - 1
        endpos = int(index[2]) - 1
        chunks_ = chunks(range(startpos, endpos), l, overlap)
        for i, chunk in enumerate(chunks_):
            start = chunk[0]
            end = chunk[-1]
            if (end - start) == l:
                if checkseq(chrkey, start, end):
                    enhancers.append((chrkey, start, end, sequences[chrkey][start:end]))
            elif (end - start) < l and i > 0:
                start = endpos - l
                end = endpos
                if checkseq(chrkey, start, end):
                    enhancers.append((chrkey, start, end, sequences[chrkey][start:end]))
            elif (end - start) < l and i == 0:
                mid = (end + start) / 2
                start = mid - l / 2
                end = mid + l / 2
                if checkseq(chrkey, start, end):
                    enhancers.append((chrkey, start, end, sequences[chrkey][start:end]))

    pkl.dump(enhancers, open(pklfile, 'w'))
    fb = open(bedfile, 'w')
    for enhancer in enhancers:
        fb.write('>%s.%010d.%010d\n%s\n' % enhancer)
    fb.close()
    fr = open(rgnfile, 'w')
    for enhancer in enhancers:
        fr.write('%s\t%010d\t%010d\n' % enhancer[:3])
    fr.close()
    print '...After chunking, there are {} enhancers'.format(len(enhancers))
    return len(enhancers)

def merge(a,b):
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
    """merge sort list in the lists"""
    L = len(lists)
    temp = merge(lists[0], lists[1])
    for i in range(2, L):
        temp = merge(lists[i], temp)
    return temp

def stopreads(file, l):
    """read stop reads for negative sample generating"""
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

def genrand(num, indexes, pklfile, bedfile, rgnfile, l=400):
    """generate random negative sequences"""
    print 'generating random negative sequence samples...'
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
        chrkey = entry[0]
        start = int(entry[1])
        end = int(entry[2])
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
    done = False
    i = 0
    while not done:
        # u = random.random() * chrbins[-1]
        u = (i + random.random()) / num * chrbins[-1]
        chrnum = bisect.bisect_left(chrbins, u)
        # random site on the chrom
        u = random.random() * bins[keys[chrnum]][-1]
        z = bisect.bisect_left(bins[keys[chrnum]], u)
        start = int(u + np.sum(noslens[keys[chrnum]][:(z+1)]))
        end = start + l
        if ('n' in sequences[keys[chrnum]][start:end]) or ('N' in sequences[keys[chrnum]][start:end]):
            continue
        else:
            nonenhancers.append((keys[chrnum], start, end, sequences[keys[chrnum]][start:end]))
            i += 1
            done = (i == num)

    # save nonenhancers
    pkl.dump(nonenhancers, open(pklfile, 'w'))
    fb = open(bedfile, 'w')
    for nonenhancer in nonenhancers:
        fb.write('>%s.%010d.%010d\n%s\n' % nonenhancer)
    fb.close()
    fr = open(rgnfile, 'w')
    for nonenhancer in nonenhancers:
        fr.write('%s\t%010d\t%010d\n' % nonenhancer[:3])
    fr.close()
    print '...There are {} nonenhancer samples generated'.format(len(nonenhancers))
    return

def main(name='ubiquitous', l=400, overlap=10):
    # Files where to save seqs NAME = 'ubiquitous' or 'gm12878'
    POSPKL = './data/%s_positive.pkl' % name
    NEGPKL = './data/%s_negative.pkl' % name
    POSBED = './data/%s_positive.fa' % name
    NEGBED = './data/%s_negative.fa' % name
    POSRGN = './data/%s_positive.rgn' % name
    NEGRGN = './data/%s_negative.rgn' % name

    indexes = loadindex(name)
    num = genseq(indexes, POSPKL, POSBED, POSRGN, l=l, overlap=overlap)
    genrand(num, indexes, NEGPKL, NEGBED, NEGRGN, l=l)
    return

if __name__ == "__main__":
    main()

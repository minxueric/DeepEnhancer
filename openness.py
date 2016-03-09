#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import csv
import matplotlib.pyplot as plt

OPEN = '/home/xumin/openness/enhancer'
TASKS = [o for o in os.listdir(OPEN) if os.path.isdir(os.path.join(OPEN, o))]
assert len(TASKS) == 628

LABELS = ['positive', 'negative']
RGNS = ['ubiquitous_%s.rgn' % label for label in LABELS]
DATAS = ['./data/ubiquitous_%s.open' % label for label in LABELS]
RESULTS = ['./data/ubiquitous_%s.op' % label for label in LABELS]
boxdata = []
for label, rgn, data, res in zip(LABELS, RGNS, DATAS, RESULTS):
    openness = []
    if os.path.isfile(data):
        raw = csv.reader(open(data, 'r'), delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        for row in raw:
            openness.append(row)
    else:
        for task in TASKS:
            file = os.path.join(OPEN, task, rgn, task)
            with open(file) as f:
                openness.append([float(line[:-1].split('\t')[-1]) for line in f.readlines()])
        openness = np.array(openness)
        openness = openness.T
        openness = openness.tolist()
        with open(data, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(openness)
    f = open(res, 'w')
    values = []
    for row in openness:
        value = np.sum([item > 2 for item in row])
        values.append(value)
        f.write('%f\n' % value)
        values.append(value)
    # plt.hist(values, bins=200)
    # plt.xlabel('# experiments with openness')
    # plt.ylabel('# samples')
    # plt.show()
    boxdata.append(values)
plt.boxplot(boxdata)
plt.show()
print np.sum([item < 25 for item in boxdata[0]])
print np.sum([item > 25 for item in boxdata[1]])





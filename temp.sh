#!/bin/bash

BASE=/home/openness/human/dump/gzip

zcat $BASE/enhancer/enhancer.txt.gz | cut -f 2-4 > ./temp/enhancer.txt
zcat $BASE/promoter/promoter.txt.gz | cut -f 2-4 > ./temp/promoter.txt
zcat $BASE/lncpromoter/lncpromoter.txt.gz | cut -f 2-4 > ./temp/lncpromoter.txt
zcat $BASE/lncrna/lncrna.txt.gz | cut -f 2-4 > ./temp/lncrna.txt
# zcat $BASE/gene/gene.txt.gz | cut -f 2-4 > ./temp/gene.txt
cp /home/openness/human/data/anno/hg19/Genes/genes.gtf.gz ./temp/genes.gtf.gz

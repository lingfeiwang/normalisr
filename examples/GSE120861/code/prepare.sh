#!/bin/bash

set -x -e -o pipefail

cd data/
mkdir reads highmoi
tar xf meta.tar.xz

#Download data GSE120861 highmoi
cd reads
wget -O highmoi_cells.txt.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120861&format=file&file=GSE120861%5Fpilot%5Fhighmoi%5Fscreen%2Ecells%2Etxt%2Egz'
wget -O highmoi_exprs.mtx 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120861&format=file&file=GSE120861%5Fpilot%5Fhighmoi%5Fscreen%2Eexprs%2Emtx%2Egz'
wget -O highmoi_genes.txt.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120861&format=file&file=GSE120861%5Fpilot%5Fhighmoi%5Fscreen%2Egenes%2Etxt%2Egz'
wget -O highmoi_phenoData.txt.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120861&format=file&file=GSE120861%5Fpilot%5Fhighmoi%5Fscreen%2EphenoData%2Etxt%2Egz'
cd ../..

#Process highmoi to matrix format
python3 code/prepare_highmoi.py 

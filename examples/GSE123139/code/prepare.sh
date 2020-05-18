#!/bin/bash

set -x -e -o pipefail

cd data/
mkdir reads raw de coex go
tar xf meta.tar.xz

#Download GO annotations
#See http://current.geneontology.org/products/pages/downloads.html
#See http://geneontology.org/docs/download-ontology/
cd go
wget -O goa-human.gaf.gz 'http://geneontology.org/gene-associations/goa_human.gaf.gz'
gunzip goa-human.gaf.gz
wget -O go-basic.obo 'http://purl.obolibrary.org/obo/go/go-basic.obo'
cd ..

#Download data GSE123139
cd reads
wget -O reads.tar 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE123139&format=file'
tar xf reads.tar
rm reads.tar
cd ../..

#Process to matrix format
python3 code/prepare_raw.py
#Process to differential expression input dataset
python3 code/prepare_de.py
#Process to co-expression input dataset
python3 code/prepare_coex.py

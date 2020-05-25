#!/usr/bin/python3

import numpy as np
from os.path import join as pjoin
from os import linesep
from shutil import copyfile
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
import gzip

diri='data/raw'
diro='data/de'

key='celltype'
values=['dysfunctional','naive']

#Load covariate info
dc=np.loadtxt(pjoin(diri,'cov.tsv.gz'),delimiter='\t')
with open(pjoin(diri,'cov.txt'),'r') as f:
	namec=f.readlines()
namec=np.array([x.strip() for x in namec])
namecdict=dict(zip(namec,range(len(namec))))

#Select cells for DE
ids=[namecdict[key+'='+x] for x in values]
ids=dc[ids].astype(bool)
assert ids.any(axis=1).all()
ida=ids.any(axis=0)

#Process covariates
namecn_id=np.array([namecdict[x] for x in filter(lambda x:not x.startswith(key+'='),namec)])
dcn=dc[namecn_id][:,ida]
#Remove single-valued covariates
t1=[len(np.unique(x))>1 for x in dcn]
namecn_id=namecn_id[t1]
dcn=dcn[t1]
namecn=namec[namecn_id]
#Output covariates
np.savetxt(pjoin(diro,'0_cov.tsv.gz'),dcn,delimiter='\t',fmt="%.8G")
with open(pjoin(diro,'0_cov.txt'),'w') as f:
	f.write(linesep.join(namecn))
del namecn,dcn

#Process cells
with open(pjoin(diri,'cell.txt'),'r') as f:
	names=f.readlines()
names=np.array([x.strip() for x in names])
namesn=names[ida]
with open(pjoin(diro,'0_cell.txt'),'w') as f:
	f.write(linesep.join(namesn))

#Process transcriptome
dt=np.loadtxt(pjoin(diri,'read.tsv.gz'),delimiter='\t')
dtn=dt[:,ida]
dtn=coo_matrix(dtn)
with gzip.open(pjoin(diro,'0_read.mtx.gz'),'w') as f:
	mmwrite(f,dtn,field='integer')

#Process grouping
dg=np.zeros(len(ida),dtype=int)
dg[ids[0]]=1
dg=dg[ida].astype(int)
#Output grouping
np.savetxt(pjoin(diro,'0_group.tsv.gz'),dg,delimiter='\t',fmt="%u")

#Copy genes
copyfile(pjoin(diri,'gene.txt'),pjoin(diro,'0_gene.txt'))

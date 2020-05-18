#!/usr/bin/python3

import itertools
import numpy as np
from os.path import join as pjoin
import pandas as pd
import os
from os import linesep

diri='data'
diro='data/raw'
#Cell types to keep for light-weight demonstration
subsets={'celltype':set(['dysfunctional','naive'])}
#Probability of dropping cell for downsampled light-weight demonstration
#Set drop_rate=0 to use full data
drop_rate=0.5


#Read meta data
diri1=pjoin(diri,'meta')
fs=os.listdir(diri1)
ds=[]
for fname in filter(lambda x:x.endswith('.tsv'),fs):
	ds.append(pd.read_csv(pjoin(diri1,fname),header=0,index_col=0,sep="\t",skiprows=2).T)
while len(ds)>1:
	ds2=[]
	for xi in range(len(ds)//2):
		ds2.append(pd.merge(ds[2*xi],ds[2*xi+1],how='outer',left_index=True,right_index=True,copy=False,suffixes=('','')))
	if len(ds)%2==1:
		ds2.append(ds[-1])
	del ds
	ds=ds2
ds=ds[0].T

ds=ds[list(filter(lambda x:len(ds[x].unique())>1,ds.columns))]
ds=ds.rename({'amp_batch_id':'batcha','seq_batch_id':'batchs','NKI_plate_id':'batchp','location':'loc','patient_id':'donor','mc_group':'celltype'},axis=1)[['batcha','batchs','batchp','live','loc','donor','celltype','processing']]
ds['loc']=ds['loc'].map(lambda x:x.replace(')','').replace('(',''))

#Subset of cells
for xi in subsets:
	ds=ds[ds[xi].isin(subsets[xi])]
dsc=ds.copy()
names0=set(list(dsc.index))

#Read reads data
diri1=pjoin(diri,'reads')
fs=os.listdir(diri1)
ds=[]
for fname in fs:
	t1=pd.read_csv(pjoin(diri1,fname),header=0,index_col=0,sep="\t")
	ds.append(t1[list(filter(lambda x:x in names0,t1.columns[np.random.rand(t1.shape[1])>drop_rate]))].copy())
	del t1
t1=list(itertools.chain.from_iterable([x.columns for x in ds]))
assert len(t1)==len(set(t1))

while len(ds)>1:
	ds2=[]
	for xi in range(len(ds)//2):
		ds2.append(pd.merge(ds[2*xi],ds[2*xi+1],how='outer',left_index=True,right_index=True,copy=False))
	if len(ds)%2==1:
		ds2.append(ds[-1])
	del ds
	ds=ds2
ds=ds[0]
ds.sort_index(axis=0,inplace=True)
ds.sort_index(axis=1,inplace=True)
ds.fillna(0,inplace=True)

namet=np.array([str(x) for x in ds.index])
names=np.array([str(x) for x in ds.columns])
dt=ds.values.astype('u4')
t1=dt.sum(axis=1)>0
namet=namet[t1]
dt=dt[t1]
del ds

dsc=dsc.loc[names].copy()
def genc(d,sep='='):
	assert d.isna().sum().sum()==0
	namec=[]
	dc=[]
	for xi in d.columns:
		assert sep not in xi
		t1=d[xi].unique()
		if len(t1)<=1:
			continue
		for xj in t1:
			namec.append('{}{}{}'.format(xi,sep,xj))
			dc.append(d[xi].values==xj)
	namec=np.array(namec)
	dc=np.array(dc).astype(float)
	return dc,namec
dc,namec=genc(dsc)

#Output data
np.savetxt(pjoin(diro,'read.tsv.gz'),dt,delimiter='\t',fmt='%u')
np.savetxt(pjoin(diro,'cov.tsv.gz'),dc,delimiter='\t',fmt='%.8G')
with open(pjoin(diro,'gene.txt'),'w') as f:
	f.write(linesep.join(namet))
with open(pjoin(diro,'cell.txt'),'w') as f:
	f.write(linesep.join(names))
with open(pjoin(diro,'cov.txt'),'w') as f:
	f.write(linesep.join(namec))


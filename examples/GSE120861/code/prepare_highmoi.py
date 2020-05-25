#!/usr/bin/python3

import logging
import numpy as np
import pandas as pd
import gzip
import itertools
from scipy.io import mmread,mmwrite
from scipy.sparse import coo_matrix
from os.path import join as pjoin
from os import linesep

np.random.seed(12345)
#Number of NTC gRNAs to keep
ng_negselect=15
#Number of TSS targeting gRNAs to keep
ng_tssselect=15
#Number of  non-NTC, non-TSS-targeting gRNAs to keep
ng_other=0
#Drop rate of cells without any gRNAs that was kept
droprate_ng=0.98
#Drop rate of cells with any gRNAs that was kept
droprate_g=0.25
dtype='f8'
fmt_f="%.8G"
fmt_i="%i"

diri='data'
diro='data/highmoi'
fb='reads/highmoi_cells.txt.gz'
fc='reads/highmoi_phenoData.txt.gz'
fg='reads/highmoi_genes.txt.gz'
fm='reads/highmoi_exprs.mtx'
fmmc1='meta/mmc1.csv'
fmmc2='meta/mmc2.csv'

fb,fc,fg,fm,fmmc1,fmmc2=[pjoin(diri,x) for x in [fb,fc,fg,fm,fmmc1,fmmc2]]
#mmc files
logging.info('Reading file '+fmmc1)
dmap1=pd.read_csv(fmmc1,header=0,index_col=0)
logging.info('Reading file '+fmmc2)
dmap2=pd.read_csv(fmmc2,header=0,index_col=0)
#Normalize column names
dmap2.rename({'Category':'category'},inplace=True,axis=1)
assert (dmap1.columns==dmap2.columns).all()
#Normalize values
t1={'TSS':'TSS_control',
	'Positive_control_to_globin_locus':'positive_globin_locus_control',
	'candidate_enhancer:repeated_from_pilot:top_gRNA_pair':'candidate_enhancer',
   'candidate_enhancer:picked_by_model_built_from_pilot':'candidate_enhancer',
   'candidate_enhancer:picked_by_exploratory_submodular_selection':'candidate_enhancer',
   'candidate_enhancer:repeated_from_pilot:alternative_gRNA_pair':'candidate_enhancer'}
dmap2['category']=dmap2['category'].apply(lambda x: x if x not in t1 else t1[x])
dmap2['Target_Site']=dmap2['Target_Site'].apply(lambda x:x if x!='11-Sep' else 'SEPT11')
dmap2['Target_Site']=dmap2['Target_Site'].apply(lambda x:'TSS_'+x if not x.startswith('chr') and not x.startswith('originalIDTorder:') and not x.startswith('pos_control_') and x not in {'control','non-targeting','cag_promoter','randomregion'} else x)
#Fill missing values
t0=list(set(dmap1.index)&set(dmap2.index))
td1=dmap1.loc[t0]
td2=dmap2.loc[t0]
for c in td1.columns:
	t1=td1[c].isna()
	t2=td1[c].copy()
	t2[t1]=td2[c][t1]
	dmap1.loc[t0,c]=t2
	t1=td2[c].isna()
	t2=td2[c].copy()
	t2[t1]=td1[c][t1]
	dmap2.loc[t0,c]=t2
#Check identical in overlap
for c in td1.columns:
	assert ((td1[c]==td2[c])|(td1[c].isna()&td2[c].isna())).all()
#Merge dmaps
dmap=[]
dmap.append(dmap1.loc[t0])
dmap.append(dmap1.loc[list(filter(lambda x:x not in t0,dmap1.index))])
dmap.append(dmap2.loc[list(filter(lambda x:x not in t0,dmap2.index))])
dmap=pd.concat(dmap,axis=0,verify_integrity=True)

#names
logging.info('Reading file '+fb)
with gzip.open(fb,'r') as f:
	names=f.read().decode()
names=names.split('\n')
names=np.array(list(filter(lambda x:len(x)>0,names)))
ns=len(names)

#dg
logging.info('Reading file '+fc)
dg=pd.read_csv(fc,sep=' ',header=None,index_col=1)
dg.rename({6:'gRNA',14:'prep',15:'chip',16:'lane'},inplace=True,axis=1)
dg=dg.loc[names]
dg=dg[['gRNA','prep','chip','lane']].copy()
if dg['prep'].dtype==bool:
	dg=dg[['gRNA']].copy()
#Empty gRNAs
t1=dg['gRNA'].copy()
t1[t1.isna()]=""
dg['gRNA']=t1

#dc
namec=['batch']
dc=np.array([[x.split('_')[-1] for x in names]])
#Convert to categorical covariates
dcn=[]
namecn=[]
for xi in range(len(namec)):
	t1=sorted(list(set(list(dc[xi]))))
	if len(t1)==1:
		continue
	elif len(t1)==2:
		dcn.append(dc[xi])
		namecn.append(namec[xi])
		continue
	for xj in t1:
		dcn.append(dc[xi]==xj)
		namecn.append(namec[xi]+'='+str(xj))
dc=np.array(dcn,dtype=dtype)
namec=np.array(namecn)

#dg & nameg
t00=[list(filter(lambda y:len(y)>0,x.split('_'))) for x in dg['gRNA']]
t1=list(set(list(itertools.chain.from_iterable(t00))))
dg=np.array([[y in x for y in dmap.index] for x in t1])
t2=dg.sum(axis=1)
assert (t2<=1).all()
t2=t2==1
logging.info('Unrecognized gRNAs: {}/{}.'.format((~t2).sum(),t2.size))
dg=[np.nonzero(x)[0] for x in dg[t2]]
t1=np.array(t1)[t2]
t1=dict(zip(t1,dg))

t00=[list(filter(lambda x:x in t1,y)) for y in t00]
nameg=np.array(list(dmap.index))
ng=len(nameg)
dg=np.zeros((ng,ns),dtype=bool)
for xi in range(len(t00)):
	dg[[t1[x] for x in t00[xi]],xi]=True
dg=dg.astype('u1')

#Remove redundant genotypes/gRNAs
t1=dg.sum(axis=1)>0
logging.info('Non-existent gRNAs: {}/{}.'.format((~t1).sum(),t1.size))
dg=dg[t1]
nameg=nameg[t1]
ng=len(nameg)

#Annotate gRNAs' categories in nameg
ans=[]
for xi in range(ng):
	t1=dmap.loc[nameg[xi],'category']
	if t1=='candidate_enhancer':
		ans.append(nameg[xi]+'_enhancerdrop')
	elif t1=='positive_globin_locus_control':
		ans.append(nameg[xi]+'_posctrldrop')
	elif t1=='NTC':
		ans.append(nameg[xi]+'_negctrl')
	elif t1=='TSS_control':
		ans.append(nameg[xi]+'_TSS_{}'.format('_'.join(dmap.loc[nameg[xi],'Target_Site'].split('_')[1:])))
	else:
		print(t1)
		assert False
nameg=np.array(ans)

#Select gRNAs
namegselect=np.zeros(len(nameg),dtype=bool)
if ng_negselect>0:
	#For NTCs
	t1=np.nonzero([x.endswith('_negctrl') for x in nameg])[0]
	if len(t1)>ng_negselect:
		t1=np.random.choice(t1,ng_negselect,replace=False)
	namegselect[t1]=True
if ng_tssselect>0:
	#For TSS-targeting
	t1=np.nonzero([x.split('_')[1]=='TSS' for x in nameg])[0]
	if len(t1)>ng_tssselect:
		t1=np.random.choice(t1,ng_tssselect,replace=False)
	namegselect[t1]=True
if ng_other>0:
	#For others
	t1=np.nonzero([x.split('_')[1]!='TSS' and not x.endswith('_negctrl') for x in nameg])[0]
	if len(t1)>ng_other:
		t1=np.random.choice(t1,ng_other,replace=False)
	namegselect[t1]=True
nameg=nameg[namegselect]
dg=dg[namegselect]
ng=len(nameg)

#Select cells
t1=dg.astype(bool).any(axis=0)
namesselect=np.random.rand(len(t1))
namesselect[t1]=namesselect[t1]>droprate_g
namesselect[~t1]=namesselect[~t1]>droprate_ng
namesselect=np.nonzero(namesselect)[0]
dg=dg[:,namesselect]
dc=dc[:,namesselect]
names=names[namesselect]
ns=len(names)
print('gRNA matrix shape: ',dg.shape)
print('Proportion of cells with gRNA: ',dg.astype(bool).any(axis=0).mean())

#namet
logging.info('Reading file '+fg)
with gzip.open(fg,'r') as f:
	namet=f.read().decode()
namet=namet.split('\n')
namet=np.array(list(filter(lambda x:len(x)>0,namet)))
nt=len(namet)

#dt
logging.info('Reading file '+fm)
dt=mmread(fm).astype(int)
dt=dt.tocsr()
dt=dt[:,namesselect]
dt=dt.toarray()
dt=coo_matrix(dt)
assert dt.shape==(nt,ns)

#Output data
with gzip.open(pjoin(diro,'0_read.mtx.gz'),'w') as f:
    mmwrite(f,dt,field='integer')
np.savetxt(pjoin(diro,'0_cov.tsv.gz'),dc,delimiter='\t',fmt=fmt_f)
np.savetxt(pjoin(diro,'0_group.tsv.gz'),dg,delimiter='\t',fmt=fmt_i)
with open(pjoin(diro,'0_gene.txt'),'w') as f:
	f.write(linesep.join(namet))
with open(pjoin(diro,'0_cell.txt'),'w') as f:
	f.write(linesep.join(names))
with open(pjoin(diro,'0_cov.txt'),'w') as f:
	f.write(linesep.join(namec))
with open(pjoin(diro,'0_gRNA.txt'),'w') as f:
	f.write(linesep.join(nameg))

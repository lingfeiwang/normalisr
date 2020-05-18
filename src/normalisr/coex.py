#!/usr/bin/python3

def coex(dt,dc,bs=0,nth=1,**ka):
	"""Performs co-expression analyses for all gene pairs.
	Parallel computation with multiple processes on the same machine.

	Model for co-expression between genes i & j:
		X_i=gamma*X_j+alpha*C+epsilon
		epsilon~i.i.d. N(0,sigma**2)
	Statistic: R^2 (or proportion of variance explained)
	Null hypothesis: gamma=0.

	dt:		numpy.array(shape=(n_gene,n_cell)). Target matrix for a list of Y to be tested, e.g. gene expression.
	dc:		numpy.array(shape=(n_cov,n_cell)). Covariate matrix C.
	bs:		Number of genes in each job. Use 0 for default: Data transfer limited to 1GB, capped at bs=500.
	nth:	Number of processes. 0 for using automatically detected CPU counts.
	ka:		Keyword arguments passed to .association.association_test_1.
	Return:	(p-values,dot,var)
	p-values:	numpy.array(shape=(n_gene,n_gene)).
	dot:		numpy.array(shape=(n_gene,n_gene)). Inner product of gene expressions after removing covariates.
	var:		numpy.array(shape=(n_gene)). Full variance of dt after removing covariates"""
	import numpy as np
	import logging
	import itertools
	from .association import inv_rank,association_test_1
	from .parallel import autopooler
	nt,ns=dt.shape
	nc=dc.shape[0]
	assert nth>=0
	if bs==0:
		#Transfer 1GB data max
		bs=int((2**30)//(2*dt.dtype.itemsize*ns))-nc
		bs=min(bs,500)
		logging.info('Using automatic batch size: {}'.format(bs))
	assert bs>0
	ka0=dict(ka)
	samexy=True

	if dc.shape[0]>0 and (dc!=0).any():
		dci,dcr=inv_rank(np.matmul(dc,dc.T))
	else:
		dci=None
		dcr=0

	it=itertools.product(*itertools.tee(map(lambda x:[x,min(x+bs,nt)],range(0,nt,bs))))
	if samexy:
		it=filter(lambda x:x[0][0]<=x[1][0],it)
	it=map(lambda x:[association_test_1,(x[0][0],x[1][0],dt[x[0][0]:x[0][1]],dt[x[1][0]:x[1][1]],dc,dci,dcr),ka0],it)
	ans0=autopooler(nth,it)
	assert len(ans0)>0

	ansdot=np.empty((nt,nt),dtype=dt.dtype)
	ansp=np.empty((nt,nt),dtype=dt.dtype)
	ansvar=np.zeros((nt,),dtype=dt.dtype)
	for xi in ans0:
		ansp[xi[0]:xi[0]+xi[2].shape[0],xi[1]:xi[1]+xi[2].shape[1]]=xi[2]
		ansdot[xi[0]:xi[0]+xi[3].shape[0],xi[1]:xi[1]+xi[3].shape[1]]=xi[3]
		ansvar[xi[0]:xi[0]+xi[5].shape[0]]=xi[5]
		ansvar[xi[1]:xi[1]+xi[6].shape[0]]=xi[6]
	del ans0

	#Convert from coef to dot
	ansdot=(ansdot.T*ansvar).T
	if samexy:
		ansdot=np.triu(ansdot,1)
		ansdot=ansdot+ansdot.T
		ansp=np.triu(ansp,1)
		ansp=ansp+ansp.T

	assert ansdot.shape==(nt,nt) and ansp.shape==(nt,nt) and ansvar.shape==(nt,)
	assert np.isfinite(ansdot).all() and np.isfinite(ansp).all() and np.isfinite(ansvar).all()
	assert (ansvar>0).all()
	assert (ansp>=0).all() and (ansp<=1).all()
	assert (((ansdot**2)/ansvar).T/ansvar).max()<=1
	return (ansp,ansdot,ansvar)










































assert __name__ != "__main__"

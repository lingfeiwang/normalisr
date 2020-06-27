#!/usr/bin/python3

def coex(dt,dc,bs=0,nth=1,**ka):
	"""Performs co-expression analyses for all gene pairs.

	Performs parallel computation with multiple processes on the same machine.

	Model for co-expression between genes i & j:
		X_i=gamma*X_j+alpha*C+epsilon,

		epsilon~i.i.d. N(0,sigma**2).

	Test statistic: conditional R**2 (or proportion of variance explained) between X_i and X_j.

	Null hypothesis: gamma=0.

	Parameters
	----------
	dt:		numpy.ndarray(shape=(n_gene,n_cell),dtype=float)
		Normalized expression matrix X.
	dc:		numpy.ndarray(shape=(n_cov,n_cell),dtype=float)
		Normalized covariate matrix C.
	bs:		int
		Batch size, i.e. number of genes in each computing batch. Use 0 for default: Data transfer limited to 1GB, capped at bs=500.
	nth:	int
		Number of parallel processes. Set to 0 for using automatically detected CPU counts.
	ka:		dict
		Keyword arguments passed to normalisr.association.association_test_1. See below.

	Returns
	-------
	P-values:	numpy.ndarray(shape=(n_gene,n_gene))
		Co-expression P-value matrix.
	dot:		numpy.ndarray(shape=(n_gene,n_gene))
		Inner product of expression between gene pairs, after removing covariates.
	var:		numpy.ndarray(shape=(n_gene))
		Variance of gene expression after removing covariates. Pearson R=(((dot/numpy.sqrt(var)).T)/numpy.sqrt(var)).T.

	Keyword arguments
	-----------------
	dimreduce:	numpy.ndarray(shape=(n_gene,),dtype=int) or int
		If dt doesn't have full rank, such as due to prior covariate removal (although the recommended method is to leave covariates in dc), this parameter allows to specify the loss of ranks/degrees of freedom to allow for accurate P-value computation. Default is 0, indicating no rank loss.
	"""
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

#!/usr/bin/python3

def de(dg,dt,dc,bs=0,nth=1,single=0,**ka):
	"""Performs differential expression analyses for all genes against all groupings.

	Allows multiple options to treat other groupings when testing on one grouping.

	Performs parallel computation with multiple processes on the same machine.

	Model for differential expression between gene Y and grouping X:
		Y=gamma*X+alpha*C+epsilon,

		epsilon~i.i.d. N(0,sigma**2).

	Test statistic: conditional R**2 (or proportion of variance explained) between Y and X.

	Null hypothesis: gamma=0.

	Parameters
	-----------
	dg:		numpy.ndarray(shape=(n_group,n_cell)).
		Grouping matrix for a list of X to be tested, e.g. grouping by gene knock-out.
	dt:		numpy.ndarray(shape=(n_gene,n_cell),dtype=float)
		Normalized expression matrix Y.
	dc:		numpy.ndarray(shape=(n_cov,n_cell),dtype=float)
		Normalized covariate matrix C.
	bs:		int
		Batch size, i.e. number of groupings and genes in each computing batch.
		For single=0,1, splits groupings & genes. Defaults to 500.
		For single=4, only allows bs=0 (default automatic setting).
	nth:	int
		Number of parallel processes. Set to 0 for using automatically detected CPU counts.
	single:	int
		Option to deal with other groupings when testing one groupings v.s. gene expression.

		* 0:	Ignores other groupings.
		* 1:	Excludes all cells belonging to any other grouping (value==1), assuming dg=0,1 only.
		* 4:	Treats other groupings as covariates for mean expression.

	ka:		dict
		Keyword arguments passed to normalisr.association.association_test_*. See below.

	Returns
	--------
	P-values:	numpy.ndarray(shape=(n_group,n_gene))
		Differential expression P-value matrix.
	gamma:		numpy.ndarray(shape=(n_group,n_gene))
		Differential expression log fold change matrix.
	alpha:		numpy.ndarray(shape=(n_group,n_gene,n_cov))
		Maximum likelihood estimators of alpha, separatly tested for each grouping.
	varg:		numpy.ndarray(shape=(n_group))
		Variance of grouping after removing covariates.
	vart:		numpy.ndarray(shape=(n_group,n_gene))
		Variance of gene expression after removing covariates.
		It can depend on the grouping being tested depending on parameter single.

	Keyword arguments
	--------------------------------
	dimreduce:	numpy.ndarray(shape=(n_gene,),dtype=int) or int
		If dt doesn't have full rank, such as due to prior covariate removal (although the recommended method is to leave covariates in dc), this parameter allows to specify the loss of ranks/degrees of freedom to allow for accurate P-value computation. Default is 0, indicating no rank loss.
	method:		str, only for single=4
		Method to compute eigenvalues in SVD-based matrix inverse (for removal of covariates):

		* auto:	Uses scipy for n_matrix<mpc or mpc==0 and sklearn otherwise. Default.
		* scipy: Uses scipy.linalg.svd.
		* scipys: NOT IMPLEMENTED.Uses scipy.sparse.linalg.svds.
		* sklearn: Uses sklearn.decomposition.TruncatedSVD.

	mpc:		int, only for single=4
		Uses only the top mpc singular values as non-zero in SVD-based matrix inverse. Here effectively reduces covariates to their top principal components. This reduction is performed after including other groupings as additional covariates. Defaults to 0 to disable dimension reduction. For very large grouping matrix, use a small value (e.g. 100) to save time at the cost of accuracy.
	qr:			int, only for single=4
		Whether to use QR decomposition method for SVD in matrix inverse. Only effective when method=sklearn, or =auto and defaults to sklearn. Takes the following values:

		* 0:	No (default).
		* 1:	Yes with default settings.
		* 2+:	Yes with n_iter=qr for sklearn.utils.extmath.randomized_svd.

	tol:		float, only for single=4
		Eigenvalues < tol*(maximum eigenvalue) are treated as zero in SVD-based matrix inverse. Default is 1E-8.
	"""
	import numpy as np
	import logging
	import itertools
	from .association import association_test_1,association_test_2,association_test_4,prod1
	from .parallel import autopooler
	if single==3:
		raise NotImplementedError('single=3 is obsolete. Please use single=4 for the same function with improved efficiency.')
	bst=2000
	ka0=dict(ka)

	#Ignore single-valued genotypes
	dg0=dg
	gid=[len(np.unique(x))>1 for x in dg]
	dg=dg0[gid]

	ng,ns=dg.shape
	nt=dt.shape[0]
	nc=dc.shape[0]
	if single==1:
		assert dg.max()==1
		t1=dg.sum(axis=0)
		t1=dg==t1
		for xi in range(ng):
			assert len(np.unique(dg[xi,t1[xi]]))>1
		sselectx=t1

	if single in {0,1}:
		if bs==0:
			bs=500
		it=itertools.product(map(lambda x:[x,min(x+bs,ng)],range(0,ng,bs)),map(lambda x:[x,min(x+bs,nt)],range(0,nt,bs)))
		if single==0:
			from .association import inv_rank
			if nc>0 and (dc!=0).any():
				dci,dcr=inv_rank(np.matmul(dc,dc.T))
			else:
				dci=None
				dcr=0
			it=map(lambda x:[association_test_1,(x[0][0],x[1][0],dg[x[0][0]:x[0][1]],dt[x[1][0]:x[1][1]],dc,dci,dcr),ka0],it)
		elif single==1:
			it=map(lambda x:[association_test_2,(x[0][0],x[1][0],dg[x[0][0]:x[0][1]],dt[x[1][0]:x[1][1]],dc,sselectx[x[0][0]:x[0][1]]),ka0],it)

		ans0=autopooler(nth,it,dummy=True)
		assert len(ans0)>0
		ans=np.ones((ng,nt),dtype=dt.dtype)
		ansc=np.zeros((ng,nt),dtype=dt.dtype)
		ansa=np.zeros((ng,nt,nc),dtype=dt.dtype)
		ansvg=np.zeros((ng),dtype=dt.dtype)
		ansvt=np.zeros((ng,nt),dtype=dt.dtype)
		for xi in ans0:
			ans[xi[0]:xi[0]+xi[2].shape[0],xi[1]:xi[1]+xi[2].shape[1]]=xi[2]
			ansc[xi[0]:xi[0]+xi[3].shape[0],xi[1]:xi[1]+xi[3].shape[1]]=xi[3]
			ansa[xi[0]:xi[0]+xi[4].shape[0],xi[1]:xi[1]+xi[4].shape[1]]=xi[4]
			ansvg[xi[0]:xi[0]+xi[5].shape[0]]=xi[5]
			if single==0:
				ansvt[:,xi[1]:xi[1]+xi[6].shape[0]]=xi[6]
			elif single==1:
				ansvt[xi[0]:xi[0]+xi[6].shape[0],xi[1]:xi[1]+xi[6].shape[1]]=xi[6]
			else:
				assert False
		del ans0
	elif single in {4}:
		#Note: a single linear regression on all X and C doesn't work.
		#Because the variance of each X unexplained by other X + C is unknown.
		if bs==0:
			bs1=500
			bs2=10
			bs3=500000
		else:
			raise NotImplementedError('Only allowing default setting (bs=0) for single=4')
		#Compute matrix products
		t1=np.concatenate([dg,dc],axis=0)
		it=itertools.product(map(lambda x:[x,min(x+bs1,ng+nc)],range(0,ng+nc,bs1)),map(lambda x:[x,min(x+bs1,ng+nc)],range(0,ng+nc,bs1)))
		it=filter(lambda x:x[0][0]<=x[1][0],it)
		it=map(lambda x:[prod1,(x[0][0],x[1][0],t1[x[0][0]:x[0][1]],t1[x[1][0]:x[1][1]]),dict()],it)
		ans0=autopooler(nth,it,dummy=True)
		tprod=np.zeros((ng+nc,ng+nc),dtype=dt.dtype)
		for xi in ans0:
			tprod[np.ix_(range(xi[0],xi[0]+xi[2].shape[0]),range(xi[1],xi[1]+xi[2].shape[1]))]=xi[2]
		del ans0
		tprod=np.triu(tprod).T+np.triu(tprod,1)
		it=itertools.product(map(lambda x:[x,min(x+bs1,ng+nc)],range(0,ng+nc,bs1)),map(lambda x:[x,min(x+bs1,nt)],range(0,nt,bs1)))
		it=map(lambda x:[prod1,(x[0][0],x[1][0],t1[x[0][0]:x[0][1]],dt[x[1][0]:x[1][1]]),dict()],it)
		ans0=autopooler(nth,it,dummy=True)
		del t1
		tprody=np.zeros((ng+nc,nt),dtype=dt.dtype)
		for xi in ans0:
			tprody[np.ix_(range(xi[0],xi[0]+xi[2].shape[0]),range(xi[1],xi[1]+xi[2].shape[1]))]=xi[2]
		del ans0
		tprodyy=(dt**2).sum(axis=1)
		#Compute DE
		it=itertools.product(map(lambda x:[x,min(x+bs2,ng)],range(0,ng,bs2)),map(lambda x:[x,min(x+bs3,nt)],range(0,nt,bs3)))
		it=map(lambda x:[association_test_4,(x[0][0],x[1][0],tprod,tprody[:,x[1][0]:x[1][1]],tprodyy[x[1][0]:x[1][1]],[ng,x[1][1]-x[1][0],nc,ns,x[0][1]-x[0][0]]),ka0],it)
		#Decide on dummy based on SVD method
		if 'method' in ka0 and ka0['method']=='scipy':
			isdummy=False
		elif 'method' in ka0 and ka0['method']=='sklearn':
			isdummy=True
		elif 'method' not in ka0 or ka0['method']=='auto':
			isdummy='mpc' in ka0 and ka0['mpc']!=0 and ka0['mpc']<ng-1
		else:
			isdummy=True
		ans0=autopooler(nth,it,dummy=isdummy)
		assert len(ans0)>0
		del tprod,tprody,tprodyy
		#Collect results
		ans=np.ones((ng,nt),dtype=dt.dtype)
		ansc=np.zeros((ng,nt),dtype=dt.dtype)
		ansa=np.zeros((ng,nt,nc),dtype=dt.dtype)
		ansvg=np.zeros((ng,),dtype=dt.dtype)
		ansvt=np.zeros((ng,nt),dtype=dt.dtype)
		for xi in ans0:
			ans[xi[0]:xi[0]+xi[2].shape[0],xi[1]:xi[1]+xi[2].shape[1]]=xi[2]
			ansc[xi[0]:xi[0]+xi[3].shape[0],xi[1]:xi[1]+xi[3].shape[1]]=xi[3]
			ansa[xi[0]:xi[0]+xi[4].shape[0],xi[1]:xi[1]+xi[4].shape[1]]=xi[4]
			ansvg[xi[0]:xi[0]+xi[5].shape[0]]=xi[5]
			ansvt[xi[0]:xi[0]+xi[6].shape[0],xi[1]:xi[1]+xi[6].shape[1]]=xi[6]
		del ans0
	else:
		raise ValueError('Unknown value {} for variable single'.format(single))

	t1=np.ones((dg0.shape[0],nt),dtype=dt.dtype)
	t1[gid]=ans
	ans=t1
	t1=np.zeros((dg0.shape[0],nt),dtype=dt.dtype)
	t1[gid]=ansc
	ansc=t1
	t1=np.zeros((dg0.shape[0],nt,nc),dtype=dt.dtype)
	t1[gid]=ansa
	ansa=t1
	t1=np.zeros((dg0.shape[0],),dtype=dt.dtype)
	t1[gid]=ansvg
	ansvg=t1
	t1=np.zeros((dg0.shape[0],nt),dtype=dt.dtype)
	t1[gid]=ansvt
	ansvt=t1
	assert ans.shape==(dg0.shape[0],nt) and ansc.shape==(dg0.shape[0],nt) and ansa.shape==(dg0.shape[0],nt,nc) and ansvg.shape==(dg0.shape[0],) and ansvt.shape==(dg0.shape[0],nt)
	assert np.isfinite(ans).all() and np.isfinite(ansc).all() and np.isfinite(ansa).all() and np.isfinite(ansvg).all() and np.isfinite(ansvt).all()
	assert (ans>=0).all() and (ans<=1).all() and (ansvg>=0).all() and (ansvt>=0).all()
	return (ans,ansc,ansa,ansvg,ansvt)











































assert __name__ != "__main__"

#!/usr/bin/python3

def de(dg,dt,dc,bs=0,nth=1,single=0,**ka):
	"""Performs differential expression analyses for all genes against all predictors.
	Allows multiple methods to treat other predictors when testing on one predictor.
	Parallel computation with multiple processes on the same machine.

	Model:
		Y=gamma*X+alpha*C+epsilon
		epsilon~i.i.d. N(0,sigma**2)
	Statistic: R^2 (or the proportion of variance explained)
	Null hypothesis: gamma=0.

	dg:		numpy.array(shape=(n_group,n_cell)). Predictor matrix for a list of X to be tested, e.g. grouping by gene knock-out.
	dt:		numpy.array(shape=(n_gene,n_cell)). Target matrix for a list of Y to be tested, e.g. gene expression.
	dc:		numpy.array(shape=(n_cov,n_cell)). Covariate matrix C.
	bs:		Number of genotypes & genes in each job.
		For single=0,1, splits genotypes & genes. Defaults to 500.
		For single=4, only allows bs=0 (default automatic setting).
	nth:	Number of parallel processes. 0 for using automatically detected CPU counts.
	single:	How to deal with other KOs when testing one KO vs no KO
			0:	Ignore the value of other KOs
			1:	Exclude all samples with any other KO. Assumes dg=0,1 only.
			4:	Treat other KOs as covariates.
	ka:		Keyword args passed .association.association_test_?.
	Return:	[p-values,gamma,alpha,varg,vart]
		p-values:	numpy.array(shape=(n_group,n_gene)).
		gamma:		numpy.array(shape=(n_group,n_gene)). Also as log fold change
		alpha:		numpy.array(shape=(n_group,n_gene,n_cov)).
		varg:		numpy.array(shape=(n_group)). Full variance of dg after removing covariates
		vart:		numpy.array(shape=(n_group,n_gene)). Full variance of dt after removing covariates"""
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
	return [ans,ansc,ansa,ansvg,ansvt]











































assert __name__ != "__main__"

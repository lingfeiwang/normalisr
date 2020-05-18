#!/usr/bin/python3

def normcov(dc,c=True):
	"""To normalize continuous covariates to 0 mean and unit variance.
	Introduce constant 1 covariate as intercept.
	Categorical covariates should be in binary/dummy form.
	Binary covariates are left unchanged.
	dc:	numpy.array(shape=[n_cov,n_cell]). Existing covariates. Each row is a covariate.
		Each column is a cell/sample. No covariate means n_cov=0.
	c:	Whether to add a constant 1 covariate
	Return: numpy.array(shape=(n_cov+1 if c else n_cov,n_cell)) Processed covariates."""
	import numpy as np
	import logging,warnings
	assert dc is not None
	#Validty checks
	if dc.ndim!=2:
		raise ValueError('Covariates must have 2 dimensions.')
	nsa=dc.shape[1]
	if dc.shape[0]==0:
		if c:
			return np.ones((1,nsa))
		else:
			return dc
	if np.any([len(np.unique(x))==1 for x in dc]):
		raise ValueError('Detected constant covariate. Please only provide full-rank covariate without constant covariate.')

	t0=((dc!=0)&(dc!=1)).any(axis=1)
	#Normalize covariates
	t1=dc[t0].mean(axis=1)
	dc2=dc[t0].T-t1
	t2=np.sqrt((dc2**2).mean(axis=0))
	if t2.min()<1E-50 or np.abs(t2/t1).min()<1E-6:
		warnings.warn('Detected near constant covariate. Results may be error-prone.',RuntimeWarning)
	dc2=(dc2/t2).T
	dc=dc.copy()
	dc[t0]=dc2
	if c:
		dc=np.concatenate([dc,np.ones((1,nsa))],axis=0)
	return dc

def compute_var(dt,dc,stepmax=1,eps=1E-6):
	"""To fit the variance of each cell with covariates.
	Use EM-like method to fit a log-linear model of variance using covariates.
	dt:		numpy.array(shape=[n_gene,n_cell]). Log expression levels. Each row is a gene.
			Each column is a cell/sample.
	dc:		numpy.array(shape=[n_cov,n_cell]). Existing covariates. Each row is a covariate.
			Each column is a cell/sample.
	stepmax:Maximum number of steps to stop regardless of convergence.
	eps:	Precision target for early stopping. Constrains the maximum relative difference of
			fitted variance across cells compared to the last step.
	Return:	variances**(-0.5), i.e. the multiplier for normalization, as numpy.array(shape=[n_cell]).
			The optimal step will be returned, defined as the least max relative change across cells."""
	import numpy as np
	from sklearn.linear_model import LinearRegression as lr0
	import logging
	if eps<=0 or stepmax<=0:
		raise ValueError('eps and stepmax must be positive.')
	if dt.ndim!=2 or dc.ndim!=2:
		raise ValueError('dt and dc must both have 2 dimensions.')
	if dt.shape[1]!=dc.shape[1]:
		raise ValueError('dt and dc must have the same cell count.')

	d1=dt
	dx=dc
	ns=dc.shape[1]
	lr=lr0(normalize=False,fit_intercept=False)
	lr2=lr0(fit_intercept=True)

	d1sscale=np.ones(ns)
	best=None
	bestv=1E300
	n=0
	while n<stepmax and bestv>eps:
		#Scale data for weighted linear regression to fit mean
		td1=d1/d1sscale
		tdx=dx/d1sscale
		lr.fit(tdx.T,td1.T)
		td1=td1-lr.predict(tdx.T).T

		#Fit log variance
		d1mean=td1.mean(axis=1)
		d1scale=np.sqrt(((td1.T-d1mean)**2).mean(axis=0))
		d1sscalenew=np.log(np.sqrt((((td1.T-d1mean)/d1scale)**2).mean(axis=1)))
		lr2.fit(dx.T,d1sscalenew.T)
		d1sscalenew=lr2.predict(dx.T).flatten()

		#Determine relative difference
		d1sscalenew=np.exp(d1sscalenew)*d1sscale
		d1sscalenew/=d1sscalenew.min()
		t1=np.abs((d1sscalenew-d1sscale)/d1sscale).max()
		d1sscale=d1sscalenew
		n+=1
		if t1<bestv:
			bestv=t1
			best=d1sscale
		logging.debug('Step {}, maximum relative difference: {}'.format(n,t1))

	d1sscale=1/best.astype(float)
	d1sscale/=d1sscale.min()
	assert d1sscale.shape==(ns,)
	assert np.isfinite(d1sscale).all()
	assert (d1sscale>0).all()
	return d1sscale

def normvar(dt,dc,w,wt,dextra=None,cat=1,keepvar=True):
	"""To linearly normalize the variance of expression using covariates,
	accounting for different scaling factors for different genes.
	Also to transform the covariates to reflect the normalization.
	Each gene x is multiplied by w**wt[x] before removing covariates as dc*(w**wt[x]).
	Covariates are transformed to dc*w.
	dt:		numpy.array(shape=[n_gene,n_cell]). Bayesian logCPM.
	dc:		numpy.array(shape=[n_cov,n_cell]). Existing covariates.
	w:		numpy.array(shape=[n_cell]). Multiplier to normalize variance.
	wt:		numpy.array(shape=[n_gene]). Scaling factor for each gene.
	dextra:	numpy.array(shape=[n_extra,n_cell]). Extra data only to be normalized like continuous covariates.
	cat:	Whether to normalize categorical covariates (those with only 0 or 1s).
		0:	No
		1:	No except constant-1 covariate
		2:	Yes
	keepvar:Whether to maintain the variance of each gene invariant in covariate removal step.
			Only affects overall variance level and its downstreams (e.g. differential expression log fold change).
			Would not affect P-value computation.
	Return: [dtn,dcn] or [dtn,dcn,dextran] if dextra is not None
	dtn:	numpy.array(shape=[n_gene,n_cell]) Transformed gene expressions.
	dcn:	numpy.array(shape=[n_cov,n_cell]) Transformed covariates.
	dextran:numpy.array(shape=[n_extra,n_cell]) Transformed extra variables.
	"""
	import numpy as np
	from .association import inv_rank
	if np.any([x.ndim!=2 for x in [dt,dc]]):
		raise ValueError('dt and dc should have 2 dimensions.')
	if np.any([x.ndim!=1 for x in [w,wt]]):
		raise ValueError('w and wt should have 1 dimension.')
	nt,ns=dt.shape
	if dc.shape[0]==0:
		raise ValueError('No covariates.')

	if dc.shape[1]!=ns or w.shape[0]!=ns or wt.shape[0]!=nt:
		raise ValueError('Unmatched gene or cell counts.')
	if dextra is not None and (dextra.ndim!=2 or dextra.shape[0]==0 or dextra.shape[1]!=ns):
		raise ValueError('Unmatched shape or size for dextra.')
	if w.min()<=0:
		raise ValueError('w must be positive.')
	if wt.min()<0:
		raise ValueError('wt must be non-negative.')

	#Apply normalization
	w2=(np.repeat([w],nt,axis=0).T**(wt)).T
	w2[wt==0]=1
	dt=dt*w2
	if keepvar:
		dv=dt.mean(axis=1)
		dv=np.sqrt(((dt.T-dv)**2).mean(axis=0))
	#Remove covariates
	dtn=[]
	for xi in range(nt):
		dc1=dc*w2[xi]
		t1=np.matmul(dc1,dc1.T)
		t1i,r=inv_rank(t1)
		if r<=0:
			raise RuntimeError('Zero-rank covariates found.')
		dtn.append(dt[xi]-np.matmul(dc1.T,np.matmul(t1i,np.matmul(dc1,dt[xi]))))
	dtn=np.array(dtn)
	if keepvar:
		dv2=np.sqrt((dtn**2).mean(axis=1))
		dtn=(dtn.T*((dv/dv2)**wt)).T

	if cat==2:
		dcn=dc*w
	elif cat==1:
		dcn=dc.copy()
		t0=((dc!=0)&(dc!=1)).any(axis=1)|((dc==1).all(axis=1))
		dcn[t0]=dc[t0]*w
	elif cat==0:
		dcn=dc.copy()
		t0=((dc!=0)&(dc!=1)).any(axis=1)
		dcn[t0]=dc[t0]*w
	else:
		raise ValueError('Invalid cat value.')
	ans=[dtn,dcn]
	assert dtn.shape==dt.shape and dcn.shape==dc.shape
	assert np.isfinite(dtn).all() and np.isfinite(dcn).all()
	if dextra is not None:
		assert dextra.shape[0]>0 and dextra.shape[1]==dc.shape[1]
		dextran=dextra*w
		assert dextran.shape==dextra.shape
		assert np.isfinite(dextran).all()
		ans.append(dextran)
	return ans










































assert __name__ != "__main__"

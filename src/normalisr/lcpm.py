#!/usr/bin/python3

def trigamma(x):
	from scipy.special import polygamma
	return polygamma(1,x)

def lcpm(reads,seed=None,nth=0,ntot=None,varscale=0,normalize=True):
	"""Bayesian model infer or resample log expression proportion from its posterior distribution based on read counts.
	reads:	Read counts as numpy.array(shape=(n_gene,n_cell),dtype=uint)
	seed:	Initial random seed if set.
	ntot:	Manually set of total number of UMIs in binomial distribution.
			Since the posterior distribution stablizes quickly as ntot increases, a large number, e.g. 1E9 is good for general use.
	varscale:	Variance is scaled by varscale before resampling data. Set to 0 (default) for expectation values.
	normalize:	Whether to normalize output to unit sum per cell.
	nth:	Number of threads to use. Set to 0 to use all cores automatically detected.

	Return:	(dnew,mean,var,cov)
	dnew:	Generated log expression proportion as numpy.array(shape=(n_gene,n_cell))
			Satisfies (numpy.exp(dnew).sum(axis=0)==1E6).all() if normalize
	mean:	Mean/Expectation of dnew's every entry's posterior distribution as numpy.array(shape=(n_gene,n_cell))
	var:	Variance of dnew's every entry's posterior distribution as numpy.array(shape=(n_gene,n_cell))
	cov:	Covariates introduced in dnew due to this generative model, as numpy.array(shape=(3,n_cell))
			cov[0]: Log total read count per cell
			cov[1]: Number of 0-read genes per cell
			cov[2]:	cov[0]**2

	Dimensions:
	n_cell:	Total number of cells
	n_gene:	Number of genes/transcripts
	"""
	d=reads
	import numpy as np
	from scipy.stats import beta
	import scipy
	from scipy.special import digamma
	from .parallel import autopooler,autocount
	if nth==0:
		nth1=autocount()
	else:
		nth1=nth

	if d.ndim!=2:
		raise ValueError('reads must have 2 dimensions.')
	if seed is not None:
		np.random.seed(seed)
	issparse=scipy.sparse.issparse(d)
	if ((d.data if issparse else d)<0).any():
		raise ValueError('Negative value in d detected.')
	if varscale<0:
		raise ValueError('varscale must be non-negative.')

	nt,nc=d.shape
	t0=d.sum()+2 if ntot is None else ntot+2
	assert t0>2
	t1=[digamma(t0),trigamma(t0)]
	t1=[float(x) if hasattr(x,'shape') else x for x in t1]

	#Paralleled gammas
	dvs=np.concatenate([[0],np.unique((d.data if issparse else d).flatten())])
	t4=dvs.max()
	t2=autopooler(nth1,map(lambda x:[digamma,[x],dict()],np.array_split(1+dvs,nth1)),dummy=True)
	t2=dict(zip(dvs,np.concatenate(list(t2))-t1[0]))
	t2=np.array([t2[x] if x in t2 else 0 for x in np.arange(t4+1)])
	if varscale!=0:
		t3=autopooler(nth1,map(lambda x:[trigamma,[x],dict()],np.array_split(1+dvs,nth1)))
		t3=dict(zip(dvs,np.concatenate(list(t3))-t1[1]))
		t3=np.array([t3[x] if x in t3 else 0 for x in np.arange(t4+1)])

	if issparse and d.size<np.prod(d.shape)/100:
		#Sparse
		t4=np.array(d.nonzero())
		d=d.toarray()
		dmean=np.ones(d.shape)*t2[0]
		dmean[t4[0],t4[1]]=[t2[x] for x in d[t4[0],t4[1]]]
		if varscale!=0:
			dvar=np.ones(d.shape)*t3[0]
			dvar[t4[0],t4[1]]=[t3[x] for x in d[t4[0],t4[1]]]
		else:
			dvar=np.zeros(d.shape)
	else:
		if issparse:
			d=d.toarray()
		d=d.astype(int)
		dmean=t2[d]
		dvar=t3[d] if varscale!=0 else np.zeros(d.shape)

	dvar*=varscale
	dtn=np.random.randn(nt,nc)*np.sqrt(dvar)+dmean if varscale!=0 else dmean.copy()
	assert dtn.shape==(nt,nc)

	#Normalize per cell
	if normalize:
		t1=np.log(np.exp(dtn).sum(axis=0))-np.log(1E6)
		dmean-=t1
		dtn-=t1

	t1=d.sum(axis=0)
	if (t1==0).any():
		raise ValueError('Found cell with no read at all. Please remove.')
	t1=np.log(t1)
	dcov=np.array([t1,d.shape[0]-(d!=0).sum(axis=0),t1**2])
	if dcov.ndim==3:
		dcov=dcov.reshape(dcov.shape[0],dcov.shape[2])
	assert np.isfinite(dtn).all()
	assert np.isfinite(dmean).all()
	assert np.isfinite(dvar).all() and (dvar>=0).all()
	assert dcov.shape==(3,nc) and np.isfinite(dcov).all()
	return (dtn,dmean,dvar,dcov)

def scaling_factor(dt,varname='nt0mean',v0=0,v1='max'):
	"""Computes scaling factor for every gene. Must use original (count) data
	dt:			Read counts as numpy.array(shape=(n_gene,n_cell),dtype=uint)
	varname:	Variable used to compute scaling factor for each gene.
		logtpropmean:	log(dt.mean(axis=1)/dt.mean(axis=1).sum())
		logtmeanprop:	log((dt/dt.sum(axis=0)).mean(axis=1))
		nt0mean:		(dt==0).mean(axis=1)
		lognt0mean:		log((dt==0).mean(axis=1))
		log1-nt0mean:	log(1-(dt==0).mean(axis=1))
	v0,
	v1:			Value to set scaling factor to 0 (for v0) and 1 (for v1)
		max:		max
		min:		min
		any float:	that float
	Return:	numpy.array(shape=[n_gene]) Scaling factor of weight for each gene
	"""
	if dt.ndim!=2:
		raise ValueError('dt must have 2 dimensions.')

	import numpy as np
	if varname=='logtpropmean':
		d=dt.mean(axis=1)
		d=np.log(d/d.sum())
	elif varname=='logtmeanprop':
		d=dt/dt.sum(axis=0)
		d=np.log(d.mean(axis=1))
	elif varname=='nt0mean':
		d=(dt==0).mean(axis=1)
	elif varname=='lognt0mean':
		d=np.log((dt==0).mean(axis=1))
	elif varname=='log1-nt0mean':
		d=np.log(1-(dt==0).mean(axis=1))
	else:
		raise ValueError('Unknown varname: {}'.format(varname))

	ans=[]
	for v in [v0,v1]:
		if v=='max':
			ans.append(d.max())
		elif v=='min':
			ans.append(d.min())
		else:
			ans.append(float(v))
	v0,v1=ans
	assert v1!=v0
	ans=(d-v0)/(v1-v0)

	assert ans.shape==(dt.shape[0],)
	assert np.isfinite(ans).all()
	return ans










































assert __name__ != "__main__"

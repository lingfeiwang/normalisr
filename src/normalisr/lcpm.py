#!/usr/bin/python3


def trigamma(x):
	"""`Tri-gamma function <https://en.wikipedia.org/wiki/Trigamma_function>`_ .

	Parameters
	------------
	x:	float or numpy.ndarray
		Input value(s)

	Returns
	-----------
	float or numpy.ndarray

	"""
	from scipy.special import polygamma
	return polygamma(1, x)


def lcpm(reads,
		 normalize=True,
		 nth=0,
		 ntot=None,
		 varscale=0,
		 seed=None,
		 lowmem=True,
		 nocov=False):
	"""Computes Bayesian log CPM from raw read counts.

	The technical sampling process is modelled as a Binomial distribution. The logCPM given read counts is a Bayesian inference problem and follows (shifted) Beta distribution. We use the expectation of posterior logCPM as the estimated expression levels. Resampling function is also provided to account for variances in the posterior distribution.

	**Warning**\ : Modifying keyword arguments other than nth or seed is neither recommended nor supported for function 'lcpm'. Do so at your own risk.

	Parameters
	----------
	reads:		numpy.ndarray(shape=(n_gene,n_cell),dtype='uint')
		Read count matrix.
	normalize:	bool
		Whether to normalize output to logCPM per cell. Default: True.
	nth:		int
		Number of threads to use. Defaults to 0 to use all cores automatically detected.
	ntot:		int
		Manually sets value of total number of reads in binomial distribution. Since the posterior distribution stablizes quickly as ntot increases, a large number, e.g. 1E9 is good for general use. Defaults to None to disable manual value.
	varscale:	float
		Resamples estimated expression using the posterior Beta distribution. varscale sets the scale of variance than its actual value from the posterior distribution. Defaults to 0, to compute expectation with no variance.
	seed:		int
		Initial random seed if set.
	lowmem:		bool
		Low memory mode disable mean and var in Returns and therefore saves memory.
	nocov:		bool
		Whether to skip producing covariate variables. If True, output cov=None

	Returns
	-------
	lcpm:	numpy.ndarray(shape=(n_gene,n_cell))
		Estimated expression as logCPM from read counts.
	mean:	numpy.ndarray(shape=(n_gene,n_cell)) or None
		Mean/Expectation of lcpm's every entry's posterior distribution. None if lowmem=True.
	var:	numpy.ndarray(shape=(n_gene,n_cell)) or None
		Variance of lcpm's every entry's posterior distribution. None if lowmem=True.
	cov:	numpy.ndarray(shape=(3,n_cell))
		Cellular summary covariates computed from read count matrix that may confound lcpm. Contains:

		* cov[0]:	Log total read count per cell
		* cov[1]:	Number of 0-read genes per cell
		* cov[2]:	cov[0]**2

	"""
	d = reads
	import numpy as np
	import scipy
	from scipy.special import digamma
	from .parallel import autopooler, autocount
	if nth == 0:
		nth1 = autocount()
	else:
		nth1 = nth

	if d.ndim != 2:
		raise ValueError('reads must have 2 dimensions.')
	if seed is not None:
		np.random.seed(seed)
	issparse = scipy.sparse.issparse(d)
	if ((d.data if issparse else d) < 0).any():
		raise ValueError('Negative value in d detected.')
	if varscale < 0:
		raise ValueError('varscale must be non-negative.')
	if not normalize or ntot is not None or varscale != 0:
		import logging
		logging.warning(
			"Modifying keyword arguments other than nth or seed is neither recommended nor supported for function 'lcpm'. Do so at your own risk."
		)

	nt, nc = d.shape
	t0 = d.sum() + 2 if ntot is None else ntot + 2
	assert t0 > 2
	t1 = [digamma(t0), trigamma(t0)]
	t1 = [float(x) if hasattr(x, 'shape') else x for x in t1]

	# Paralleled gammas
	dvs = np.concatenate([[0], np.unique((d.data if issparse else d).ravel())])
	t4 = dvs.max()
	t2 = autopooler(nth1,
					map(lambda x: [digamma, [x], dict()],
						np.array_split(1 + dvs, nth1)),
					dummy=True)
	t2 = dict(zip(dvs, np.concatenate(list(t2)) - t1[0]))
	t2 = np.array([t2[x] if x in t2 else 0 for x in np.arange(t4 + 1)])
	if varscale != 0:
		t3 = autopooler(
			nth1,
			map(lambda x: [trigamma, [x], dict()], np.array_split(1 + dvs, nth1)))
		t3 = dict(zip(dvs, np.concatenate(list(t3)) - t1[1]))
		t3 = np.array([t3[x] if x in t3 else 0 for x in np.arange(t4 + 1)])

	if lowmem:
		if issparse and d.size < np.prod(d.shape) / 100:
			# Sparse
			t4 = np.array(np.nonzero(d))
			d = d.toarray()
			if not np.issubdtype(d.dtype, np.integer):
				d = d.astype(int, copy=False)
			if varscale != 0:
				# First compute variance
				dtn = np.ones(d.shape) * t3[0]
				dtn[t4[0], t4[1]] = [t3[x] for x in d[t4[0], t4[1]]]
				del t3
				dtn *= varscale
				dtn = np.sqrt(dtn)
				dtn *= np.random.randn(nt, nc)
				# Then mean
				dtn += t2[0]
				dtn[t4[0], t4[1]] += [t2[x] - t2[0] for x in d[t4[0], t4[1]]]
			else:
				dtn = np.ones(d.shape) * t2[0]
				dtn[t4[0], t4[1]] = [t2[x] for x in d[t4[0], t4[1]]]
		else:
			if issparse:
				d = d.toarray()
			if not np.issubdtype(d.dtype, np.integer):
				d = d.astype(int, copy=False)
			# First compute variance
			if varscale != 0:
				dtn = t3[d] * varscale
				del t3
				dtn = np.sqrt(dtn)
				dtn *= np.random.randn(nt, nc)
				dtn += t2[d]
			else:
				dtn = t2[d]
		del t2
		assert dtn.shape == (nt, nc)
		# Normalize per cell
		if normalize:
			t1 = np.log(np.exp(dtn).sum(axis=0)) - np.log(1E6)
			dtn -= t1
	else:
		if issparse and d.size < np.prod(d.shape) / 100:
			# Sparse
			t4 = np.array(d.nonzero())
			d = d.toarray()
			if not np.issubdtype(d.dtype, np.integer):
				d = d.astype(int, copy=False)
			dmean = np.ones(d.shape) * t2[0]
			dmean[t4[0], t4[1]] = [t2[x] for x in d[t4[0], t4[1]]]
			if varscale != 0:
				dvar = np.ones(d.shape) * t3[0]
				dvar[t4[0], t4[1]] = [t3[x] for x in d[t4[0], t4[1]]]
			else:
				dvar = np.zeros(d.shape)
		else:
			if issparse:
				d = d.toarray()
			if not np.issubdtype(d.dtype, np.integer):
				d = d.astype(int, copy=False)
			dmean = t2[d]
			dvar = t3[d] if varscale != 0 else np.zeros(d.shape)

		dvar *= varscale
		dtn = np.random.randn(
			nt, nc) * np.sqrt(dvar) + dmean if varscale != 0 else dmean.copy()
		assert dtn.shape == (nt, nc)

		# Normalize per cell
		if normalize:
			t1 = np.log(np.exp(dtn).sum(axis=0)) - np.log(1E6)
			dmean -= t1
			dtn -= t1

	if nocov:
		dcov = None
	else:
		t1 = d.sum(axis=0)
		if (t1 == 0).any():
			raise ValueError('Found cell with no read at all. Please remove.')
		t1 = np.log(t1)
		dcov = np.array([t1, d.shape[0] - (d != 0).sum(axis=0), t1**2])
		if dcov.ndim == 3:
			dcov = dcov.reshape(dcov.shape[0], dcov.shape[2])
		assert dcov.shape == (3, nc) and np.isfinite(dcov).all()
	assert np.isfinite(dtn).all()
	if not lowmem:
		assert np.isfinite(dmean).all()
		assert np.isfinite(dvar).all() and (dvar >= 0).all()
	else:
		dmean = dvar = None
	return (dtn, dmean, dvar, dcov)


def scaling_factor(dt, varname='nt0mean', v0=0, v1='max'):
	"""Computes scaling factor of variance normalization for every gene.

	Lowly expressed genes need full variance normalization because of technical confounding from sequencing depth. Highly expressed genes do not need variance normalization because they are already accurately measured. The scaling factor operates as a exponential factor on the variance normalization scale for each gene. It should be maximum/minimum for genes with lowest/highest expression.

	**Warning**\ : Modifying keyword arguments is neither recommended nor supported for function 'scaling_factor'. Do so at your own risk.

	Parameters
	----------
	dt:			numpy.ndarray(shape=(n_gene,n_cell),dtype='uint')
		Read count matrix.
	varname:	str
		Variable used to compute scaling factor for each gene. Can be:

		* logtpropmean:	log(dt.mean(axis=1)/dt.mean(axis=1).sum())
		* logtmeanprop:	log((dt/dt.sum(axis=0)).mean(axis=1))
		* nt0mean:		(dt==0).mean(axis=1)
		* lognt0mean:	log((dt==0).mean(axis=1))
		* log1-nt0mean:	log(1-(dt==0).mean(axis=1))

		Defaults to nt0mean.
	v0,v1:		float
		Variable values to set scaling factor to 0 (for v0) and 1 (for v1). Linear assignment is applied for values inbetween. Can be:

		* max:			max
		* min:			min
		* any float:	that float

	Returns
	--------
	numpy.ndarray(shape=(n_gene,))
		Scaling factor of variance normalization for each gene

	"""
	if dt.ndim != 2:
		raise ValueError('dt must have 2 dimensions.')
	if v0 != 0 or v1 != 'max' or varname != 'nt0mean':
		import logging
		logging.warning(
			"Modifying keyword arguments is neither recommended nor supported for function 'scaling_factor'. Do so at your own risk."
		)

	import numpy as np
	if varname == 'logtpropmean':
		d = dt.mean(axis=1)
		d = np.log(d / d.sum())
	elif varname == 'logtmeanprop':
		d = dt / dt.sum(axis=0)
		d = np.log(d.mean(axis=1))
	elif varname == 'nt0mean':
		d = (dt == 0).mean(axis=1)
	elif varname == 'lognt0mean':
		d = np.log((dt == 0).mean(axis=1))
	elif varname == 'log1-nt0mean':
		d = np.log(1 - (dt == 0).mean(axis=1))
	else:
		raise ValueError('Unknown varname: {}'.format(varname))

	ans = []
	for v in [v0, v1]:
		if v == 'max':
			ans.append(d.max())
		elif v == 'min':
			ans.append(d.min())
		else:
			ans.append(float(v))
	v0, v1 = ans
	assert v1 != v0
	ans = (d - v0) / (v1 - v0)

	assert ans.shape == (dt.shape[0], )
	assert np.isfinite(ans).all()
	return ans


assert __name__ != "__main__"

#!/usr/bin/python3


def inv_rank(m, tol=1E-8, method='auto', logger=None, mpc=0, qr=0, **ka):
	"""Computes matrix (pseudo-)inverse and rank with SVD.

	Eigenvalues smaller than tol*largest eigenvalue are set to 0. Rank of inverted matrix is also returned. Provides to limit the number of eigenvalues to speed up computation. Broadcasts to the last 2 dimensions of the matrix.

	Parameters
	------------
	m:		numpy.ndarray(shape=(...,n,n),dtype=float)
		2-D or higher matrix to be inverted
	tol:	float
		Eigenvalues < tol*maximum eigenvalue are treated as zero.
	method:	str
		Method to compute eigenvalues:

		* auto:	Uses scipy for n<mpc or mpc==0 and sklearn otherwise
		* scipy: Uses scipy.linalg.svd
		* scipys: NOT IMPLEMENTED. Uses scipy.sparse.linalg.svds
		* sklearn: Uses sklearn.decomposition.TruncatedSVD

	logger:	object
		Logger to output warning. Defaults (None) to logging module
	mpc:	int
		Maximum rank or number of eigenvalues/eigenvectors to consider.
		Defaults to 0 to disable limit.
		For very large input matrix, use a small value (e.g. 500) to save time at the cost of accuracy.
	qr:		int
		Whether to use QR decomposition for matrix inverse.
		Only effective when method=sklearn, or =auto that defaults to sklearn.
		* 0:	No
		* 1:	Yes with default settings
		* 2+:	Yes with n_iter=qr for sklearn.utils.extmath.randomized_svd
	ka:		Keyword args passed to method

	Returns
	-------
	mi:		numpy.ndarray(shape=(...,n,n),dtype=float)
		Pseudo-inverse matrices
	r:		numpy.ndarray(shape=(...),dtype=int) or int
		Matrix ranks

	"""
	import numpy as np
	from numpy.linalg import LinAlgError
	if logger is None:
		import logging as logger
	if m.ndim <= 1 or m.shape[-1] != m.shape[-2]:
		raise ValueError('Wrong shape for m.')
	if tol <= 0:
		raise ValueError('tol must be positive.')
	if qr < 0 or int(qr) != qr:
		raise ValueError('qr must be non-negative integer.')
	if method == 'auto':
		if m.ndim > 2 and mpc > 0:
			raise NotImplementedError(
				'No current method supports >2 dimensions with mpc>0.')
		elif m.shape[-1] <= mpc or mpc == 0:
			# logger.debug('Automatically selected scipy method for matrix inverse.')
			method = 'scipy'
		else:
			# logger.debug('Automatically selected sklearn method for matrix inverse.')
			method = 'sklearn'

	n = m.shape[-1]
	if m.ndim == 2:
		if method == 'scipy':
			from scipy.linalg import svd
			try:
				s = svd(m, **ka)
			except LinAlgError:
				logger.warning(
					"Default scipy.linalg.svd failed. Falling back to option lapack_driver='gesvd'. Expecting much slower computation."
				)
				s = svd(m, lapack_driver='gesvd', **ka)
			n2 = n - np.searchsorted(s[1][::-1], tol * s[1][0])
			if mpc > 0:
				n2 = min(n2, mpc)
			ans = np.matmul(s[2][:n2].T / s[1][:n2], s[2][:n2]).T
		elif method == 'sklearn':
			from sklearn.utils.extmath import randomized_svd as svd
			n2 = min(mpc, n) if mpc > 0 else n
			# Find enough n_components by increasing in steps
			while True:
				if qr == 1:
					s = svd(m, n2, power_iteration_normalizer='QR', **ka)
				elif qr > 1:
					s = svd(m, n2, power_iteration_normalizer='QR', n_iter=qr, **ka)
				else:
					s = svd(m, n2, **ka)
				if n2 == n or s[1][-1] <= tol * s[1][0] or mpc > 0:
					break
				n2 += np.min([n2, n - n2])
			n2 = n2 - np.searchsorted(s[1][::-1], tol * s[1][0])
			if mpc > 0:
				n2 = min(n2, mpc)
			ans = np.matmul(s[2][:n2].T / s[1][:n2], s[2][:n2]).T
		else:
			raise ValueError('Unknown method {}'.format(method))
	else:
		if method == 'scipy':
			from scipy.linalg import svd
			if mpc > 0:
				raise NotImplementedError('Not supporting >2 dimensions for mpc>0.')
			warned = False

			m2 = m.reshape(np.prod(m.shape[:-2]), *m.shape[-2:])
			s = []
			for xi in m2:
				try:
					s.append(svd(xi, **ka))
				except LinAlgError:
					if not warned:
						warned = True
						logger.warning(
							"Default scipy.linalg.svd failed. Falling back to option lapack_driver='gesvd'. Expecting much slower computation."
						)
					s.append(svd(xi, lapack_driver='gesvd', **ka))
			n2 = n - np.array([np.searchsorted(x[1][::-1], tol * x[1][0]) for x in s])
			ans = [np.matmul(x[2][:y].T / x[1][:y], x[2][:y]).T for x, y in zip(s, n2)]
			ans = np.array(ans).reshape(*m.shape)
			n2 = n2.reshape(*m.shape[:-2])
		elif method == 'sklearn':
			raise NotImplementedError(
				'Not supporting >2 dimensions for method=sklearn.')
		else:
			raise ValueError('Unknown method {}'.format(method))
	try:
		if n2.ndim == 0:
			n2 = n2.item()
	except Exception:
		pass
	return ans, n2


def association_test_1(vx, vy, dx, dy, dc, dci, dcr, dimreduce=0):
	"""Fast linear association testing in single-cell non-cohort settings with covariates.

	Single threaded version to allow for parallel computing wrapper. Mainly used for naive differential expression and co-expression. Computes exact P-value and effect size (gamma) with the model for linear association testing between each vector x and vector y:
		y=gamma*x+alpha*C+epsilon,

		epsilon~i.i.d. N(0,sigma**2).

	Test statistic: conditional R**2 (or proportion of variance explained) between x and y.

	Null hypothesis: gamma=0.

	Parameters
	----------
	vx:		any
		Starting indices of dx. Only used for information passing.
	vy:		any
		Starting indices of dy. Only used for information passing.
	dx:		numpy.ndarray(shape=(n_x,n_cell)).
		Predictor matrix for a list of vector x to be tested, e.g. gene expression or grouping.
	dy:		numpy.ndarray(shape=(n_y,n_cell)).
		Target matrix for a list of vector y to be tested, e.g. gene expression.
	dc:		numpy.ndarray(shape=(n_cov,n_cell)).
		Covariate matrix as C.
	dci:	numpy.ndarray(shape=(n_cov,n_cov)).
		Low-rank inverse matrix of dc*dc.T.
	dcr:	int
		Rank of dci.
	dimreduce:	numpy.ndarray(shape=(ny,),dtype='uint') or int.
		If each vector y doesn't have full rank in the first place, this parameter is the loss of degree of freedom to allow for accurate P-value computation.

	Returns
	----------
	vx:		any
		vx from input for information passing.
	vy:		any
		vy from input for information passing.
	pv:		numpy.ndarray(shape=(n_x,n_y))
		P-values of association testing (gamma==0).
	gamma:	numpy.ndarray(shape=(n_x,n_y))
		Maximum likelihood estimator of gamma in model.
	alpha:	numpy.ndarray(shape=(n_x,n_y,n_cov))
		Maximum likelihood estimator of alpha in model.
	var_x:	numpy.ndarray(shape=(n_x,))
		Variance of dx unexplained by covariates C.
	var_y:	numpy.ndarray(shape=(n_y,))
		Variance of dy unexplained by covariates C.

	"""
	import numpy as np
	from scipy.stats import beta
	import logging
	if len(dx.shape) != 2 or len(dy.shape) != 2 or len(dc.shape) != 2:
		raise ValueError('Incorrect dx/dy/dc size.')
	n = dx.shape[1]
	if dy.shape[1] != n or dc.shape[1] != n:
		raise ValueError('Unmatching dx/dy/dc dimensions.')
	nc = dc.shape[0]
	if nc == 0:
		logging.warning('No covariate dc input.')
	elif dci.shape != (nc, nc):
		raise ValueError('Unmatching dci dimensions.')
	if dcr < 0:
		raise ValueError('Negative dcr detected.')
	elif dcr > nc:
		raise ValueError('dcr higher than covariate dimension.')
	if n <= dcr + dimreduce + 1:
		raise ValueError(
			'Insufficient number of cells: must be greater than degrees of freedom removed + covariate + 1.'
		)

	nx = dx.shape[0]
	ny = dy.shape[0]

	dx1 = dx
	dy1 = dy

	if dcr > 0:
		# Remove covariates
		ccx = np.matmul(dci, np.matmul(dc, dx1.T)).T
		ccy = np.matmul(dci, np.matmul(dc, dy1.T)).T
		dx1 = dx1 - np.matmul(ccx, dc)
		dy1 = dy1 - np.matmul(ccy, dc)
	ansvx = (dx1**2).mean(axis=1)
	ansvx[ansvx == 0] = 1
	ansvy = (dy1**2).mean(axis=1)
	ansvy[ansvy == 0] = 1
	ansc = (np.matmul(dy1, dx1.T) / (n * ansvx)).T
	ansp = ((ansc**2).T * ansvx).T / ansvy
	if dcr > 0:
		ansa = np.repeat(
			ccy.reshape(1, ccy.shape[0], ccy.shape[1]), ccx.shape[0],
			axis=0) - (ansc * np.repeat(ccx.T.reshape(ccx.shape[1], ccx.shape[0], 1),
										ccy.shape[0],
										axis=2)).transpose(1, 2, 0)
	else:
		ansa = np.zeros((nx, ny, nc), dtype=dx.dtype)

	# Compute p-values
	assert (ansp >= 0).all() and (ansp <= 1 + 1E-8).all()
	ansp = beta.cdf(1 - ansp, (n - 1 - dcr - dimreduce) / 2, 0.5)
	assert ansp.shape == (nx, ny) and ansc.shape == (nx, ny) and ansa.shape == (
		nx, ny, nc) and ansvx.shape == (nx, ) and ansvy.shape == (ny, )
	assert np.isfinite(ansp).all()
	assert np.isfinite(ansc).all() and np.isfinite(ansa).all()
	assert np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp >= 0).all() and (ansp <= 1).all() and (ansvx >= 0).all() and (
		ansvy >= 0).all()
	return [vx, vy, ansp, ansc, ansa, ansvx, ansvy]


def association_test_2(vx, vy, dx, dy, dc, sselectx, dimreduce=0):
	"""Like association_test_1, but takes a different subset of samples for each x.

	See association_test_1 for additional details.

	Parameters
	----------
	vx:		any
		Starting indices of dx. Only used for information passing.
	vy:		any
		Starting indices of dy. Only used for information passing.
	dx:		numpy.ndarray(shape=(n_x,n_cell)).
		Predictor matrix for a list of vector x to be tested, e.g. gene expression or grouping.
	dy:		numpy.ndarray(shape=(n_y,n_cell)).
		Target matrix for a list of vector y to be tested, e.g. gene expression.
	dc:		numpy.ndarray(shape=(n_cov,n_cell)).
		Covariate matrix as C.
	sselectx:	numpy.ndarray(shape=(n_x,n_cell),dtype=bool)
		Subset of samples to use for each x.
	dimreduce:	numpy.ndarray(shape=(ny,),dtype='uint') or int.
		If each vector y doesn't have full rank in the first place, this parameter is the loss of degree of freedom to allow for accurate P-value computation.

	Returns
	--------
	vx:		any
		vx from input for information passing.
	vy:		any
		vy from input for information passing.
	pv:		numpy.ndarray(shape=(n_x,n_y))
		P-values of association testing (gamma==0).
	gamma:	numpy.ndarray(shape=(n_x,n_y))
		Maximum likelihood estimator of gamma in model.
	alpha:	numpy.ndarray(shape=(n_x,n_y,n_cov))
		Maximum likelihood estimator of alpha in model.
	var_x:	numpy.ndarray(shape=(n_x,))
		Variance of dx unexplained by covariates C.
	var_y:	numpy.ndarray(shape=(n_x,n_y))
		Variance of dy unexplained by covariates C.

	"""
	import numpy as np
	import logging
	from scipy.stats import beta
	if len(dx.shape) != 2 or len(dy.shape) != 2 or len(dc.shape) != 2:
		raise ValueError('Incorrect dx/dy/dc size.')
	n = dx.shape[1]
	if dy.shape[1] != n or dc.shape[1] != n:
		raise ValueError('Unmatching dx/dy/dc dimensions.')
	nc = dc.shape[0]
	if nc == 0:
		logging.warning('No covariate dc input.')
	if sselectx.shape != dx.shape:
		raise ValueError('Unmatching sselectx dimensions.')

	nx = dx.shape[0]
	ny = dy.shape[0]
	ansp = np.zeros((nx, ny), dtype=float)
	ansvx = np.zeros((nx, ), dtype=float)
	ansvy = np.zeros((nx, ny), dtype=float)
	ansc = np.zeros((nx, ny), dtype=float)
	ansa = np.zeros((nx, ny, nc), dtype=float)
	ansn = np.zeros((nx, ny), dtype=int)

	for xi in range(nx):
		# Select samples
		t1 = np.nonzero(sselectx[xi])[0]
		ns = len(t1)
		if len(np.unique(dx[xi, t1])) < 2:
			continue
		dx1 = dx[xi, t1]
		dy1 = dy[:, t1]
		if nc > 0:
			# Remove covariates
			dc1 = dc[:, t1]
			t1 = np.matmul(dc1, dc1.T)
			t1i, r = inv_rank(t1)
		else:
			r = 0
		ansn[xi] = r
		if r > 0:
			# Remove covariates
			ccx = np.matmul(t1i, np.matmul(dc1, dx1.T)).T
			ccy = np.matmul(t1i, np.matmul(dc1, dy1.T)).T
			dx1 = dx1 - np.matmul(ccx, dc1)
			dy1 = dy1 - np.matmul(ccy, dc1)
		t1 = (dx1**2).mean()
		if t1 == 0:
			# X completely explained by covariate. Should never happen in theory.
			t1 = 1
		ansvx[xi] = t1
		ansvy[xi] = (dy1**2).mean(axis=1)
		ansc[xi] = np.matmul(dx1, dy1.T).ravel() / (ns * t1)
		if r > 0:
			ansa[xi] = ccy - np.repeat(ansc[xi].reshape(ny, 1), nc,
									   axis=1) * ccx.ravel()
		ansp[xi] = (ansc[xi]**2) * t1 / ansvy[xi]

	# Compute p-values
	assert (ansp >= 0).all() and (ansp <= 1 + 1E-8).all()
	t1 = (sselectx.sum(axis=1) - 1 - ansn.T - dimreduce).T
	if (t1 <= 0).any():
		raise RuntimeError(
			'Insufficient number of cells: must be greater than degrees of freedom removed + covariate + 1.'
		)
	ansp = beta.cdf(1 - ansp, t1 / 2, 0.5)
	assert ansp.shape == (nx, ny) and ansc.shape == (nx, ny) and ansa.shape == (
		nx, ny, nc) and ansvx.shape == (nx, ) and ansvy.shape == (nx, ny)
	assert np.isfinite(ansp).all() and np.isfinite(ansc).all() and np.isfinite(
		ansa).all() and np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp >= 0).all() and (ansp <= 1).all() and (ansvx >= 0).all() and (
		ansvy >= 0).all()
	return [vx, vy, ansp, ansc, ansa, ansvx, ansvy]


def prod1(vx, vy, dx, dy):
	"""Pickleable function for matrix product that keeps information

	Parameters
	-------------
	vx:	any
		Information passed
	vy:	any
		Information passed
	dx: numpy.ndarray(shape=(...,n))
		Matrix for multiplication
	dy: numpy.ndarray(shape=(...,n))
		Matrix for multiplication

	Returns
	---------
	vx: any
		vx
	vy:	any
		vy
	product: numpy.ndarray(shape=(...))
		dx\ @\ dy.T

	"""
	import numpy as np
	return (vx, vy, np.matmul(dx, dy.T))


def association_test_4(vx, vy, prod, prody, prodyy, na, dimreduce=0, **ka):
	"""Like association_test_1, but regards all other x's as covariates when testing each x.

	See association_test_1 for additional details. Other x's are treated as covariates but their coefficients (alpha) would not be returned to reduce memory footprint.

	Parameters
	----------
	vx:		any
		Starting indices of dx.
	vy:		any
		Starting indices of dy. Only used for information passing.
	prod:	numpy.ndarray(shape=(n_x+n_cov,n_x+n_cov))
		A\ @\ A.T, where A=numpy.block([dx,dc]).
	prody:	numpy.ndarray(shape=(n_x+n_cov,n_y))
		A\ @\ dy.T, where A=numpy.block([dx,dc])
	prodyy:	numpy.ndarray(shape=(n_y,))
		(dy**2).sum(axis=1)
	na:		tuple
		(n_x,n_y,n_cov,n_cell,lenx). Numbers of (x's, y's, covariates, cells, x's to compute association for)
	dimreduce:	numpy.ndarray(shape=(ny,),dtype='uint') or int.
		If each vector y doesn't have full rank in the first place, this parameter is the loss of degree of freedom to allow for accurate P-value computation.
	ka:		dict
		Keyword arguments passed to inv_rank.

	Returns
	--------
	vx:		any
		vx from input for information passing.
	vy:		any
		vy from input for information passing.
	pv:		numpy.ndarray(shape=(n_x,n_y))
		P-values of association testing (gamma==0).
	gamma:	numpy.ndarray(shape=(n_x,n_y))
		Maximum likelihood estimator of gamma in model.
	alpha:	numpy.ndarray(shape=(n_x,n_y,n_cov))
		Maximum likelihood estimator of alpha in model.
	var_x:	numpy.ndarray(shape=(lenx,))
		Variance of dx unexplained by covariates C.
	var_y:	numpy.ndarray(shape=(lenx,n_y))
		Variance of dy unexplained by covariates C.

	"""
	import numpy as np
	from scipy.stats import beta
	import logging
	if len(na) != 5:
		raise ValueError('Wrong format for na')
	nx, ny, nc, n, lenx = na
	if nx == 0 or ny == 0 or n == 0:
		raise ValueError('Dimensions in na==0 detected.')
	if nc == 0:
		logging.warning('No covariate dc input.')
	if lenx <= 0:
		raise ValueError('lenx must be positive.')
	if vx < 0 or vx + lenx > nx:
		raise ValueError('Wrong values of vx and/or lenx, negative or beyond nx.')
	if prod.shape != (nx + nc, nx + nc):
		raise ValueError('Unmatching shape for prod. Expected: {}. Got: {}.'.format(
			(nx + nc, nx + nc), prod.shape))
	if prody.shape != (nx + nc, ny):
		raise ValueError('Unmatching shape for prody. Expected: {}. Got: {}.'.format(
			(nx + nc, ny), prody.shape))
	if prodyy.shape != (ny, ):
		raise ValueError(
			'Unmatching shape for prodyy. Expected: {}. Got: {}.'.format(
				(ny, ), prodyy.shape))
	ansp = np.zeros((lenx, ny), dtype=float)
	ansvx = np.zeros((lenx, ), dtype=float)
	ansvy = np.zeros((lenx, ny), dtype=float)
	ansc = np.zeros((lenx, ny), dtype=float)
	ansa = np.zeros((lenx, ny, nc), dtype=float)
	ansn = np.zeros((lenx, ny), dtype=int)

	for xi in range(lenx):
		t0 = list(filter(lambda x: x != vx + xi, range(nx + nc)))
		if len(t0) > 0:
			t1 = prod[np.ix_(t0, t0)]
			t1i, r = inv_rank(t1, **ka)
		else:
			r = 0
		ansn[xi] = r
		if r == 0:
			# No covariate
			dxx = prod[vx + xi, vx + xi] / n
			dyy = prodyy / n
			dxy = prody[vx + xi] / n
		else:
			ccx = np.matmul(prod[[vx + xi], t0], t1i)
			dxx = (prod[vx + xi, vx + xi]
				   - float(np.matmul(ccx, prod[t0, [vx + xi]]))) / n
			ccy = np.matmul(prody[t0].T, t1i)
			dyy = (prodyy - (ccy.T * prody[t0]).sum(axis=0)) / n
			dxy = (prody[vx + xi] - np.matmul(ccy, prod[t0, [vx + xi]]).ravel()) / n
		if dxx == 0:
			# X completely explained by covariate. Should never happen in theory.
			dxx = 1
		ansvx[xi] = dxx
		ansvy[xi] = dyy
		ansc[xi] = dxy / dxx
		if r > 0:
			ansa[xi] = ccy[:, -nc:] - np.repeat(ansc[xi].reshape(ny, 1), nc,
												axis=1) * ccx[-nc:]
		ansp[xi] = (dxy**2) / (dxx * dyy)

	# Compute p-values
	assert (ansp >= 0).all() and (ansp <= 1 + 1E-8).all()
	t1 = n - 1 - ansn - dimreduce
	if (t1 <= 0).any():
		raise RuntimeError(
			'Insufficient number of cells: must be greater than degrees of freedom removed + covariate + 1.'
		)
	ansp = beta.cdf(1 - ansp, t1 / 2, 0.5)
	assert ansp.shape == (lenx, ny) and ansc.shape == (
		lenx, ny) and ansa.shape == (lenx, ny, nc) and ansvx.shape == (
			lenx, ) and ansvy.shape == (lenx, ny)
	assert np.isfinite(ansp).all() and np.isfinite(ansc).all() and np.isfinite(
		ansa).all() and np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp >= 0).all() and (ansp <= 1).all() and (ansvx >= 0).all() and (
		ansvy >= 0).all()
	return [vx, vy, ansp, ansc, ansa, ansvx, ansvy]


def association_tests_1(dx, dy, dc, bsx=0, bsy=0, nth=1, lowmem=True, **ka):
	"""Performs association tests between all pairs of two (or one) variables. Performs parallel computation with multiple processes on the same machine.

	Model for differential expression between X and Y:
		Y=gamma*X+alpha*C+epsilon,

		epsilon~i.i.d. N(0,sigma**2).

	Test statistic: conditional R**2 (or proportion of variance explained) between Y and X.

	Null hypothesis: gamma=0.

	Parameters
	-----------
	dx:		numpy.ndarray(shape=(n_x,n_cell)).
		Normalized matrix X.
	dy:		numpy.ndarray(shape=(n_y,n_cell),dtype=float) or None
		Normalized matrix Y. If None, indicates dy=dx, i.e. self-association between pairs within X.
	dc:		numpy.ndarray(shape=(n_cov,n_cell),dtype=float)
		Normalized covariate matrix C.
	bsx:	int
		Batch size, i.e. number of Xs in each computing batch. Defaults to 500.
	bsy:	int
		Batch size, i.e. number of Xs in each computing batch. Defaults to 500.
	nth:	int
		Number of parallel processes. Set to 0 for using automatically detected CPU counts.
	ka:		dict
		Keyword arguments passed to normalisr.association.association_test_1. See below.

	Returns
	--------
	P-values:	numpy.ndarray(shape=(n_x,n_y))
		Differential expression P-value matrix.
	dot:		numpy.ndarray(shape=(n_x,n_y))
		Inner product between X and Y pairs, after removing covariates.
	alpha:		numpy.ndarray(shape=(n_x,n_y,n_cov)) or None.
		Maximum likelihood estimators of alpha, separatly tested for each grouping. None if lowmem=True.
	varx:		numpy.ndarray(shape=(n_x))
		Variance of grouping after removing covariates.
	vary:		numpy.ndarray(shape=(n_y))
		Variance of gene expression after removing covariates.
		It can depend on the grouping being tested depending on parameter single.

	Keyword arguments
	--------------------------------
	dimreduce:	numpy.ndarray(shape=(n_y,),dtype=int) or int
		If dy doesn't have full rank, such as due to prior covariate removal (although the recommended method is to leave covariates in dc), this parameter allows to specify the loss of ranks/degrees of freedom to allow for accurate P-value computation. Default is 0, indicating no rank loss.
	"""
	import numpy as np
	import logging
	import itertools
	from .parallel import autopooler
	ka0 = dict(ka)
	if dy is None:
		dy = dx
		samexy = True
	else:
		samexy = False
	# Ignore single-valued genotypes
	nx, ns = dx.shape
	ny = dy.shape[0]
	nc = dc.shape[0]
	if bsx == 0:
		# Transfer 1GB data max
		bsx = int((2**30 - dc.dtype.itemsize * nc * ns) //
				  (2 * dx.dtype.itemsize * ns))
		bsx = min(bsx, 500)
		if samexy:
			logging.info('Using automatic batch size: {}'.format(bsx))
		else:
			logging.info('Using automatic batch size for X: {}'.format(bsx))
	if bsy == 0:
		if samexy:
			bsy = bsx
		else:
			bsy = int((2**30 - dc.dtype.itemsize * nc * ns) //
					  (2 * dy.dtype.itemsize * ns))
			bsy = min(bsy, 500)
			logging.info('Using automatic batch size for Y: {}'.format(bsy))

	it = itertools.product(
		map(lambda x: [x, min(x + bsx, nx)], range(0, nx, bsx)),
		map(lambda x: [x, min(x + bsy, ny)], range(0, ny, bsy)))
	if samexy:
		it = filter(lambda x: x[0][0] <= x[1][0], it)
	if nc > 0 and (dc != 0).any():
		dci, dcr = inv_rank(np.matmul(dc, dc.T))
	else:
		dci = None
		dcr = 0
	it = map(
		lambda x: [
			association_test_1,
			(x[0][0], x[1][0], dx[x[0][0]:x[0][1]], dy[x[1][0]:x[1][1]], dc, dci,
			 dcr), ka0], it)

	ans0 = autopooler(nth, it, dummy=True)
	assert len(ans0) > 0
	ans = np.ones((nx, ny), dtype=dy.dtype)
	ansdot = np.zeros((nx, ny), dtype=dy.dtype)
	ansa = None if lowmem else np.zeros((nx, ny, nc), dtype=dy.dtype)
	ansvx = np.zeros((nx), dtype=dy.dtype)
	ansvy = ansvx if samexy else np.zeros((ny), dtype=dy.dtype)
	for xi in ans0:
		ans[xi[0]:xi[0] + xi[2].shape[0], xi[1]:xi[1] + xi[2].shape[1]] = xi[2]
		ansdot[xi[0]:xi[0] + xi[3].shape[0], xi[1]:xi[1] + xi[3].shape[1]] = xi[3]
		if not lowmem:
			ansa[xi[0]:xi[0] + xi[4].shape[0], xi[1]:xi[1] + xi[4].shape[1]] = xi[4]
		ansvx[xi[0]:xi[0] + xi[5].shape[0]] = xi[5]
		ansvy[xi[1]:xi[1] + xi[6].shape[0]] = xi[6]

	# Convert from coef to dot
	ansdot = (ansdot.T * ansvx).T
	if samexy:
		ans = np.triu(ans, 1)
		ans = ans + ans.T
		ansdot = np.triu(ansdot, 1)
		ansdot = ansdot + ansdot.T
		if not lowmem:
			ansa = np.triu(ansa.transpose(2, 0, 1))
			ansa = (ansa + ansa.transpose(0, 2, 1)).transpose(1, 2, 0)

	assert ans.shape == (nx, ny) and ansdot.shape == (nx, ny) and ansvx.shape == (
		nx, ) and ansvy.shape == (ny, )
	assert np.isfinite(ans).all() and np.isfinite(ansdot).all() and np.isfinite(
		ansvx).all() and np.isfinite(ansvy).all()
	assert (ans >= 0).all() and (ans <= 1).all() and (ansvx >= 0).all() and (
		ansvy >= 0).all()
	if not lowmem:
		assert ansa.shape == (nx, ny, nc) and np.isfinite(ansa).all()
	else:
		ansa = None
	return (ans, ansdot, ansa, ansvx, ansvy)


assert __name__ != "__main__"

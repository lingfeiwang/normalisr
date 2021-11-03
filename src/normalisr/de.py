#!/usr/bin/python3


def de(dg, dt, dc, bs=0, **ka):
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
		For single=4, defaults to splitting grouping by 10 and not on genes.
	ka:		dict
		Keyword arguments passed to normalisr.association.association_test_*. See below.

	Returns
	--------
	P-values:	numpy.ndarray(shape=(n_group,n_gene))
		Differential expression P-value matrix.
	gamma:		numpy.ndarray(shape=(n_group,n_gene))
		Differential expression log fold change matrix.
	alpha:		numpy.ndarray(shape=(n_group,n_gene,n_cov)) or None
		Maximum likelihood estimators of alpha, separatly tested for each grouping if not lowmem else None.
	varg:		numpy.ndarray(shape=(n_group))
		Variance of grouping after removing covariates.
	vart:		numpy.ndarray(shape=(n_group,n_gene))
		Variance of gene expression after removing covariates.
		It can depend on the grouping being tested depending on parameter single.

	Keyword arguments
	--------------------------------
	single:	int
		Option to deal with other groupings when testing one groupings v.s. gene expression.

		* 0:    Ignores other groupings (default).
		* 1:	Excludes all cells belonging to any other grouping (value==1), assuming dg=0,1 only. This is suitable for low-MOI single-cell CRISPR screens.
		* 4:	Treats other groupings as covariates for mean expression. This is suitable for high-MOI single-cell CRISPR screens.

	lowmem:	bool
		Whether to replace alpha in return value with None to save memory.
	nth:	int
		Number of parallel processes. Set to 0 for using automatically detected CPU counts.
	dimreduce:	numpy.ndarray(shape=(n_gene,),dtype=int) or int
		If dt doesn't have full rank, such as due to prior covariate removal (although the recommended method is to leave covariates in dc), this parameter allows to specify the loss of ranks/degrees of freedom to allow for accurate P-value computation. Default is 0, indicating no rank loss.
	method:		str, only for single=4
		Method to compute eigenvalues in SVD-based matrix inverse (for removal of covariates):

		* auto:	Uses scipy for n_matrix<mpc or mpc==0 and sklearn otherwise. Default.
		* scipy: Uses scipy.linalg.svd.
		* scipys: NOT IMPLEMENTED. Uses scipy.sparse.linalg.svds.
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
	import itertools
	from .association import association_tests
	from .parallel import autopooler
	ka0 = dict(ka)

	# Ignore single-valued genotypes
	dg0 = dg
	gid = [len(np.unique(x)) > 1 for x in dg]
	dg = dg0[gid]
	ng, ns = dg.shape
	nt = dt.shape[0]
	nc = dc.shape[0]

	ans, ansc, ansa, ansvg, ansvt = association_tests(dg,
													  dt,
													  dc,
													  bsx=bs,
													  bsy=bs,
													  return_dot=False,
													  **ka)
	# Reformat shape for single-valued genotypes
	t1 = np.ones((dg0.shape[0], nt), dtype=dt.dtype)
	t1[gid] = ans
	ans = t1
	t1 = np.zeros((dg0.shape[0], nt), dtype=dt.dtype)
	t1[gid] = ansc
	ansc = t1
	if ansa is not None:
		t1 = np.zeros((dg0.shape[0], nt, nc), dtype=dt.dtype)
		t1[gid] = ansa
		ansa = t1
	t1 = np.zeros((dg0.shape[0], ), dtype=dt.dtype)
	t1[gid] = ansvg
	ansvg = t1
	t1 = np.zeros((dg0.shape[0], nt), dtype=dt.dtype)
	t1[gid] = ansvt
	ansvt = t1
	assert ans.shape == (dg0.shape[0], nt) and ansc.shape == (
		dg0.shape[0], nt) and ansvg.shape == (dg0.shape[0], ) and ansvt.shape == (
			dg0.shape[0], nt)
	assert np.isfinite(ans).all() and np.isfinite(ansc).all() and np.isfinite(
		ansvg).all() and np.isfinite(ansvt).all()
	assert (ans >= 0).all() and (ans <= 1).all() and (ansvg >= 0).all() and (
		ansvt >= 0).all()
	if ansa is not None:
		assert ansa.shape == (dg0.shape[0], nt, nc) and np.isfinite(ansa).all()
	return (ans, ansc, ansa, ansvg, ansvt)


assert __name__ != "__main__"

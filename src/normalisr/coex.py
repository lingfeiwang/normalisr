#!/usr/bin/python3


def coex(dt, dc, **ka):
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
	ka:		dict
		Keyword arguments passed to normalisr.association.association_tests. See below.

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
	bs:		int
		Batch size, i.e. number of genes in each computing batch. Use 0 for default: Data transfer limited to 1GB, capped at bs=500.
	nth:	int
		Number of parallel processes. Set to 0 for using automatically detected CPU counts.
	dimreduce:	numpy.ndarray(shape=(n_gene,),dtype=int) or int
		If dt doesn't have full rank, such as due to prior covariate removal (although the recommended method is to leave covariates in dc), this parameter allows to specify the loss of ranks/degrees of freedom to allow for accurate P-value computation. Default is 0, indicating no rank loss.

	"""
	from .association import association_tests
	ans = association_tests(dt, None, dc, **ka)
	return (ans[0], ans[1], ans[4])


assert __name__ != "__main__"

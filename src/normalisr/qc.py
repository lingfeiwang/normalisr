#!/usr/bin/python3

def qc_reads(reads,n_gene,nc_gene,ncp_gene,n_cell,nt_cell,ntp_cell):
	"""QC by bounding UMIs. All QC parameters can be set to 0 to disable.
	reads:			Dense reads matrix (n_gene,n_cell)
	Gene removal:
		n_gene:		Lower bound on total read counts.
		nc_gene:	Lower bound on number of expressed cells.
		ncp_gene:	Lower bound on proportion of expressed cells.
	Cell removal:
		n_cell:		Lower bound on total read counts.
		nt_cell:	Lower bound on number of expressed genes.
		ntp_cell:	Lower bound on proportion of expressed genes.
	Return: (genes_select,cells_select). Each as a index array to indicate genes/cells passed QC.
	"""
	import numpy as np
	import logging
	if reads.ndim!=2:
		raise ValueError('reads must have 2 dimensions.')
	if not np.all([x>=0 for x in [n_gene,nc_gene,ncp_gene,n_cell,nt_cell,ntp_cell]]):
		raise ValueError('All parameters must be non-negative.')
	if not np.all([x<=1 for x in [ncp_gene,ntp_cell]]):
		raise ValueError('Proportional parameters must be no greater than 1.')

	dt=reads
	nt,ns=dt.shape
	nt0=ns0=0
	st=np.arange(nt)
	ss=np.arange(ns)
	while nt0!=nt or ns0!=ns:
		nt0=nt
		ns0=ns
		st1=np.ones(len(st),dtype=bool)
		ss1=np.ones(len(ss),dtype=bool)
		#Filter genes
		if n_gene>0:
			st1&=dt.sum(axis=1)>=n_gene
		if nc_gene>0 or ncp_gene>0:
			t1=(dt>0).sum(axis=1)
			if nc_gene>0:
				st1&=t1>=nc_gene
			if ncp_gene>0:
				st1&=t1>=ncp_gene*ns
		#Filter cells
		if n_cell>0:
			ss1&=dt.sum(axis=0)>=n_cell
		if nt_cell>0 or ntp_cell>0:
			t1=(dt>0).sum(axis=0)
			if nt_cell>0:
				ss1&=t1>=nt_cell
			if ntp_cell>0:
				ss1&=t1>=ntp_cell*nt
		#Removals
		st=st[st1]
		ss=ss[ss1]
		dt=dt[st1][:,ss1]
		nt=len(st)
		ns=len(ss)
		if nt==0:
			raise RuntimeError('All genes removed in QC.')
		if ns==0:
			raise RuntimeError('All cells removed in QC.')
	logging.info('Removed {}/{} genes and {}/{} cells in QC.'.format(reads.shape[0]-len(st),reads.shape[0],reads.shape[1]-len(ss),reads.shape[1]))
	return (st,ss)

def qc_outlier(dw,pcut=1E-10,outrate=0.02):
	"""QC by removing cell outliers by variance. Fit normal distribution and detect outliers iteratively.
	dw:		Fitted cell weights (i.e. variance**-0.5) as numpy.array(shape=[n_cell])
	pcut:	Bonferroni P-value cutoff for outliers in a normally distribution of fitted cell weight.
	outrate:Upper bound of proportion of outliers on either side of variance distribution.
			Used for initial outlier assignment and validity check.
	Return:	Whether each cell passed QC as numpy.array(shape=[n_cell],dtype=bool)
	"""
	import numpy as np
	from sklearn.linear_model import LinearRegression as lr0
	from scipy.stats import norm
	import logging
	if pcut<=0 or pcut>=1:
		raise ValueError('Parameter pcut should be between 0 and 1 (exclusive).')
	if outrate<=0 or outrate>=0.5:
		raise ValueError('Parameter outrate and outrate should be between 0 and 0.5 (exclusive).')
	if dw.min()<=0:
		raise ValueError('Non-positive cell weight found..')
	lr=lr0(fit_intercept=False)
	ns=len(dw)

	outlier_cut=pcut/ns
	samples=np.ones(ns,dtype=bool)
	start=True
	samples_all=np.array([],dtype=bool).reshape(0,ns)

	#Iterative outlier search
	n=0
	#Fitted value must be present at least 3 and 10% times
	while len(samples_all)==0 or t1.mean()<0.1 or t1.sum()<3:
		n+=1
		samples_all=np.concatenate([samples_all,[samples]],axis=0)
		if start:
			#Initialize outliers by partitioning
			t2=[int(np.ceil(outrate*ns)),int(np.floor((1-outrate)*ns))]
			t3=np.partition(dw,t2)
			samples=(dw>=t3[t2[0]])&(dw<=t3[t2[1]])
			start=False
		#Fit normal distribution
		dmean=dw[samples].mean()
		dvar=np.sqrt(((dw[samples]-dmean)**2).mean())
		t1=(dw-dmean)/dvar
		t2=t1
		#Compute P-value
		t1=np.min([norm.sf(t1),norm.cdf(t1)],axis=0)*2
		samples=t1>=outlier_cut
		logging.debug('Step {}, outlier count/rate: {}/{}'.format(n,(~samples).sum(),(~samples).mean()))
		t1=((samples^samples_all).sum(axis=1)==0)
	if (~samples).mean()>2*outrate:
		raise RuntimeError('Fitted outlier rate {}>{}.'.format((~samples).mean(),2*outrate))
	logging.info('Removed {}/{} cells due to variance outlier'.format((~samples).sum(),samples.size))
	return samples










































assert __name__ != "__main__"

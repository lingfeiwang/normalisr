#!/usr/bin/python3

def nodiag(d,split=False):
	"""Removes diagonals from 2D matrix.

	d:	numpy.array(shape=[n1,n2]) Input matrix
	split:	Whether to output list of 1D arrays for each row.
	Return:	If split:	[numpy.array(shape=[n2 or n2-1])] List of 1D arrays for each row.
			Else:		numpy.array(shape=[n1*n2-min(n1,n2)]) Reduced 1D array"""
	import numpy as np
	assert d.ndim==2
	ans=[]
	for xi in range(np.min(d.shape)):
		ans.append(d[xi,:xi] if xi>0 else np.array([],dtype=d.dtype))
		if xi<d.shape[1]-1:
			ans[-1]=np.concatenate([ans[-1],d[xi,xi+1:]])
	if d.shape[0]>d.shape[1]:
		ans+=list(d[d.shape[1]:])
	if not split:
		ans=np.concatenate(ans)
	return ans

def rediag(d,fill=0,shape=None):
	"""Convert a 'nodiag'ed vector back to matrix of original shape.

	d: numpy.array(shape=[n]) of output array from nodiag(.,split=False)
	fill: Values to fill in the diagonal entries
	shape: Specify the shape of original matrix. If omitted, assume original is a square matrix.
	Return: numpy.array(shape=shape) of original matrix shape."""
	import numpy as np
	if shape is None:
		t1=int(np.sqrt(d.size))+1
		assert t1*(t1-1)==d.size
		shape=(t1,t1)
	else:
		assert len(shape)==2
		assert shape[0]*shape[1]-np.min(shape)==d.size
	m=np.ones(shape,dtype=d.dtype)*fill
	xj=0
	for xi in range(np.min(m.shape)):
		t1=xi
		if t1>0:
			m[xi,:xi]=d[xj:xj+t1]
		xj+=t1
		t1=m.shape[1]-xi-1
		if t1>0:
			m[xi,(xi+1):]=d[xj:xj+t1]
		xj+=t1
	if m.shape[0]>m.shape[1]:
		m[m.shape[1]:]=d[xj:].reshape(m.shape[0]-m.shape[1],m.shape[1])
	return m

def bh(pv,weight=None):
	"""Convert P-values to Q-values using Benjamini–Hochberg procedure

	pv:		1D numpy.array of P-values
	weight:	Weight of each P-value. Defaults to equal.
	Return:	1D numpy.array of Q-values
	Ref:	Controlling the false discovery rate: a practical and powerful approach to multiple testing, 1995
	"""
	import numpy as np
	import logging
	assert pv.ndim==1 and pv.size>0
	assert np.isfinite(pv).all() and pv.min()>=0 and pv.max()<=1

	n0=pv.size
	if weight is None:
		weight=np.ones(n0)
	else:
		assert weight.shape==pv.shape
		assert np.isfinite(weight).all() and weight.min()>=0 and weight.max()>0

	#Shrink data
	pv2,ids=np.unique(pv,return_inverse=True)
	n=pv2.size
	if n==1:
		logging.warn('Identical p-value in all entries.')
	w=np.zeros(n,dtype=pv.dtype)
	for xi in range(n0):
		w[ids[xi]]+=weight[xi]

	#BH method
	w=np.cumsum(w)
	w/=w[-1]
	pv2/=w
	pv2[~np.isfinite(pv2)]=1
	pv2=np.min([pv2,np.repeat(1,n)],axis=0)
	pv2=np.max([pv2,np.repeat(0,n)],axis=0)
	for xi in range(n-2,-1,-1):
		pv2[xi]=min(pv2[xi],pv2[xi+1])

	#Recover index
	ans=pv2[ids].astype(pv.dtype)
	assert ans.shape==pv.shape
	return ans

def binnet(net,qcut):
	"""Binarizes P-value co-expresion network to thresholded Q-value network.

	Q-values are computed separately per row to account for differences in the number of genes co-expressed, especially by master regulators, using Benjamini–Hochberg procedure. Co-expression Q-value matrix is thresholded for return.

	Parameters
	----------
	net:	numpy.ndarray(shape=(n_gene,n_gene),dtype=float)
		Symmetric co-expression P-value matrix.
	qcut:	float
		Cutoff for Q-value network.

	Returns
	-------
	numpy.ndarray(shape=(n_gene,n_gene),dtype=bool)
		Binarized, assymmetric co-expression matrix.

	"""
	import numpy as np
	assert net.ndim==2
	assert np.isfinite(net).all()
	assert net.min()>=0 and net.max()<=1
	nt=net.shape[0]
	if net.shape[1]!=nt or nt<=1:
		raise ValueError('Wrong shape of net or namet.')
	if qcut<=0 or qcut>=1:
		raise ValueError('Q-value cutoff must be between 0 and 1.')

	#To Q-value network
	shape=net.shape
	net=np.concatenate([bh(x) for x in nodiag(net,split=True)])
	net=rediag(net,shape=shape,fill=1)
	assert np.isfinite(net).all()
	assert net.min()>=0
	assert net.max()<=1
	#To binary network
	net=net<=qcut
	if net.sum()==0:
		raise RuntimeError("Empty binary network.")
	return net































assert __name__ != "__main__"

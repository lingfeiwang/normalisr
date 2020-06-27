#!/usr/bin/python3

def inv_rank(m,tol=1E-8,method='auto',logger=None,mpc=0,qr=0,**ka):
	"""Computes matrix (pseudo-)inverse and rank with SVD.

	Broadcasts to the last 2 dimensions of the matrix.

	Parameters
	----------
	m : np.array([...,n,n])
		Matrix to be inverted
	tol : float
		Eigenvalues < tol*maximum eigenvalue are treated as zero.
	method : str
		Method to compute eigenvalues:
		* auto:	Uses scipy for n<mpc or mpc==0 and sklearn otherwise
		* scipy: Uses scipy.linalg.svd
		* scipys: NOT IMPLEMENTED.Uses scipy.sparse.linalg.svds
		* sklearn: Uses sklearn.decomposition.TruncatedSVD
	logger : object
		Logger to output warning. Defaults (None) to logging module
	mpc : int
		Maximum rank or number of principal components to consider.
		Defaults to 0 to disable limit.
		For very large input matrix, use a small value (e.g. 100) to save time at the cost of accuracy.
	qr : int
		Whether to use QR decomposition for matrix inverse.
		Only effective when method=sklearn, or =auto and defaults to sklearn.
		* 0:	No
		* 1:	Yes with default settings
		* 2+:	Yes with n_iter=qr for sklearn.utils.extmath.randomized_svd
	ka:	Keyword args passed to method

	Returns
	-------
	mi:	np.array([...,n,n]) as inverse matrices
	r:	np.array([...]) or int as ranks

	"""
	import numpy as np
	from numpy.linalg import LinAlgError
	if logger is None:
		import logging as logger
	if m.ndim<=1 or m.shape[-1]!=m.shape[-2]:
		raise ValueError('Wrong shape for m.')
	if tol<=0:
		raise ValueError('tol must be positive.')
	if qr<0 or int(qr)!=qr:
		raise ValueError('qr must be non-negative integer.')
	if method=='auto':
		if m.ndim>2 and mpc>0:
			raise NotImplementedError('No current method supports >2 dimensions with mpc>0.')
		elif m.shape[-1]<=mpc or mpc==0:
			# logger.debug('Automatically selected scipy method for matrix inverse.')
			method='scipy'
		else:
			# logger.debug('Automatically selected sklearn method for matrix inverse.')
			method='sklearn'

	n=m.shape[-1]
	if m.ndim==2:
		if method=='scipy':
			from scipy.linalg import svd
			try:
				s=svd(m,**ka)
			except LinAlgError as e:
				logger.warn("Default scipy.linalg.svd failed. Falling back to option lapack_driver='gesvd'. Expecting much slower computation.")
				s=svd(m,lapack_driver='gesvd',**ka)
			n2=n-np.searchsorted(s[1][::-1],tol*s[1][0])
			if mpc>0:
				n2=min(n2,mpc)
			ans=np.matmul(s[2][:n2].T/s[1][:n2],s[2][:n2]).T
		elif method=='sklearn':
			from sklearn.utils.extmath import randomized_svd as svd
			n2=min(mpc,n) if mpc>0 else n
			#Find enough n_components by increasing in steps
			while True:
				if qr==1:
					s=svd(m,n2,power_iteration_normalizer='QR',**ka)
				elif qr>1:
					s=svd(m,n2,power_iteration_normalizer='QR',n_iter=qr,**ka)
				else:
					s=svd(m,n2,**ka)
				if n2==n or s[1][-1]<=tol*s[1][0] or mpc>0:
					break
				n2+=np.min([n2,n-n2])
			n2=n2-np.searchsorted(s[1][::-1],tol*s[1][0])
			if mpc>0:
				n2=min(n2,mpc)
			ans=np.matmul(s[2][:n2].T/s[1][:n2],s[2][:n2]).T
		else:
			raise ValueError('Unknown method {}'.format(method))
	else:
		if method=='scipy':
			from scipy.linalg import svd
			if mpc>0:
				raise NotImplementedError('Not supporting >2 dimensions for mpc>0.')
			warned=False

			m2=m.reshape(np.prod(m.shape[:-2]),*m.shape[-2:])
			s=[]
			for xi in m2:
				try:
					s.append(svd(xi,**ka))
				except LinAlgError as e:
					if not warned:
						warned=True
						logger.warn("Default scipy.linalg.svd failed. Falling back to option lapack_driver='gesvd'. Expecting much slower computation.")
					s.append(svd(xi,lapack_driver='gesvd',**ka))
			n2=n-np.array([np.searchsorted(x[1][::-1],tol*x[1][0]) for x in s])
			ans=[np.matmul(x[2][:y].T/x[1][:y],x[2][:y]).T for x,y in zip(s,n2)]
			ans=np.array(ans).reshape(*m.shape)
			n2=n2.reshape(*m.shape[:-2])
		elif method=='sklearn':
			raise NotImplementedError('Not supporting >2 dimensions for method=sklearn.')
		else:
			raise ValueError('Unknown method {}'.format(method))
	try:
		if n2.ndim==0:
			n2=n2.item()
	except:
		pass
	return ans,n2

def gexpand_add(dg):
	"""Expand genotype using additive model.

	dg:		numpy.array([ng,nd],dtype='u1') as input genotype matrix.
	dge:	numpy.array([ng,1,nd],dtype='u1') as output expanded genotype matrix."""
	dge=dg.reshape(dg.shape[0],1,dg.shape[1]).astype('u1')
	return dge

def gexpand_cat(dg):
	"""Expand genotype using categorical model.

	dg:		numpy.array([ng,nd],dtype='u1') as input genotype matrix.
	dge:	numpy.array([ng,n-1,nd],dtype='u1') as output expanded genotype matrix.
	n:		Number of unique values in dg."""
	import numpy as np
	t1=np.unique(dg.flatten())[1:]
	assert len(t1)>0
	dge=np.array([dg==x for x in t1]).astype('u1')
	dge=dge.transpose(1,0,2)
	return dge

def association_test_1(vx,vy,dx,dy,dc,dci,dcr,dimreduce=0):
	"""Fast association testing in single-cell non-cohort settings with covariates.

	Single threaded version to allow for parallel computing from outside.
	Mainly used for naive differential expression and co-expression.
	Computes the p-value of null (gamma=0) in model Y=X*gamma+C*alpha+epsilon,
	where epsilon ~ i.i.d. N(0,sigma^2). X and Y are vectors in each test. C is a matrix.
	Uses analytical method to compute p-values from R^2.
	vx,vy:	Starting indices of dx and dy. Only used for info passing.
	dx:		numpy.array(shape=(nx,ns)). Predictor matrix for a list of X to be tested, e.g. gene expression or grouping.
	dy:		numpy.array(shape=(ny,ns)). Target matrix for a list of Y to be tested, e.g. gene expression.
	dc:		numpy.array(shape=(nc,ns)). Covariate/confounder matrix for C.
	dci:	numpy.array(shape=(nc,nc)). Low-rank compatible inverse matrix of dc*dc.T.
	dcr:	Rank of dc*dc.T.
	dimreduce:	np.array(shape=(ny,),dtype='uint') or uint. If Y doesn't have full rank in the first place,
				this parameter allows to specify the loss to allow for accurate p-value computation.

	Return:	[vx,vy,pv,gamma,alpha,var_x,var_y]
	vx,
	vy:		For info passing.
	pv:		numpy.array(shape=(nx,ny)). P-values of association testing (gamma==0).
	gamma:	numpy.array(shape=(nx,ny)). MLE of gamma in model.
	alpha:	numpy.array(shape=(nx,ny,nc)). MLE of alpha in model.
	var_x,
	var_y:	numpy.array(shape=(nx or ny)). variance of dx/dy unexplained by covariates."""
	import numpy as np
	from scipy.stats import beta
	import logging
	if len(dx.shape)!=2 or len(dy.shape)!=2 or len(dc.shape)!=2:
		raise ValueError('Incorrect dx/dy/dc size.')
	n=dx.shape[1]
	if dy.shape[1]!=n or dc.shape[1]!=n:
		raise ValueError('Unmatching dx/dy/dc dimensions.')
	nc=dc.shape[0]
	if nc==0:
		logging.warn('No covariate dc input.')
	elif dci.shape!=(nc,nc):
		raise ValueError('Unmatching dci dimensions.')
	if dcr<0:
		raise ValueError('Negative dcr detected.')
	elif dcr>nc:
		raise ValueError('dcr higher than covariate dimension.')

	nx=dx.shape[0]
	ny=dy.shape[0]

	dx1=dx
	dy1=dy

	if dcr>0:
		#Remove covariates
		ccx=np.matmul(dci,np.matmul(dc,dx1.T)).T
		ccy=np.matmul(dci,np.matmul(dc,dy1.T)).T
		dx1=dx1-np.matmul(ccx,dc)
		dy1=dy1-np.matmul(ccy,dc)
	ansvx=(dx1**2).mean(axis=1)
	ansvx[ansvx==0]=1
	ansvy=(dy1**2).mean(axis=1)
	ansvy[ansvy==0]=1
	ansc=(np.matmul(dy1,dx1.T)/(n*ansvx)).T
	ansp=((ansc**2).T*ansvx).T/ansvy
	if dcr>0:
		ansa=np.repeat(ccy.reshape(1,ccy.shape[0],ccy.shape[1]),ccx.shape[0],axis=0)-(ansc*np.repeat(ccx.T.reshape(ccx.shape[1],ccx.shape[0],1),ccy.shape[0],axis=2)).transpose(1,2,0)
	else:
		ansa=np.zeros((nx,ny,nc),dtype=dx.dtype)

	#Compute p-values
	assert (ansp>=0).all() and (ansp<=1+1E-8).all()
	ansp=beta.cdf(1-ansp,(n-1-dcr-dimreduce)/2,0.5)
	assert ansp.shape==(nx,ny) and ansc.shape==(nx,ny) and ansa.shape==(nx,ny,nc) and ansvx.shape==(nx,) and ansvy.shape==(ny,)
	assert np.isfinite(ansp).all() and np.isfinite(ansc).all() and np.isfinite(ansa).all() and np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp>=0).all() and (ansp<=1).all() and (ansvx>=0).all() and (ansvy>=0).all()
	return [vx,vy,ansp,ansc,ansa,ansvx,ansvy]

def association_test_2(vx,vy,dx,dy,dc,sselectx,dimreduce=0):
	"""Like association_test_1, but takes a different subset of samples for each X.

	See association_test_1 for details on unexplained variables.
	sselectx:	numpy.array(shape=(nx,ns),bool). Subset of samples to use for each X.
	Return:		Same as association_test_1, except:
	var_x:	numpy.array(shape=(nx)). variance of dx unexplained by covariates.
	var_y:	numpy.array(shape=(nx,ny)). variance of dx/dy unexplained by covariates."""
	import numpy as np
	import logging
	from scipy.stats import beta
	if len(dx.shape)!=2 or len(dy.shape)!=2 or len(dc.shape)!=2:
		raise ValueError('Incorrect dx/dy/dc size.')
	n=dx.shape[1]
	if dy.shape[1]!=n or dc.shape[1]!=n:
		raise ValueError('Unmatching dx/dy/dc dimensions.')
	nc=dc.shape[0]
	if nc==0:
		logging.warn('No covariate dc input.')
	if sselectx.shape!=dx.shape:
		raise ValueError('Unmatching sselectx dimensions.')

	nx=dx.shape[0]
	ny=dy.shape[0]
	ansp=np.zeros((nx,ny),dtype=float)
	ansvx=np.zeros((nx,),dtype=float)
	ansvy=np.zeros((nx,ny),dtype=float)
	ansc=np.zeros((nx,ny),dtype=float)
	ansa=np.zeros((nx,ny,nc),dtype=float)
	ansn=np.zeros((nx,ny),dtype=int)

	for xi in range(nx):
		#Select samples
		t1=np.nonzero(sselectx[xi])[0]
		ns=len(t1)
		if len(np.unique(dx[xi,t1]))<2:
			ans[xi]=0
			continue
		dx1=dx[xi,t1]
		dy1=dy[:,t1]
		if nc>0:
			#Remove covariates
			dc1=dc[:,t1]
			t1=np.matmul(dc1,dc1.T)
			t1i,r=inv_rank(t1)
		else:
			r=0
		ansn[xi]=r
		if r>0:
			#Remove covariates
			ccx=np.matmul(t1i,np.matmul(dc1,dx1.T)).T
			ccy=np.matmul(t1i,np.matmul(dc1,dy1.T)).T
			dx1=dx1-np.matmul(ccx,dc1)
			dy1=dy1-np.matmul(ccy,dc1)
		t1=(dx1**2).mean()
		if t1==0:
			#X completely explained by covariate. Should never happen in theory.
			t1=1
		ansvx[xi]=t1
		ansvy[xi]=(dy1**2).mean(axis=1)
		ansc[xi]=np.matmul(dx1,dy1.T).flatten()/(ns*t1)
		if r>0:
			ansa[xi]=ccy-np.repeat(ansc[xi].reshape(ny,1),nc,axis=1)*ccx.flatten()
		ansp[xi]=(ansc[xi]**2)*t1/ansvy[xi]

	#Compute p-values
	assert (ansp>=0).all() and (ansp<=1+1E-8).all()
	ansp=beta.cdf(1-ansp,(sselectx.sum(axis=1)-1-ansn.T-dimreduce).T/2,0.5)
	assert ansp.shape==(nx,ny) and ansc.shape==(nx,ny) and ansa.shape==(nx,ny,nc) and ansvx.shape==(nx,) and ansvy.shape==(nx,ny)
	assert np.isfinite(ansp).all() and np.isfinite(ansc).all() and np.isfinite(ansa).all() and np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp>=0).all() and (ansp<=1).all() and (ansvx>=0).all() and (ansvy>=0).all()
	return [vx,vy,ansp,ansc,ansa,ansvx,ansvy]

def prod1(vx,vy,dx,dy):
	import numpy as np
	return [vx,vy,np.matmul(dx,dy.T)]

def association_test_4(vx,vy,prod,prody,prodyy,na,dimreduce=0,**ka):
	"""Like association_test_1, but regards all other X's as covariates when testing each X.

	See association_test_1 for details on unexplained variables.
	Note:	Other X's are treated as covariates but would not include their alphas in return to reduce memory footprint.
	vx,
	vy:		Starting indices of dx and dy for computation. vy is only used for info passing.
	prod:	Matrix product of A*A.T, where A=numpy.block([dx,dc])
	prody:	Matrix product of A*Y.T, where A=numpy.block([dx,dc])
	prodyy:	Matrix product of diag(Y*Y.T).
	na:		[nx,ny,nc,ns,lenx] Numbers of (Xs, Ys, Cs, samples/cells, X to compute association for)
	ka:		Keyword args for inv_rank.
	Return:	Same as association_test_1, except:
	var_x:	numpy.array(shape=(lenx,)). Variance of dx unexplained by covariates.
	var_y:	numpy.array(shape=(lenx,ny)). Variance of dy unexplained by covariates."""
	import numpy as np
	from scipy.stats import beta
	import logging
	if len(na)!=5:
		raise ValueError('Wrong format for na')
	nx,ny,nc,n,lenx=na
	if nx==0 or ny==0 or n==0:
		raise ValueError('Dimensions in na==0 detected.')
	if nc==0:
		logging.warn('No covariate dc input.')
	if lenx<=0:
		raise ValueError('lenx must be positive.')
	if vx<0 or vx+lenx>nx:
		raise ValueError('Wrong values of vx and/or lenx, negative or beyond nx.')
	if prod.shape!=(nx+nc,nx+nc):
		raise ValueError('Unmatching shape for prod. Expected: {}. Got: {}.'.format((nx+nc,nx+nc),prod.shape))
	if prody.shape!=(nx+nc,ny):
		raise ValueError('Unmatching shape for prody. Expected: {}. Got: {}.'.format((nx+nc,ny),prody.shape))
	if prodyy.shape!=(ny,):
		raise ValueError('Unmatching shape for prodyy. Expected: {}. Got: {}.'.format((ny,),prodyy.shape))
	ansp=np.zeros((lenx,ny),dtype=float)
	ansvx=np.zeros((lenx,),dtype=float)
	ansvy=np.zeros((lenx,ny),dtype=float)
	ansc=np.zeros((lenx,ny),dtype=float)
	ansa=np.zeros((lenx,ny,nc),dtype=float)
	ansn=np.zeros((lenx,ny),dtype=int)

	for xi in range(lenx):
		t0=list(filter(lambda x:x!=vx+xi,range(nx+nc)))
		if len(t0)>0:
			t1=prod[np.ix_(t0,t0)]
			t1i,r=inv_rank(t1,**ka)
		else:
			r=0
		ansn[xi]=r
		if r==0:
			#No covariate
			dxx=prod[vx+xi,vx+xi]/n
			dyy=prodyy/n
			dxy=prody[vx+xi]/n
		else:
			ccx=np.matmul(prod[[vx+xi],t0],t1i)
			dxx=(prod[vx+xi,vx+xi]-float(np.matmul(ccx,prod[t0,[vx+xi]])))/n
			ccy=np.matmul(prody[t0].T,t1i)
			dyy=(prodyy-(ccy.T*prody[t0]).sum(axis=0))/n
			dxy=(prody[vx+xi]-np.matmul(ccy,prod[t0,[vx+xi]]).flatten())/n
		if dxx==0:
			#X completely explained by covariate. Should never happen in theory.
			dxx=1
		ansvx[xi]=dxx
		ansvy[xi]=dyy
		ansc[xi]=dxy/dxx
		if r>0:
			ansa[xi]=ccy[:,-nc:]-np.repeat(ansc[xi].reshape(ny,1),nc,axis=1)*ccx[-nc:]
		ansp[xi]=(dxy**2)/(dxx*dyy)

	#Compute p-values
	assert (ansp>=0).all() and (ansp<=1+1E-8).all()
	ansp=beta.cdf(1-ansp,(n-1-ansn-dimreduce)/2,0.5)
	assert ansp.shape==(lenx,ny) and ansc.shape==(lenx,ny) and ansa.shape==(lenx,ny,nc) and ansvx.shape==(lenx,) and ansvy.shape==(lenx,ny)
	assert np.isfinite(ansp).all() and np.isfinite(ansc).all() and np.isfinite(ansa).all() and np.isfinite(ansvx).all() and np.isfinite(ansvy).all()
	assert (ansp>=0).all() and (ansp<=1).all() and (ansvx>=0).all() and (ansvy>=0).all()
	return [vx,vy,ansp,ansc,ansa,ansvx,ansvy]






































assert __name__ != "__main__"

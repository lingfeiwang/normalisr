#!/usr/bin/python3

import multiprocessing

autocount=lambda:multiprocessing.cpu_count()

def autopooler_caller(a):
	return a[0](*a[1],**a[2])

def autopooler(n,it,*a,chunksize=1,dummy=False,return_iter=False,unordered=False,**ka):
	"""Uses multiprocessing.Pool or multiprocessing.dummy.Pool to run iterator in parallel.
	n:	Number of parallel processes. Set to 0 to use auto detected CPU count.
	it:	Iterator of [func,a,ka], so each iteration computes func(*a,**ka). Func must be picklable, i.e. a base level function in a module or file.
	a:	args passed to Pool
	chunksize:	Number of iterations passed to each process each time.
	dummy:	Whether to use multiprocessing.dummy instead
	return_iter:	Not Implemented. Whether to return iterator of results instead. If not, return list of results.
	unordered:	Whether the order of output matters.
	ka:	Keyword args passed to Pool
	Return:	List (or iterator if return_iter) of results returned by func(*a,**ka), in same order of the iterator."""
	import multiprocessing
	import logging
	if dummy:
		import multiprocessing.dummy as m
	else:
		import multiprocessing as m
	if n==0:
		import logging
		n=autocount()
		logging.info('Using {} threads'.format(n))
	if n==1:
		ans=map(autopooler_caller,it)
		if not return_iter:
			ans=list(ans)
			assert len(ans)>0
	else:
		import itertools
		#Catches iterator errors (only if occurs at the first), and emptiness
		it=itertools.chain([next(it)],it)
		with m.Pool(n,*a,**ka) as p:
			if unordered:
				ans=p.imap_unordered(autopooler_caller,it,chunksize)
			else:
				ans=p.imap(autopooler_caller,it,chunksize)
			if not return_iter:
				ans=list(ans)
			else:
				raise NotImplementedError
	return ans










































assert __name__ != "__main__"

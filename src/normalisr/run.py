#!/usr/bin/python3

fmt_float='%.8G'
fmt_int='%i'
import numpy as np
import logging

def file_read_coo(f,**ka):
	"""Read COO/mtx files"""
	import scipy
	import scipy.io
	logging.debug('Start reading file '+f)
	ans=scipy.io.mmread(f,**ka)
	logging.debug('Finish reading file '+f)
	return ans

def file_read_tsv(f,delimiter='\t',**ka):
	"""Read tsv files"""
	logging.debug('Start reading file '+f)
	ans=np.loadtxt(f,delimiter=delimiter,**ka)
	logging.debug('Finish reading file '+f)
	if ans.ndim==1:
		ans=ans.reshape(1,-1)
	return ans

def file_write_tsv(f,d,delimiter='\t',fmt=fmt_float,**ka):
	"""Write tsv files"""
	logging.debug('Start writing file '+f)
	ans=np.savetxt(f,d,delimiter=delimiter,fmt=fmt,**ka)
	logging.debug('Finish writing file '+f)
	return ans

def file_write_table(f,d,compression='infer',**ka):
	"""Write pandas.dataframe to table tsv file"""
	if compression=='infer' and f.endswith('.gz'):
		compression='gzip'
	logging.debug('Start writing file '+f)
	ans=d.to_csv(f,compression=compression,**ka)
	logging.debug('Finish writing file '+f)
	return ans

def file_read_txtlist(f,**ka):
	"""Read txt file as list"""
	logging.debug('Start reading file '+f)
	with open(f,'r',**ka) as fh:
		ans=fh.readlines()
	logging.debug('Finish reading file '+f)
	ans=[x.strip() for x in ans]
	ans=np.array(list(filter(lambda x:len(x)>0,ans)))
	return ans

def file_write_txtlist(f,d,**ka):
	"""Write txt file as list"""
	from os import linesep
	d=linesep.join(d)
	logging.debug('Start writing file '+f)
	with open(f,'w',**ka) as fh:
		fh.write(d)
	logging.debug('Finish writing file '+f)
	return d

def qc_reads(args):
	from .qc import qc_reads
	#Read input files
	if args['sparse']:
		d=file_read_coo(args['reads_in']).astype(int).toarray()
	else:
		d=file_read_tsv(args['reads_in'],dtype=int)
	nt,ns=d.shape
	namet=file_read_txtlist(args['genes_in'])
	if len(namet)!=nt:
		raise ValueError("Gene count in genes_in doesn't match row count in reads_in.")
	names=file_read_txtlist(args['cells_in'])
	if len(names)!=ns:
		raise ValueError("Cell count in cells_in doesn't match column count in reads_in.")

	#Run computation
	logging.debug('Start calculation.')
	qc=qc_reads(d,args['n_gene'],args['nc_gene'],args['ncp_gene'],args['n_cell'],args['nt_cell'],args['ntp_cell'])
	namet=namet[qc[0]]
	names=names[qc[1]]
	logging.debug('Finish calculation.')

	#Write output files
	file_write_txtlist(args['genes_out'],namet)
	file_write_txtlist(args['cells_out'],names)

def subset(args):
	if args['r'] is None and args['c'] is None:
		raise ValueError('Please indicate row (-r) or column (-c) for subsetting.')
	if args['nodummy'] and args['r'] is not None and args['c'] is not None:
		raise ValueError('Only supports nodumy when one of row or column needs subsetting.')

	#Read input file
	if args['sparse']:
		d=file_read_coo(args['matrix_in']).astype(int).toarray()
	else:
		d=file_read_tsv(args['matrix_in'])

	#Subset rows
	if args['r'] is not None:
		names=[file_read_txtlist(x) for x in args['r']]
		assert len(names[0])==d.shape[0]
		assert np.all([len(set(x))==len(x) for x in names])
		t1=dict(zip(names[0],range(len(names[0]))))
		t2=list(filter(lambda x:x not in t1,names[1]))
		if len(t2)>0:
			raise ValueError('Subset row names not found: {}...'.format(','.join(t2[:3])))
		t1=[t1[x] for x in names[1]]
		d=d[t1]
		if args['nodummy']:
			t1=[len(np.unique(x))>1 for x in d.T]
			d=d[:,t1]
	#Subset columns
	if args['c'] is not None:
		names=[file_read_txtlist(x) for x in args['c']]
		assert len(names[0])==d.shape[1]
		assert np.all([len(set(x))==len(x) for x in names])
		t1=dict(zip(names[0],range(len(names[0]))))
		t2=list(filter(lambda x:x not in t1,names[1]))
		if len(t2)>0:
			raise ValueError('Subset column names not found: {}...'.format(','.join(t2[:3])))
		t1=[t1[x] for x in names[1]]
		d=d[:,t1]
		if args['nodummy']:
			t1=[len(np.unique(x))>1 for x in d]
			d=d[t1]
	if np.any([x==0 for x in d.shape]):
		raise RuntimeError('Empty matrix after subsetting, maybe because nodummy option.')
	#Write output files
	file_write_tsv(args['matrix_out'],d,fmt=fmt_int if np.issubdtype(d.dtype,np.integer) else fmt_float)

def lcpm(args):
	from .lcpm import lcpm,scaling_factor
	#Read input files
	if args['sparse']:
		d=file_read_coo(args['reads_in']).astype(int)
	else:
		d=file_read_tsv(args['reads_in'],dtype=int)
	if args['cov_in'] is not None:
		cov=file_read_tsv(args['cov_in'])
	else:
		cov=None

	#Optional parameters
	ka=dict()
	if args['nth'] is not None:
		if args['nth']<0:
			raise ValueError('Parameter nth must be non-negative.')
		ka['nth']=args['nth']
	if args['rseed'] is not None:
		ka['seed']=args['rseed']
	#Run computation
	logging.debug('Start calculation.')
	ans=lcpm(d,**ka)
	ans2=scaling_factor(d)
	logging.debug('Finish calculation.')
	#Concatenate covariates
	if cov is not None:
		cov=np.concatenate([cov,ans[3]],axis=0)
	else:
		cov=ans[3]

	#Write output files
	file_write_tsv(args['lcpm_out'],ans[0])
	file_write_tsv(args['cov_out'],cov)
	file_write_tsv(args['scale_out'],ans2)
	if args['var_out'] is not None:
		file_write_tsv(args['var_out'],ans[2])

def normcov(args):
	from .norm import normcov
	#Read input files
	dc=file_read_tsv(args['cov_in'])
	#Run computation
	logging.debug('Start calculation.')
	if args['no1']:
		ans=normcov(dc,c=False)
	else:
		ans=normcov(dc)
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['cov_out'],ans)

def fitvar(args):
	from .norm import compute_var
	#Read input files
	dt=file_read_tsv(args['lcpm_in'])
	dc=file_read_tsv(args['cov_in'])
	#Run computation
	logging.debug('Start calculation.')
	mult=compute_var(dt,dc)
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['weights_out'],mult)

def qc_outlier(args):
	from .qc import qc_outlier
	#Read input files
	d=file_read_tsv(args['weights_in']).flatten()
	names=file_read_txtlist(args['cells_in'])
	if len(names)!=len(d):
		raise ValueError("Cell count in cells_in doesn't match entry count in weights_in.")
	#Run computation
	logging.debug('Start calculation.')
	qc=qc_outlier(d,outrate=args['outrate'],pcut=args['pcut'])
	names=names[qc]
	logging.debug('Finish calculation.')
	#Write output files
	file_write_txtlist(args['cells_out'],names)

def normvar(args):
	from .norm import normvar
	#Read input files
	dt=file_read_tsv(args['lcpm_in'])
	dc=file_read_tsv(args['cov_in'])
	dmult=file_read_tsv(args['weights_in']).flatten()
	dw=file_read_tsv(args['scale_in']).flatten()
	#Run computation
	logging.debug('Start calculation.')
	ans=normvar(dt,dc,dmult,dw)
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['exp_out'],ans[0])
	file_write_tsv(args['cov_out'],ans[1])

def de(args):
	from .de import de as de_func
	#Read input files
	dg=file_read_tsv(args['design_in'])
	dt=file_read_tsv(args['exp_in'])
	dc=file_read_tsv(args['cov_in'])
	#Run computation
	ka=dict()
	if args['method'] is not None:
		ka['single']={'ignore':0,'single':1,'covariate':4}[args['method']]
	if args['nth'] is not None:
		ka['nth']=args['nth']
	if args['bs'] is not None:
		ka['bs']=args['bs']
	if args['dimr'] is not None:
		ka['dimreduce']=args['dimr']
	logging.debug('Start calculation.')
	ans=de_func(dg,dt,dc,**ka)
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['pv_out'],ans[0])
	#Convert ln to log2
	file_write_tsv(args['lfc_out'],ans[1])
	if args['clfc_out'] is not None:
		file_write_tsv(args['clfc_out'],ans[2])
	if args['vard_out'] is not None:
		file_write_tsv(args['vard_out'],ans[3])
	if args['vart_out'] is not None:
		file_write_tsv(args['vart_out'],ans[4])

def coex(args):
	from .coex import coex as coex_func
	#Read input files
	dt=file_read_tsv(args['exp_in'])
	dc=file_read_tsv(args['cov_in'])
	#Run computation
	ka=dict()
	if args['nth'] is not None:
		ka['nth']=args['nth']
	if args['bs'] is not None:
		ka['bs']=args['bs']
	if args['dimr'] is not None:
		ka['dimreduce']=args['dimr']
	logging.debug('Start calculation.')
	ans=coex_func(dt,dc,**ka)
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['pv_out'],ans[0])
	if args['dot_out'] is not None:
		file_write_tsv(args['dot_out'],ans[1])
	if args['var_out'] is not None:
		file_write_tsv(args['var_out'],ans[2])

def binnet(args):
	from .binnet import binnet as binnet_func
	#Read input files
	net=file_read_tsv(args['pv_in'])
	logging.debug('Start calculation.')
	ans=binnet_func(net,args['qcut'])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['net_out'],ans.astype('u1'),fmt=fmt_int)

def gocovt(args):
	from .gocovt import gotop,pccovt
	#Read input files
	dt=file_read_tsv(args['exp_in'])
	dc=file_read_tsv(args['cov_in'])
	net=file_read_tsv(args['net_in'],dtype='u1').astype(bool)
	namet=file_read_txtlist(args['genes_in'])
	ka=dict()
	if 'n' in args:
		ka['n']=args['n']
	if 'm' in args:
		ka['nmin']=args['m']
	if args['c'] is not None and args['c'][0]!=args['c'][1]:
		ka['conversion']=tuple(args['c'])
	#Run computation
	logging.debug('Start calculation.')
	ans1=gotop(net,namet,args['go_in'],args['goa_in'],**ka)
	ans2=pccovt(dt,dc,namet,ans1[3])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['cov_out'],ans2)
	if args['master_out'] is not None:
		file_write_txtlist(args['master_out'],ans1[0])
	if args['goe_out'] is not None:
		file_write_table(args['goe_out'],ans1[1],header=True,index=False,sep='\t')
	if args['go_out'] is not None:
		file_write_txtlist(args['go_out'],ans1[2])

def cohort_kinshipeigen(args):
	#Read input files
	mk=file_read_tsv(args['kinship_in'])
	nc=file_read_tsv(args['ncell_in'],dtype=int).flatten()
	#Run computation
	from .cohort.kinship import eigen
	logging.debug('Start calculation.')
	ans=eigen(mk,nc,tol=args['tol'])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['eigenvalue_out'],ans[0])
	file_write_tsv(args['eigenvector_out'],ans[1])

def cohort_heritability(args):
	#Read input files
	dt=file_read_tsv(args['transcript_in'])
	dc=file_read_tsv(args['cov_in'])
	nc=file_read_tsv(args['ncell_in'],dtype=int).flatten()
	mkl=file_read_tsv(args['eigenvalue_in']).flatten()
	mku=file_read_tsv(args['eigenvector_in'])
	#Run computation
	from .cohort.heritability import estimate
	logging.debug('Start calculation.')
	ans=estimate(dt,dc,nc,mkl,mku,tol=args['tol'])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['sigma_out'],ans[0])
	file_write_tsv(args['beta_out'],ans[1])
	file_write_tsv(args['alpha_out'],ans[2])

def cohort_eqtl(args):
	#Read input files
	dg=file_read_tsv(args['genotype_in'],dtype=int)
	dt=file_read_tsv(args['transcript_in'])
	dc=file_read_tsv(args['cov_in']) if 'cov_in' in args else None
	ns=file_read_tsv(args['ncell_in'],dtype=int).flatten()
	mkl=file_read_tsv(args['eigenvalue_in']).flatten()
	mku=file_read_tsv(args['eigenvector_in'])
	sigma=file_read_tsv(args['sigma_in']).flatten()
	beta=file_read_tsv(args['beta_in']).flatten()
	#Run computation
	from .cohort.eqtl import eqtl as eqtlf
	logging.debug('Start calculation.')
	ans,ansgamma=eqtlf(dg,dt,dc,ns,mkl,mku,sigma,beta,gmodel=args['model'],bst=args['nts'],nth=args['nth'],dimreduce=args['dimr'])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['eqtlpv_out'],ans)
	file_write_tsv(args['eqtlgamma_out'],ansgamma.reshape(ansgamma.shape[0]*ansgamma.shape[1],ansgamma.shape[2]))

def cohort_coex(args):
	#Read input files
	dt=file_read_tsv(args['transcript_in'])
	dc=file_read_tsv(args['cov_in']) if 'cov_in' in args else None
	ns=file_read_tsv(args['ncell_in'],dtype=int).flatten()
	mkl=file_read_tsv(args['eigenvalue_in']).flatten()
	mku=file_read_tsv(args['eigenvector_in'])
	beta=file_read_tsv(args['beta_in']).flatten()
	#Run computation
	from .cohort.coex import coex as coexf
	logging.debug('Start calculation.')
	ans=coexf(dt,dc,ns,mkl,mku,beta,bst=args['nts'],nth=args['nth'],dimreduce=args['dimr'])
	logging.debug('Finish calculation.')
	#Write output files
	file_write_tsv(args['coexpv_out'],ans)




































assert __name__ != "__main__"

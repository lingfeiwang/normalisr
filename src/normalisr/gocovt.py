#!/usr/bin/python3

def pc1(d):
	"""Helper function to compute top principal component.
	d:		[n,ns]
	Return:	PC 1 as [ns]
	"""
	import numpy as np
	from sklearn.decomposition import TruncatedSVD
	#Normalize data
	t1=d.T
	t1=t1-t1.mean(axis=0)
	t1=t1/(np.sqrt((t1**2).mean(axis=0))+1E-200)
	t0=TruncatedSVD(n_components=1)
	t1=t0.fit_transform(t1).T.astype(d.dtype).flatten()
	assert t1.shape==(d.shape[1],)
	return t1

def goe(genelist,go_file,goa_file,bg=None,nmin=5,conversion=None,evidence_set= {'EXP','IDA','IPI','IMP','IGI','HTP','HDA','HMP','HGI','IBA','IBD','IKR','IRD','ISS','ISO','ISA','ISM'},**ka):
	"""Use python package goatools (0.7.11 tested) to find GO enrichment.
	WARNING: This method is inaccurate for multimaps in gene name conversion.
	However, it has a negligible effect in top GO component removal in single-cell co-expression.
	genelist:	The list of genes to search enrichment
	go_file,
	goa_file:	Files for GO definitions and GO associations.
	bg:			Background gene list
	nmin:		Minimum number of principal genes required in GO
	conversion:	(name_from,name_to,species) if gene list needs conversion to gene ID systems in the GO annotation.
				Names of gene naming systems can be found at https://docs.mygene.info/en/latest/doc/data.html.
		name_from:	Gene naming system of genelist. For gene names, use 'symbol,alias'
		name_to:	Gene naming system of goa_file. Examples:
					For human (http://geneontology.org/gene-associations/goa_human.gaf.gz), use 'uniprot.Swiss-Prot'.
					For mouse (http://current.geneontology.org/annotations/mgi.gaf.gz), use 'MGI'.
		species:	Species for gene name conversion. Examples: human, mouse.
	evidence_set:	Set of GO evidences to include. Defaults for non-expression based results.
	ka:			arguments passed to enrichment.
	Return:		(goe,gotop,genes)
	goe:		Pandas DataFrame of GO enrichment
	gotop:		Top enriched GO ID
	genes:		Genes in the gotop from the bg list. genes=None if bg is None.
	"""
	from tempfile import NamedTemporaryFile
	from os import linesep
	from goatools.go_enrichment import GOEnrichmentStudy
	from goatools.obo_parser import GODag
	from goatools.associations import read_gaf
	from goatools.associations import read_associations
	from collections import defaultdict
	import itertools
	import mygene
	import pandas as pd
	from os.path import join as pjoin
	import logging
	from operator import truediv
	import numpy as np
	assert type(genelist) is list and len(genelist)>0
	if nmin<1:
		nmin=1

	bg0=bg
	#Convert gene names
	if conversion is not None:
		assert len(conversion)==3
		name_from,name_to,species=conversion
		mg=mygene.MyGeneInfo()
		ans=set(genelist)
		if bg is not None:
			t1=set(bg)
			assert len(ans-t1)==0
			ans|=t1
		ans=list(ans)
		ans=mg.querymany(ans,scopes=name_from,fields=name_to,species=species)
		t1=set(['query','_score',name_to.split('.')[0]])
		ans=list(filter(lambda x:len(t1-set(x))==0,ans))
		ans=sorted(ans,key=lambda x:x['_score'])
		convert={x['query']:x for x in ans}
		for xi in name_to.split('.'):
			convert=filter(lambda x:xi in x[1],convert.items())
			convert={x[0]:x[1][xi] for x in convert}
		convert={x[0]:x[1] if type(x[1]) is str else x[1][0] for x in convert.items()}
		genelist2=list(set([convert[x] for x in filter(lambda x:x in convert,genelist)]))
		if bg is not None:
			bg=list(set([convert[x] for x in filter(lambda x:x in convert,bg)]))
		t1=set(genelist)
		converti=list(filter(lambda x:x[0] in t1,convert.items()))
		t1=defaultdict(list)
		for xi in converti:
			t1[xi[1]].append(xi[0])
		converti=dict(t1)
		t1=defaultdict(list)
		for xi in convert.items():
			t1[xi[1]].append(xi[0])
		convertia=dict(t1)
	else:
		genelist2=genelist

	#Load GO DAG and association files
	logging.debug('Reading GO DAG file '+go_file)
	godag=GODag(go_file)
	logging.debug('Reading GO association file '+goa_file)
	goa=read_gaf(goa_file,evidence_set=evidence_set)
	if bg is None:
		bg=list(goa.keys())

	#Compute enrichment
	goe=GOEnrichmentStudy(bg,goa,godag)
	ans=goe.run_study(genelist2)
	#Format output
	with NamedTemporaryFile() as f:
		goe.wr_tsv(f.name,ans)
		ans=f.read()
	ans=ans.decode()
	ans=[x.split('\t') for x in ans.split(linesep)]
	if len(ans[-1])<2:
		ans=ans[:-1]
	if len(ans)==0 or len(ans[0])==0:
		raise ValueError('No enrichment found. Check your input ID type.')
	ans[0][0]=ans[0][0].strip('# ')
	ans=pd.DataFrame(ans[1:],columns=ans[0])
	ans.drop(['NS','enrichment','study_count','p_sidak','p_holm'],axis=1,inplace=True)
	for xj in ['p_uncorrected','p_bonferroni']:
		ans[xj]=pd.to_numeric(ans[xj],errors='raise')
	ans['depth']=pd.to_numeric(ans['depth'],errors='raise',downcast='unsigned')
	#Odds ratio column and sort column
	toratio=lambda z:z.apply(lambda x:truediv(*[int(y) for y in x.split('/')]))
	ans['odds_ratio']=toratio(ans['ratio_in_study'])/toratio(ans['ratio_in_pop'])
	ans=ans[['name','depth','p_uncorrected','p_bonferroni','odds_ratio','ratio_in_study','ratio_in_pop','GO','study_items']]
	ans['study_items']=ans['study_items'].apply(lambda x:x.replace(' ',''))
	#Convert back study_items
	if conversion is not None:
		ans['study_items']=ans['study_items'].apply(lambda x:','.join(list(itertools.chain.from_iterable([converti[y] for y in x.split(',')]))) if len(x)>0 else x)
	ans.sort_values('p_uncorrected',inplace=True)

	#Get top enriched GO by P-value
	gotop=ans[(ans['odds_ratio']>1)&ans['ratio_in_study'].apply(lambda x:int(x.split('/')[0])>=nmin)]
	if len(gotop)==0:
		raise ValueError('No GO enrichment found for given criteria.')
	gotop=str(gotop.iloc[0]['GO'])
	if bg0 is not None:
		#Children GOs
		gos=set([gotop]+list(godag.query_term(gotop).get_all_children()))
		#Look for genes
		genes=list(filter(lambda x:len(list(filter(lambda y:y in gos,goa[x])))>0,goa))
		if conversion is not None:
			genes=[convertia[x] for x in filter(lambda x:x in convertia,genes)]
			genes=list(set(list(itertools.chain.from_iterable(genes))))
		genes=set(genes)
		genes=list(filter(lambda x:x in genes,bg0))
	else:
		genes=None
	return (ans,gotop,genes)

def gotop(net,namet,go_file,goa_file,n=100,**ka):
	"""Finds the top variable GO enrichment of top principal genes in the binary co-expression network.

	Principal genes are those with most co-expressed genes. They reflect the most variable pathways in the dataset. When the variable pathways are housekeeping related, they may conceal cell-type-specific co-expression patterns from being observed and understood. This function identifies the most variable pathway with gene ontology enrichment study of the top principal genes. Background genes are all genes provided.

	Parameters
	-----------
	net:		numpy.ndarray(shape=(n_gene,n_gene),dtype=bool)
		Binary co-expression network matrix.
	namet:		list of str
		Gene names matching the rows and columns of net.
	go_file:	str
		Path of GO DAG file
	goa_file:	str
		Path of GO annotation file
	n:			int
		Number of top principal genes to include for GO enrichment. Default is 100, giving good performance in general.
	ka:			dict
		**IMPORTANT**: Keyword arguments passed to normalisr.gocovt.goe to determine how to perform GO enrichment study. If you see no gene mapped, check your gene name conversion rule in conversion parameter of goe. GO annotation have a specific gene ID system.

	Returns
	-------
	principals:	list of str
		List of principal genes.
	goe:		pandas.DataFrame
		GO enrichment results.
	gotop:		str
		Top enriched GO ID.
	genes:		list of str
		List of genes in the gotop GO ID.
	"""
	import numpy as np
	nt=len(namet)
	if net.shape!=(nt,nt) or nt<=1:
		raise ValueError('Wrong shape of net or namet.')
	if n<=1 or n>=nt:
		raise ValueError('Number of principal genes must be from 1 to the number of all genes (exclusive).')

	#Find principal genes
	t1=net.sum(axis=1)
	t2=t1[t1.argsort()[::-1][n]]

	if t2==0:
		raise RuntimeError('Not enough principal genes that have co-expression')
	t2=np.nonzero(t1>=t2)[0]
	assert len(t2)>=n
	assert len(t2)<nt
	t1=[str(x) for x in namet[t2]]
	t2=[str(x) for x in namet]
	return tuple([t1]+list(goe(t1,go_file,goa_file,bg=t2,**ka)))

def pccovt(dt,dc,namet,genes,condcov=True):
	"""Introduces an extra covariate from the top principal component of given genes.

	The extra covariate is the top principal component of normalized expressions of the selected genes. Adding a covariate from housekeeping pathway can reveal cell-type-specific activities in co-expression networks.

	Parameters
	----------
	dt:		numpy.ndarray(shape=(n_gene,n_cell))
		Normalized expression matrix.
	dc:		numpy.ndarray(shape=(n_cov,n_cell))
	 	Existing normalized covariate matrix.
	namet:	list of str
		List of gene names for rows in dt.
	genes:	list of str
		List of gene names to include in finding top PC of their expression as an extra covariate.
	condcov:bool
		Whether to condition on existing covariates before computing top PC. Default: True.

	Returns
	-------
	numpy.ndarray(shape=(n_cov+1,n_cell))
		New normalized covariate matrix.
	"""
	import numpy as np
	nt,ns=dt.shape
	nc=dc.shape[0]
	if nt==0 or ns==0:
		raise ValueError('Empty normalized expression.')
	if len(namet)!=nt or dc.shape[1]!=ns:
		raise ValueError('Incompatible input shapes.')
	t1=set(genes)-set(namet)
	if len(t1)>0:
		raise ValueError('Genes not found: {},...'.format(','.join(list(t1)[:3])))

	if condcov and nc>0:
		#Remove covariates
		from sklearn.linear_model import LinearRegression as lr0
		lr=lr0(fit_intercept=True,normalize=True)
		lr.fit(dc.T,dt.T)
		dt=dt-lr.predict(dc.T).T

	nametd=dict(zip(namet,range(len(namet))))
	ans=pc1(dt[[nametd[x] for x in genes]])
	n=1

	dcn=np.concatenate([dc,ans.reshape(1,dc.shape[1])],axis=0)
	assert dcn.shape==(nc+1,ns)
	return dcn






































assert __name__ != "__main__"

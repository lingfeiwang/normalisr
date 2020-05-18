#!/usr/bin/python3

assert __name__ == "__main__"

import argparse
import sys
import logging

p0=argparse.ArgumentParser(prog='normalisr',description='Normalisr is a parameter-free normalization-association two-step inferential framework for scRNA-seq that solves case-control DE, co-expression, and pooled CRISPRi scRNA-seq screen under one umbrella of linear association testing. Normalisr uses submodules for each step of analysis. Details and examples at https://github.com/lingfeiwang/normalisr.')
p0.add_argument('-v',dest='verbose',action='store_true',help='Verbose mode.')
p1s=p0.add_subparsers(help='sub-commands',dest='cmd')


p1=p1s.add_parser('qc_reads',help='Perform quality control (QC) on read count matrix. Default parameter set suits 10x datasets.')
p1.add_argument('reads_in', help='Input matrix file of the number of reads for each gene in each cell. Rows are genes/transcripts; columns are cells. No row or column names allowed. File type: tsv if dense (default), COO/mtx if sparse (need the -s argument).')
p1.add_argument('genes_in', help='Input text file of gene names (row names of reads_in), one per line.')
p1.add_argument('cells_in', help='Input text file of cell names (column names of reads_in), one per line.')
p1.add_argument('genes_out', help='Output text file of gene names passed QC. File format is the same as input.')
p1.add_argument('cells_out', help='Output text file of cell names passed QC. File format is the same as input.')

p1.add_argument('--gene_read_count',dest='n_gene',action='store',type=int,default='0',help='QC: Required number of reads for gene. Set to 0 to disable. Defaults to 0.')
p1.add_argument('--gene_cell_count',dest='nc_gene',action='store',type=int,default='50',help='QC: Required number of cells expressing gene. Set to 0 to disable. Defaults to 50.')
p1.add_argument('--gene_cell_prop',dest='ncp_gene',action='store',type=float,default='0.02',help='QC: Required proportion of cells expressing gene. Set to 0 to disable. Defaults to 0.02.')
p1.add_argument('--cell_read_count',dest='n_cell',action='store',type=int,default='500',help='QC: Required number of reads in cell. Set to 0 to disable. Defaults to 500.')
p1.add_argument('--cell_gene_count',dest='nt_cell',action='store',type=int,default='100',help='QC: Required number of genes expressed in cell. Set to 0 to disable. Defaults to 100.')
p1.add_argument('--cell_gene_prop',dest='ntp_cell',action='store',type=float,default='0',help='QC: Required proportion of genes expressed in cell. Set to 0 to disable. Defaults to 0.')
p1.add_argument('-s',dest='sparse',action='store_true',help='Use sparse (.mtx or .mtx.gz, i.e. COO) input file format for read counts.')
# p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')


p1=p1s.add_parser('subset', help='Subsets matrix based on row and column names selected. Used for gene and cell removal after QC.')
p1.add_argument('matrix_in', help='Input matrix file. No row or column names allowed. File type: tsv if dense (default), COO/mtx if sparse (need the -s argument).')
p1.add_argument('matrix_out', help='Output matrix file after subsetting. File type: tsv.')

p1.add_argument('-r',nargs=2,metavar=('row_before','row_after'),help='Input text file of row names before and after subsetting, one per line.')
p1.add_argument('-c',nargs=2,metavar=('col_before','col_after'),help='Input text file of column names before and after subsetting, one per line.')
p1.add_argument('--nodummy',dest='nodummy',action='store_true',help='Remove single-valued rows/columns when only subsetting on the other.')
p1.add_argument('-s',dest='sparse',action='store_true',help='Use sparse (.mtx or .mtx.gz, i.e. COO) input file format for input matrix.')


p1=p1s.add_parser('lcpm', help='Compute Bayesian expectation of logCPM and cellular summary covariates from read counts.')
p1.add_argument('reads_in', help='Input matrix file of the number of reads for each gene in each cell. Rows are genes/transcripts; columns are cells. No row or column name allowed. File type: tsv if dense (default), COO/mtx if sparse (need the -s argument).')
p1.add_argument('lcpm_out', help='Output matrix file of Bayesian logCPM. File format is the same as input, except being always dense. File type: tsv.')
p1.add_argument('scale_out', help='Output vector file of the scaling factor of variance normalization for each gene. File type: tsv.')
p1.add_argument('cov_out', help='Output matrix file of 3 cellular summary covariates. If the -c option is specified, new covariates will be concatenated after the originals in this output file. File format is the same as input, except being always dense and having 3 (more) rows. File type: tsv.')

p1.add_argument('-s',dest='sparse',action='store_true',help='Use sparse (.mtx or .mtx.gz, i.e. COO) input file format. However, feeding in 10x result directly without quality control is not recommended.')
p1.add_argument('-r',dest='rseed',action='store',type=int,help='Initial random seed')
p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')
p1.add_argument('-c',dest='cov_in',help='Input matrix file of existing covariates to account for potential batch effects. Rows are covariates; columns are cells. If specified, the output covariates will be appended after the input covariates. File type: tsv.')
p1.add_argument('--var_out', help='Output matrix file of the variance of the posterior distribution of logCPM. File format is the same as input, except being always dense. File type: tsv.')


p1=p1s.add_parser('normcov', help="Normalize continuous covariates and include constant 1 covariate as intercept.")
p1.add_argument('cov_in',help='''Input matrix file of covariates  Rows are covariates; columns are cells. No row or column name allowed. Can be cov_out file from method lcpm. File type: tsv.''')
p1.add_argument('cov_out',help='''Output matrix file of normalized covariates, in the same format.''')

p1.add_argument('--no1',dest='no1',action='store_true',default=False,help='Do not add constant 1 covariate. Used only if it is not the last normcov step.')


p1=p1s.add_parser('fitvar', help="Fit lognormal distribution of variance with covariates.")
p1.add_argument('lcpm_in', help='Input matrix file of Bayeisan logCPM of each gene in each cell. Rows are genes/transcripts; columns are cells. No row or column name allowed. File type: tsv.')
p1.add_argument('cov_in',help='''Input matrix file of covariates. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('weights_out',help='''Output vector file of fitted weight (variance**-0.5) for each cell. File type: tsv.''')


p1=p1s.add_parser('qc_outlier', help="Find cell outliers from fitted variance.")
p1.add_argument('weights_in', help='Input vector file of fitted weight of each cell. File type: tsv.')
p1.add_argument('cells_in', help='Input text file of cell names (column names of reads_in), one per line.')
p1.add_argument('cells_out', help='Output text file of cell names passed QC. File format is the same as input.')

p1.add_argument('--pcut',dest='pcut',action='store',type=float,default='1E-10',help='Bonferroni P-value cutoff for outlier detection. Default is 1E-10.')
p1.add_argument('--outrate',dest='outrate',action='store',type=float,default='0.02',help='Maximum rate of outlier on each tail of variance distribution. Used for initial outlier assignment and final validation. Default is 0.02.')


p1=p1s.add_parser('normvar', help="Normalize variances of gene expressions and covariates.")
p1.add_argument('lcpm_in', help='Input matrix file of Bayeisan logCPM of each gene in each cell. Rows are genes/transcripts; columns are cells. No row or column name allowed. File type: tsv.')
p1.add_argument('weights_in', help='Input vector file of fitted weight of each cell. File type: tsv.')
p1.add_argument('cov_in',help='''Input file for covariates matrix. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('scale_in', help='Input vector file of scaling factor of variance normalization for each gene. File type: tsv.')
p1.add_argument('exp_out', help='Output matrix file of normalized expressions, in the same format as lcpm_in.')
p1.add_argument('cov_out',help='''Output matrix file of normalized covariates, in the same format.''')


p1=p1s.add_parser('de', help="Differential expression analysis.")
p1.add_argument('design_in', help='Input design/predictor matrix file. Rows are predictors; columns are cells. No row or column name allowed. Each predictor/grouping is tested separately. How to treat other predictors depends on the method (-m). File type: tsv.')
p1.add_argument('exp_in',help='''Input matrix file of normalized expressions. Rows are genes; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('cov_in',help='''Input matrix file of covariates. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('pv_out', help='Output matrix file of P-values of each differential expression test. Rows are predictors; columns are genes. File type: tsv.')
p1.add_argument('lfc_out', help='Output matrix file of log fold changes. Rows are predictors; columns are genes. File type: tsv.')

p1.add_argument('-m',dest='method',action='store',default='ignore',help='When testing differential expression on each predictor, how to treat other predictors: "ignore" (default): ignore other predictors; "single": use only the subset of cells that have all other predictors==0 (suitable for MOI<1 Perturb-seq/CROP-seq experiments and requires binary design_in matrix); "covariate": regard other predictors as covariates (suitable for MOI>>1 Perturb-seq/CROP-seq experiments).')
p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')
p1.add_argument('-b',dest='bs',action='store',type=int,help='Batch gene/predictor size for each parallel job. For -m ignore or single, specifies batch size for gene and predictor, and defaults to 500; for -m covariate, specifies batch size for predictor only, and defaults to 20.')
p1.add_argument('-d',dest='dimr',action='store',type=int,help='Extra number of effective dimension loss in transcriptome data (in samples/cells) due to preprocessing. Extra dimension loss distorts the null distribution and the p-value estimation if not adjusted for properly. Default: 0.')
p1.add_argument('--clfc_out',dest='clfc_out',action='store',help='Output matrix file (shape=(#predictors tested, #genes, #covariates)) of log fold changes incurred by unit change in each (normalized) covariate in each setting tested. File type: tsv (row majored, needs reshaping back to 3-dimensional).')
p1.add_argument('--vard_out',dest='vard_out',action='store',help='Output vector file of the variance of design matrix unexplained by covariates. Each entry is a predictor. File type: tsv.')
p1.add_argument('--vart_out',dest='vart_out',action='store',help='Output matrix file of the variance of normalized expression unexplained by covariates. Rows are predictors; columns are genes. File type: tsv.')


p1=p1s.add_parser('coex', help="Co-expression analysis.")
p1.add_argument('exp_in',help='''Input matrix file of normalized expressions. Rows are genes; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('cov_in',help='''Input matrix file of covariates. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('pv_out', help='Output matrix file of P-values of gene pairwise co-expression. Rows and colums are genes. File type: tsv.')

p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')
p1.add_argument('-b',dest='bs',action='store',type=int,help='Batch gene size for each parallel job. Defaults to 500.')
p1.add_argument('-d',dest='dimr',action='store',type=int,help='Extra number of effective dimension loss in transcriptome data (in samples/cells) due to preprocessing. Extra dimension loss distorts the null distribution and the p-value estimation if not adjusted for properly. Default: 0.')
p1.add_argument('--var_out',dest='var_out',action='store',help='Output vector file of the variance of normalized expression unexplained by covariates. Each entry is a gene. File type: tsv.')
p1.add_argument('--dot_out',dest='dot_out',action='store',help='Output matrix file of inner/dot product of gene pairs, scaled by cell count. Pearson R=(dot_out/sqrt(var_out)).T/sqrt(var_out). Rows and columns are genes. File type: tsv.')


p1=p1s.add_parser('binnet', help="Binarize P-value co-expression network.")
p1.add_argument('pv_in', help='Input matrix file of P-values of gene pairwise co-expression. Rows and colums are genes. File type: tsv.')
p1.add_argument('net_out', help='Output matrix file of binary co-expression network. Rows and colums are genes. File type: tsv.')
p1.add_argument('qcut',type=float,help='Q-value cutoff for binary network.')


p1=p1s.add_parser('gocovt', help="Introduce principal component of strongest GO pathway as an additional covariates.")
p1.add_argument('exp_in',help='''Input matrix file of normalized expressions. Rows are genes; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('cov_in',help='''Input matrix file of covariates. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')
p1.add_argument('net_in', help='Input matrix file of binary co-expression network. Rows and colums are genes. File type: tsv.')
p1.add_argument('genes_in', help='Input text file of gene names (row names of exp_in), one per line.')
p1.add_argument('go_in', help='Input file of GO DAG. Downloadable from http://geneontology.org/docs/download-ontology/.')
p1.add_argument('goa_in', help='Input file of GO annotation. Downloadable from http://current.geneontology.org/products/pages/downloads.html.')
p1.add_argument('cov_out',help='''Output matrix file of covariates. Rows are covariates; columns are cells. No row or column name allowed. File type: tsv.''')

p1.add_argument('--master_out',dest='master_out',action='store',help='Output text file of master regulator genes, one per line.')
p1.add_argument('--goe_out',dest='goe_out',action='store',help='Output table file of GO enrichments of master regulators. File type: tsv.')
p1.add_argument('--go_out',dest='go_out',action='store',help='Output text file of the GO ID whose top principal component is introduced as a covariate.')
p1.add_argument('-n',action='store',type=int,default='100',help='Number of top master regulators to include for GO enrichment. Defaults to 100.')
p1.add_argument('-m',action='store',type=int,default='5',help='Minimum number of master regulators required for GO enrichment. Defaults to 5.')
p1.add_argument('-c',nargs=3,metavar=('gene_from','gene_to','species'),help='''Gene ID system conversion if needed for GO enrichment analyses. Gene names in current/GO files are in gene_from/gene_to systems respectively. Examples:
From human gene names to GO in uniprot: -c symbol,alias uniprot.Swiss-Prot human
From mouse ensembl IDs to GO in MGI: -c ensembl.gene MGI mouse
For a full list, see https://docs.mygene.info/en/latest/doc/data.html.''')

# p1=p1s.add_parser('cohort_kinshipeigen', help='eigen decomposition of kinship matrix')
# p1.add_argument('-t',dest='tol',action='store',type=float,default='1E-8',help='Tolerance level for low-rank detection. Eigenvalues below tol*maximum eigenvalue are regarded as 0 and disgarded. Default is 1E-8.')

# p1.add_argument('kinship_in', help='Input file for kinship matrix, in reduced form (i.e. between donors, not cells). Row and column are donors. No row or column header allowed. File type: tsv.')
# p1.add_argument('ncell_in', help='Input file for number of cells from each donor. Each entry is the number of cells in the corresponding donor (row or column) in kinship_in file. No header allowed. File type: tsv.')
# p1.add_argument('eigenvalue_out', help='Output file for eigenvalues of kinship matrix. Entry x is the eigenvalue of the eigenvector x in file eigenvector_out. File type: tsv.')
# p1.add_argument('eigenvector_out', help='Output file for eigenvectors of kinship matrix in reduced form. Each row is a reduced eigenvector. A full eigenvector can be obtained from a reduced eigenvector, by sequentially repeating each element at location x by ncell_in[x] times. File type: tsv.')


# p1=p1s.add_parser('cohort_heritability', help="estimate heritability, variance, and covariates' contribution of each gene.",description="Model: transcript=alpha*cov+epsilon, epsilon~N(0,sigma**2*(I+beta*K)).")
# p1.add_argument('-t',dest='tol',action='store',type=float,default='1E-7',help='Numerical precision level for scipy.optimize.minimize. Default is 1E-7.')

# p1.add_argument('transcript_in', help='Input file for transcriptome matrix. Rows are genes/transcripts; columns are cells. No row or column header allowed. Should (mostly) follow a normal distribution for every gene (after removing covariates). Can be logprop_out file from method gen. File type: tsv.')
# p1.add_argument('cov_in', help='''Input file for covariates matrix. Rows are covariates; columns are cells. No row or column header allowed. Can be cov_out file from method normcov. Only full-rank covariates are supported. File type: tsv.''')
# p1.add_argument('ncell_in', help='Input file for number of cells from each donor. Each entry is the number of cells in the corresponding donor (row or column) in kinship_in file. No header allowed. File type: tsv.')
# p1.add_argument('eigenvalue_in', help='Input file for eigenvalues of kinship matrix K. Entry x is the eigenvalue of the eigenvector x in file eigenvector_in. Can be eigenvalue_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('eigenvector_in', help='Input file for eigenvectors of kinship matrix K in reduced form. Each row is a reduced eigenvector. A full eigenvector can be obtained from a reduced eigenvector, by sequentially repeating each element at location x by ncell_in[x] times. Can be eigenvector_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('sigma_out', help='Output file for maximum likelihood estimator of (sqrt) variance of each gene. Each entry is a gene, corresponding to transcript_in. See model above. File type: tsv.')
# p1.add_argument('beta_out', help='Output file for maximum likelihood estimator of (transformed) heritability of each gene. Each entry is a gene, corresponding to transcript_in. See model above. File type: tsv.')
# p1.add_argument('alpha_out', help='Output file for maximum likelihood estimators of covariate contributions of each gene. Rows are genes; columns are covariates. See model above. File type: tsv.')


# p1=p1s.add_parser('cohort_eqtl', help="discover eQTLs on multiple CPU cores.",description="Model: transcript=gamma*genotype+alpha*cov+epsilon, epsilon~N(0,sigma**2*(I+beta*K)). Null hypothesis: gamma=0.")
# p1.add_argument('-m',dest='model',action='store',choices=['add','cat'],default='add',help='Dependency model of transcript on genotype. Additive (add) or categorical (cat). Additive model has 1 degree of freedom (DoF) from each genotype, i.e. the number of variant alleles. Categorical model has n DoFs from each genotype, i.e. the number of variant alleles=1,2,...n, where n is the maximum number of variant alleles for any genotype. Default: add.')
# p1.add_argument('-t',dest='nts',action='store',type=int,default='50',help='Number of transcripts to process in each job (per batch in parallel). Default: 50.')
# p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')
# p1.add_argument('-d',dest='dimr',action='store',type=int,default='0',help='Extra number of effective dimension loss in transcriptome data (in samples/cells) due to preprocessing. Extra dimension loss distorts the null distribution and also the p-value estimation if not adjusted for properly. Default: 0.')

# p1.add_argument('genotype_in', help='Input file for genotype matrix. Rows are genes/transcripts; columns are donors and should match ncell. No row or column header allowed. Can be logprop_out file from method gen. File type: tsv.')
# p1.add_argument('transcript_in', help='Input file for transcriptome matrix. Rows are genes/transcripts; columns are cells. No row or column header allowed. Should (mostly) follow a normal distribution for every gene (after removing covariates). Can be logprop_out file from method gen. File type: tsv.')
# p1.add_argument('cov_in',help='''Input file for covariates matrix. Rows are covariates; columns are cells. No row or column header allowed. Can be cov_out file from method gen. Only full-rank covariates are supported. File type: tsv.''')
# p1.add_argument('ncell_in', help='Input file for number of cells from each donor. Each entry is the number of cells in the corresponding donor (row or column) in kinship_in file. No header allowed. File type: tsv.')
# p1.add_argument('eigenvalue_in', help='Input file for eigenvalues of kinship matrix K. Entry x is the eigenvalue of the eigenvector x in file eigenvector_in. Can be eigenvalue_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('eigenvector_in', help='Input file for eigenvectors of kinship matrix K in reduced form. Each row is a reduced eigenvector. A full eigenvector can be obtained from a reduced eigenvector, by sequentially repeating each element at location x by ncell_in[x] times. Can be eigenvector_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('sigma_in', help='Input file for maximum likelihood estimator of (sqrt) variance of each gene. Each entry is a gene, corresponding to transcript_in. See model above. Can be sigma_out file from method heritability. File type: tsv.')
# p1.add_argument('beta_in', help='Input file for maximum likelihood estimator of (transformed) heritability of each gene. Each entry is a gene, corresponding to transcript_in. See model above. Can be beta_out file from method heritability. File type: tsv.')
# p1.add_argument('eqtlpv_out', help='Output file for p-value matrix of all eQTL associations tested. Rows are genotypes; columns are genes/transcripts. File type: tsv.')
# p1.add_argument('eqtlgamma_out', help='Output file for 2-D-reshaped maximum likelihood estimator 3-D matrix for gamma of all eQTL associations tested. Rows are (genotypes,genotype DoFs) row major. Columns are genes. For genotype DoFs, see definition in argument "-m". This output file should be interpreted with extreme care when the categorical model is used. File type: tsv.')


# p1=p1s.add_parser('cohort_coex', help="compute co-expression p-values on multiple CPU cores.",description="Model: transcript_i-epsilon_i=gamma*(transcript_j-epsilon_j)+alpha*cov, epsilon_k~N(0,sigma_k**2*(I+beta_k*K)). Null hypothesis: gamma=0.")
# p1.add_argument('-t',dest='nts',action='store',type=int,default='50',help='Number of transcripts to process in each job (per batch in parallel). Default: 50.')
# p1.add_argument('-n',dest='nth',action='store',type=int,default='0',help='Number of CPU cores to use. Default: all cores detected.')
# p1.add_argument('-d',dest='dimr',action='store',type=int,default='0',help='Extra number of effective dimension loss in transcriptome data (in samples/cells) due to preprocessing. Extra dimension loss distorts the null distribution and also the p-value estimation if not adjusted for properly. Default: 0.')

# p1.add_argument('transcript_in', help='Input file for transcriptome matrix. Rows are genes/transcripts; columns are cells. No row or column header allowed. Should (mostly) follow a normal distribution for every gene (after removing covariates). Can be logprop_out file from method gen. File type: tsv.')
# p1.add_argument('cov_in',help='''Input file for covariates matrix. Rows are covariates; columns are cells. No row or column header allowed. Can be cov_out file from method gen. Only full-rank covariates are supported. File type: tsv.''')
# p1.add_argument('ncell_in', help='Input file for number of cells from each donor. Each entry is the number of cells in the corresponding donor (row or column) in kinship_in file. No header allowed. File type: tsv.')
# p1.add_argument('eigenvalue_in', help='Input file for eigenvalues of kinship matrix K. Entry x is the eigenvalue of the eigenvector x in file eigenvector_in. Can be eigenvalue_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('eigenvector_in', help='Input file for eigenvectors of kinship matrix K in reduced form. Each row is a reduced eigenvector. A full eigenvector can be obtained from a reduced eigenvector, by sequentially repeating each element at location x by ncell_in[x] times. Can be eigenvector_out file from method kinship_eigen. File type: tsv.')
# p1.add_argument('beta_in', help='Input file for maximum likelihood estimator of (transformed) heritability of each gene. Each entry is a gene, corresponding to transcript_in. See model above. Can be beta_out file from method heritability. File type: tsv.')
# p1.add_argument('coexpv_out', help='Output file for p-value matrix of all co-expressions tested. Row and column are genes/transcripts. File type: tsv.')

if len(sys.argv)==1:
	p0.print_help(sys.stderr)
	sys.exit(1)

args=p0.parse_args()
args=vars(args)
logging.basicConfig(format='%(levelname)s:%(process)d:%(asctime)s:%(pathname)s:%(lineno)d:%(message)s',level=logging.DEBUG if args['verbose'] else logging.WARNING)
from . import run
func=getattr(run,args['cmd'])
func(args)

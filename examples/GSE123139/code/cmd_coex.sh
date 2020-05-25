#!/bin/bash

set -x -e -o pipefail

#For more verbosity, use verbose="-v"
verbose=""
#Gene name system as provided. Details see normalisr gocovt -h.
genefrom="symbol,alias"
#Gene name system of GO annotations. Details see normalisr gocovt -h.
geneto="uniprot.Swiss-Prot"
#Species for gene name conversion.
species="human"
#Maximum number of iterations of top GO removal
lvmax=5
#Q-value cutoff to binarize P-value network. Choose this cutoff by trying out steps 10 and onwards for the desired network sparsity.
qcut="1E-8"

#Step 1: Find low-read genes and cells for removal
normalisr "$verbose" qc_reads -s --gene_cell_count 500 data/coex/0_read.mtx.gz data/coex/0_gene.txt data/coex/0_cell.txt data/coex/1_gene.txt data/coex/1_cell.txt
#Step 2: Remove found low-read genes and cells, and single-valued covariates
normalisr "$verbose" subset -s -r data/coex/0_gene.txt data/coex/1_gene.txt -c data/coex/0_cell.txt data/coex/1_cell.txt data/coex/0_read.mtx.gz data/coex/2_read.tsv.gz
normalisr "$verbose" subset --nodummy -c data/coex/0_cell.txt data/coex/1_cell.txt data/coex/0_cov.tsv.gz data/coex/2_cov.tsv.gz
#Step 3: Convert read counts to Bayesian logCPM with cellular summary covariates
normalisr "$verbose" lcpm -c data/coex/2_cov.tsv.gz data/coex/2_read.tsv.gz data/coex/3_lcpm.tsv.gz data/coex/3_scale.tsv.gz data/coex/3_cov.tsv.gz
#Step 4: Normalize covariates to zero mean and unit variance
normalisr "$verbose" normcov data/coex/3_cov.tsv.gz data/coex/4_cov.tsv.gz
#Step 5: Log linear fit of cell variance with covariates
normalisr "$verbose" fitvar data/coex/3_lcpm.tsv.gz data/coex/4_cov.tsv.gz data/coex/5_weight.tsv.gz
#Step 6: Detect cell outliers by low fitted variance
normalisr "$verbose" qc_outlier data/coex/5_weight.tsv.gz data/coex/1_cell.txt data/coex/6_cell.txt
#Step 7: Remove found outlier cells
normalisr "$verbose" subset -c data/coex/1_cell.txt data/coex/6_cell.txt data/coex/3_lcpm.tsv.gz data/coex/7_lcpm.tsv.gz
normalisr "$verbose" subset -c data/coex/1_cell.txt data/coex/6_cell.txt data/coex/4_cov.tsv.gz data/coex/7_cov.tsv.gz
normalisr "$verbose" subset -c data/coex/1_cell.txt data/coex/6_cell.txt data/coex/5_weight.tsv.gz data/coex/7_weight.tsv.gz
#Step 8: Normalize expression and covariates
normalisr "$verbose" normvar data/coex/7_lcpm.tsv.gz data/coex/7_weight.tsv.gz data/coex/7_cov.tsv.gz data/coex/3_scale.tsv.gz data/coex/8_exp.tsv.gz data/coex/lv0_cov.tsv.gz
lv=0
while [ $lv -le $lvmax ]; do
	#Per-level step 1: Compute co-expression
	normalisr "$verbose" coex --var_out data/coex/lv"$lv"_var.tsv.gz --dot_out data/coex/lv"$lv"_dot.tsv.gz data/coex/8_exp.tsv.gz data/coex/lv"$lv"_cov.tsv.gz data/coex/lv"$lv"_pv.tsv.gz
	#Per-level step 2: Convert P-value co-expression network into binary network. 
	normalisr "$verbose" binnet data/coex/lv"$lv"_pv.tsv.gz data/coex/lv"$lv"_net.tsv.gz $qcut
	#Per-level step 3: GO enrichment of top master regulators and preparation of next level GO removal
	normalisr "$verbose" gocovt --master_out data/coex/lv"$lv"_master.txt --goe_out data/coex/lv"$lv"_goe.tsv.gz --go_out data/coex/lv"$lv"_go.txt -c $genefrom $geneto $species data/coex/8_exp.tsv.gz data/coex/lv"$lv"_cov.tsv.gz data/coex/lv"$lv"_net.tsv.gz data/coex/1_gene.txt data/go/go-basic.obo data/go/goa-human.gaf data/coex/lv"$(( lv + 1 ))"_cov.tsv.gz
	lv=$(( lv + 1 ))
done

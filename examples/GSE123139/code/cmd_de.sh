#!/bin/bash

set -x -e -o pipefail

#Step 1: Find low-read genes and cells for removal
normalisr qc_reads -s --gene_cell_count 500 data/de/0_read.mtx.gz data/de/0_gene.txt data/de/0_cell.txt data/de/1_gene.txt data/de/1_cell.txt
#Step 2: Remove found low-read genes and cells, and single-valued covariates
normalisr subset -s -r data/de/0_gene.txt data/de/1_gene.txt -c data/de/0_cell.txt data/de/1_cell.txt data/de/0_read.mtx.gz data/de/2_read.tsv.gz
normalisr subset --nodummy -c data/de/0_cell.txt data/de/1_cell.txt data/de/0_cov.tsv.gz data/de/2_cov.tsv.gz
normalisr subset -c data/de/0_cell.txt data/de/1_cell.txt data/de/0_group.tsv.gz data/de/2_group.tsv.gz
#Step 3: Convert read counts to Bayesian logCPM with cellular summary covariates
normalisr lcpm -c data/de/2_cov.tsv.gz data/de/2_read.tsv.gz data/de/3_lcpm.tsv.gz data/de/3_scale.tsv.gz data/de/3_cov.tsv.gz
#Step 4: Normalize covariates to zero mean and unit variance
normalisr normcov data/de/3_cov.tsv.gz data/de/4_cov.tsv.gz
#Step 5: Log linear fit of cell variance with covariates
normalisr fitvar data/de/3_lcpm.tsv.gz data/de/4_cov.tsv.gz data/de/5_weight.tsv.gz
#Step 6: Detect cell outliers by low fitted variance
normalisr qc_outlier data/de/5_weight.tsv.gz data/de/1_cell.txt data/de/6_cell.txt
#Step 7: Remove found outlier cells
normalisr subset -c data/de/1_cell.txt data/de/6_cell.txt data/de/3_lcpm.tsv.gz data/de/7_lcpm.tsv.gz
normalisr subset -c data/de/1_cell.txt data/de/6_cell.txt data/de/4_cov.tsv.gz data/de/7_cov.tsv.gz
normalisr subset -c data/de/1_cell.txt data/de/6_cell.txt data/de/5_weight.tsv.gz data/de/7_weight.tsv.gz
normalisr subset -c data/de/1_cell.txt data/de/6_cell.txt data/de/2_group.tsv.gz data/de/7_group.tsv.gz
#Step 8: Normalize expression and covariates
normalisr normvar data/de/7_lcpm.tsv.gz data/de/7_weight.tsv.gz data/de/7_cov.tsv.gz data/de/3_scale.tsv.gz data/de/8_exp.tsv.gz data/de/8_cov.tsv.gz
#Step 9: Compute differential expression
normalisr de data/de/7_group.tsv.gz data/de/8_exp.tsv.gz data/de/8_cov.tsv.gz data/de/9_pv.tsv.gz data/de/9_lfc.tsv.gz

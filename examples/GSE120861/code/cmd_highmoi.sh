#!/bin/bash

set -x -e -o pipefail

#Step 1: Find low-read genes and cells for removal
normalisr -v qc_reads -s data/highmoi/0_read.mtx.gz data/highmoi/0_gene.txt data/highmoi/0_cell.txt data/highmoi/1_gene.txt data/highmoi/1_cell.txt
#Step 2: Remove found low-read genes and cells, and single-valued covariates
normalisr -v subset -s -r data/highmoi/0_gene.txt data/highmoi/1_gene.txt -c data/highmoi/0_cell.txt data/highmoi/1_cell.txt data/highmoi/0_read.mtx.gz data/highmoi/2_read.tsv.gz
normalisr -v subset --nodummy -c data/highmoi/0_cell.txt data/highmoi/1_cell.txt data/highmoi/0_cov.tsv.gz data/highmoi/2_cov.tsv.gz
normalisr -v subset -c data/highmoi/0_cell.txt data/highmoi/1_cell.txt data/highmoi/0_group.tsv.gz data/highmoi/2_group.tsv.gz
#Step 3: Convert read counts to Bayesian logCPM with cellular summary covariates
normalisr -v lcpm -c data/highmoi/2_cov.tsv.gz data/highmoi/2_read.tsv.gz data/highmoi/3_lcpm.tsv.gz data/highmoi/3_scale.tsv.gz data/highmoi/3_cov.tsv.gz
#Step 4: Normalize covariates to zero mean and unit variance
normalisr -v normcov data/highmoi/3_cov.tsv.gz data/highmoi/4_cov.tsv.gz
#Step 5: Log linear fit of cell variance with covariates
normalisr -v fitvar data/highmoi/3_lcpm.tsv.gz data/highmoi/4_cov.tsv.gz data/highmoi/5_weight.tsv.gz
#Step 6: Normalize expression and covariates
normalisr -v normvar data/highmoi/3_lcpm.tsv.gz data/highmoi/5_weight.tsv.gz data/highmoi/4_cov.tsv.gz data/highmoi/3_scale.tsv.gz data/highmoi/6_exp.tsv.gz data/highmoi/6_cov.tsv.gz
#Step 7: Compute competition-naive differential expression
normalisr -v de data/highmoi/2_group.tsv.gz data/highmoi/6_exp.tsv.gz data/highmoi/6_cov.tsv.gz data/highmoi/7_pv.tsv.gz data/highmoi/7_lfc.tsv.gz
#Step 8: Compute competition-aware differential expression
normalisr -v de -m covariate data/highmoi/2_group.tsv.gz data/highmoi/6_exp.tsv.gz data/highmoi/6_cov.tsv.gz data/highmoi/8_pv.tsv.gz data/highmoi/8_lfc.tsv.gz

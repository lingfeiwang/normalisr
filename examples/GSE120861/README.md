# Example GSE120861

High-MOI CRISPRi CROP-seq pilot dataset for enhancer and gene-regulation screens. Contains competition-naive and competition-aware methods for differential expression.

## Usage
### Data preparation
1. Enter GSE120861 as the working directory.
2. Run ./code/prepare.sh to download and convert dataset into Normalisr's tsv format. See prepared inputs in data/highmoi.

### Option 1: analyses at command line
1. Read and run ./code/cmd_highmoi.sh to see each step. Final outputs are:
	* data/highmoi/X_pv.tsv.gz: differential expression matrix for P-values;
	* data/highmoi/X_lfc.tsv.gz: differential expression matrix for logFCs;
	* data/highmoi/1_gene.txt: gene names as columns of above files;
	* data/highmoi/0_gRNA.txt: gRNA names as rows of above files;
	* X=7 for competition-naive method, and 8 for competition-aware method.
2. See simple visualizations of the output in folder ipynb.

### Option 2: analyses with python
Read and run jupyter notebook at ./code/notebook_highmoi.ipynb

## Next
1. Redo the analyses with full dataset. This example uses 15 non-targeting and 15 TSS-targeting gRNAs and around 7,000 cells to work on 16GB of memory. Start from a clean example folder. Change variable ng_negselect, ng_tssselect, ng_other, droprate_ng, and droprate_g in code/prepare_highmoi.py. Then rerun the full example again.
2. Reformat your own dataset and run Normalisr, or try the full-scale dataset also in GSE120861!

## Data references
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120861
* Gasperini et al, Cell 2019, https://www.cell.com/cell/fulltext/S0092-8674(18)31554-X

# Example GSE123139
MARS-seq of T cells from human melanoma. This example contains two sub-examples of differential expression between dysfunctional and naive T cells, and transcriptome-wide co-expression in dysfunctional T cells with automatic GO covariate removal.

## Usage
### Data preparation
1. Enter GSE123139 as the working directory.
2. Run ./code/prepare.sh to download and convert the dataset into Normalisr's tsv format. See prepared inputs in data/de and data/coex.

### Option 1: analyses at command line
1. For differential expression, read and run ./code/cmd_de.sh to see each step. Final outputs are:
	* data/de/9_pv.tsv.gz: differential expression P-values;
	* data/de/9_lfc.tsv.gz: differential expression logFCs;
	* data/de/1_gene.txt: gene names as columns of above files.
2. For co-expression, edit, read, and run ./code/cmd_coex.sh to see each step. Final outputs are:
	* data/coex/lvX_pv.tsv.gz: co-expression P-values;
	* data/coex/lvX_net.tsv.gz: binary co-expression network;
	* data/coex/lvX_var.tsv.gz: gene variances;
	* data/coex/lvX_dot.tsv.gz: gene-pair inner products variances ( Pearson R = (dot/sqrt(var)).T/sqrt(var) );
	* data/coex/lvX_master.txt: list of master regulators;
	* data/coex/lvX_goe.tsv.gz: table of gene ontology enrichment of master regulators;
	* data/coex/1_gene.txt: gene names as rows and columns of co-expression result matrices;
	* X in lvX indicates the number of GO pathway covariates removed, for a more cell-type-specific co-expression network.
3. See simple visualizations of the output in folder ipynb.

### Option 2: analyses with python
1. For differential expression, read and run jupyter notebook at ./code/notebook_de.ipynb.
2. For co-expression, edit, read, and run jupyter notebook at ./code/notebook_coex.ipynb.

## Next
1. Redo the analyses with full dataset. This example uses around half of the cells to work on 16GB of memory. For the full dataset, start from a clean example folder. Change variable drop_rate in code/prepare_raw.py and qcut in code/cmd_coex.sh. Then rerun the full example again.
2. Reformat your own dataset and run Normalisr!

## Data references
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123139
* Li et al, Cell 2018, https://www.cell.com/cell/fulltext/S0092-8674(18)31568-X

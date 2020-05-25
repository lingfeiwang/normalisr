# Normalisr
Normalisr is a parameter-free normalization-association two-step inferential framework for scRNA-seq that solves case-control DE, co-expression, and pooled CRISPRi scRNA-seq screen under one umbrella of linear association testing. Normalisr addresses sparsity and technical confounding challenges of scRNA-seq with posterior mRNA abundances, nonlinear cellular summary covariates, and mean and variance normalization. All these enable linear association testing to achieve optimal sensitivity, specificity, and speed in all above scenarios.

Normalisr follows the conventional framework of normalization/imputation of scRNA-seq, and aims to recover the true, biological, but hidden expression levels which any analyses may then operate upon. Then, linear association testing provides a unified inferential framework with numerous advantages: (i) exact P-value estimation without permutation, (ii) native removal of covariates as fixed effects, (iii) non-parametric robustness, (iv) unbeatable time and memory complexities, and (v) extension potentials such as variations in genetic relatedness.

Normalisr is written in Python3 and provides a command-line and a python functional interface. You can read more about Normalisr from our preprint (See [Citations](#citations)).

## Install
Normalisr is on PyPI (link TBA) and can be installed straightaway with pip (TB-uploaded): `pip install normalisr`. You can also install Normalisr from github (link TBD): `pip install`. Make sure you have added Normalisr's install path into PATH environment before using the command-line interface (See [FAQ](#faq)).

There are more advanced installation methods but if you want that, most likely you already know how to do it ;). If not, give me a shout (See [Contact](#contact)).

## Usage
Normalisr provides a command-line and a python functional interface below. You can use the examples provided below to guide yourself through Normalisr's use.

### Commmand-line interface
You can run Normalisr by typing `normalisr` on command-line. Normalisr uses submodules for different analysis steps. Type `normalisr` or `normalisr -h` for general help, and for example `normalisr de -h` for help on submodule 'de' of differential expression.

Normalisr uses tsv (tab separated values) file format for input and output matrices, and text file for row and column names, such as cells and genes, one per line. For initial input, Normalisr also accepts the sparse mtx format (Cell Ranger output) for raw read count matrix. Gzipped input/output files are automatically recognized if file name suffix '.gz' is present.

### Python functional interface
Normalisr's python functional interface are more flexible than command-line, but requires knowledge of python programming. You can use command autocomplete to find the functions available in Normalisr. Then, documentation of any function can be obtained with `?` in ipython or jupyter notebook, such as:
```
import normalisr as norm
?norm.de.de
```
The example jupyter notebooks can also illustrate the scope of functions Normalisr provides.

## Examples and pipelines
You can find several examples in the 'examples' folder, to cover all functions Normalisr currently provides. The example datasets have been scaled down to run on a 16GB-memory personal computer. Although they only serve as demonstrations of work here, the pipelines should be transferable to a full-scale, different dataset. Since Normalisr is non-parametric, the only adjustable parameters are for quality control and final cutoffs of differential or co-expression. You can change down-sampling parameters in the examples to run the full datasets on a larger computer.

You can find more details in the respective examples.

## Contact
We look forward to your feedbacks or questions of any kind.
* Regarding method and manuscript, please reach me by email (Lingfei.Wang.CN@gmail.com).
* Regarding Normalisr package, please raise an issue on github (https://github.com/lingfeiwang/normalisr/issues/new) or reach me by email (Lingfei.Wang.github@outlook.com).

## Citations
Please cite our preprint if you use Normalisr:
* TBA

## FAQ
* What does Normalisr stand for?

**N**ormalisr **O**ffers **R**obust **M**odelling of **A**ssociations **L**inearly **I**n **S**ingle-cell **R**NA-seq. Yes, it's a recursive sentence. See [GNU](https://en.wikipedia.org/wiki/GNU).

* I installed Normalisr but typing `normalisr` says 'command not found'.
* How do I use a specific python version for Normalisr's command-line interface?

You can always use the python command to run Normalisr, such as `python3 -m normalisr` to replace command `normalisr`. You can also use a specific path or version for python, such as `python3.7 -m normalisr` or `/usr/bin/python3.7 -m normalisr`. Make sure you have installed Normalisr for this python version.

* Why don't the examples work?

Please make sure you followed every step in the README.md of the respective example folder with Internet connection, and then submit an issue report detailing at which executed line the error occurred with input and output.

* Does Normalisr run on Windows?

I have not tested Normalisr on Windows. However, it is purely in python and should be able to function properly.

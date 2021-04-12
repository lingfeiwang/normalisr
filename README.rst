=========
Normalisr
=========
.. image:: https://img.shields.io/pypi/v/normalisr?color=informational
   :target: https://pypi.python.org/pypi/normalisr

.. image:: https://zenodo.org/badge/242889849.svg
   :target: https://zenodo.org/badge/latestdoi/242889849


Normalisr is a parameter-free normalization-association two-step inferential framework for scRNA-seq that solves case-control differential expression, co-expression, and pooled CRISPR scRNA-seq screen analysis with linear association testing. By systematically detecting and removing nonlinear confounding from library size, Normalisr achieves high sensitivity, specificity, speed, and generalizability across multiple scRNA-seq protocols and experimental conditions with unbiased P-value estimation.

Normalisr first removes confounding technical noises from raw read counts to recover the biological variations. Then, linear association testing provides a unified inferential framework with several advantages: (i) exact P-value estimation without permutation, (ii) native removal of covariates (*e.g.* batches, house-keeping programs, and untested gRNAs) as fixed effects, and (iii) computational efficiency.

Normalisr is written in Python3 and provides a command-line and a python functional interface. You can read more about Normalisr from our preprint (See References_).

Installation
=============
Normalisr is on `PyPI <https://pypi.org/project/normalisr>`_ and can be installed with pip: ``pip install normalisr``. You can also install Normalisr from github: ``pip install git+https://github.com/lingfeiwang/normalisr.git``. Make sure you have added Normalisr's install path into PATH environment before using the command-line interface (See FAQ_). Normalisr's installation should take less than a minute.

There are more advanced installation methods but if you want that, most likely you already know how to do it. If not, give me a shout (See Contact_).

Usage
=====
Normalisr provides a command-line and a python functional interface below. You can use the examples provided below to guide yourself through Normalisr's use. Sphinx-based documentation is underway.

* Commmand-line interface
	You can run Normalisr by typing ``normalisr`` on command-line. Normalisr uses submodules for different analysis steps. Type ``normalisr`` or ``normalisr -h`` for general help, and for example ``normalisr de -h`` for help on submodule 'de' of differential expression.

	Normalisr uses tsv (tab separated values) file format for input and output matrices, and text file for row and column names, such as cells and genes, one per line. For initial input, Normalisr also accepts the sparse mtx format (Cell Ranger output) for raw read count matrix. Gzipped input/output files are automatically recognized if file name suffix '.gz' is present.

* Python functional interface
	Normalisr's python functional interface is more flexible than command-line, but requires knowledge of python programming. Documentation of any function can be obtained with ``?`` in ipython or jupyter notebook, such as:

	.. code-block::

		import normalisr.normalisr as norm
		?norm.de

	The example jupyter notebooks also illustrate the scope of functions Normalisr provides.


Documentation
=============
Documentations are available as `html <https://lingfeiwang.github.io/normalisr/index.html>`_ and `pdf <https://github.com/lingfeiwang/normalisr/raw/master/docs/build/latex/normalisr.pdf>`_.

Examples and pipelines
==========================
You can find several examples in the 'examples' folder, to cover all functions Normalisr currently provides. The example datasets have been scaled down to run on a 16GB-memory personal computer. Although they only serve as demonstrations of work here, the pipelines should be transferable to a full-scale, different dataset. Since Normalisr is non-parametric, the only adjustable parameters are for quality control and final cutoffs of differential or co-expression. You can change down-sampling parameters in the examples to run the full datasets on a larger computer.

You can find more details in the respective examples.

Contact
==========================
Pease raise an issue on `github <https://github.com/lingfeiwang/normalisr/issues/new>`_ or reach me by e-mail (Lingfei.Wang.github@outlook.com or contact on the manuscript).

References
==========================
TBA

FAQ
==========================
* What does Normalisr stand for?
	**N**\ ormalisr **O**\ ffers **R**\ obust **M**\ odelling of **A**\ ssociations **L**\ inearly **I**\ n **S**\ ingle-cell **R**\ NA-seq. Yes, it's a recursive acronym. See `GNU <https://www.gnu.org/gnu/gnu-history.en.html>`_ and `pip <http://www.ianbicking.org/blog/2008/10/28/pyinstall-is-dead-long-live-pip/index.html>`_.

* I installed Normalisr but typing ``normalisr`` says 'command not found'.
	See below.
	
* How do I use a specific python version for Normalisr's command-line interface?
	You can always use the python command to run Normalisr, such as ``python3 -m normalisr`` to replace command ``normalisr``. You can also use a specific path or version for python, such as ``python3.7 -m normalisr`` or ``/usr/bin/python3.7 -m normalisr``. Make sure you have installed Normalisr for this python version.


* Why don't the examples work?
	Please make sure you followed every step in the README.md of the respective example folder with Internet connection, and then submit an issue report detailing at which executed line the error occurred with input and output.


* Does Normalisr run on Windows?
	I have not tested Normalisr on Windows. However, it is purely in python and should be able to function properly.

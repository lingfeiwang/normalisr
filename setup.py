#!/usr/bin/python3

pkgname="normalisr"
pkgnamefull="Normalisr Offers Robust Modelling of Associations Linearly In Single-cell RNA-seq"
version=[0,4,0]
license="BSD-3-Clause"
url="https://github.com/lingfeiwang/"+pkgname
author="Lingfei Wang"
author_email="Lingfei.Wang.github@outlook.com"


def pkg_setup():
	from setuptools import setup
	from os.path import join as pjoin
	pkgnameu=pkgname[0].upper()+pkgname[1:].lower()
	setup(name=pkgname,
		version='.'.join(map(str,version)),
		author=author,
		author_email=author_email,
		description=pkgnamefull,
		long_description='{0} is a parameter-free normalization-association two-step inferential framework for scRNA-seq that solves case-control DE, co-expression, and pooled CRISPRi scRNA-seq screen under one umbrella of linear association testing. {0} addresses sparsity and technical confounding challenges of scRNA-seq with posterior mRNA abundances, nonlinear cellular summary covariates, and mean and variance normalization. All these enable linear association testing to achieve optimal sensitivity, specificity, and speed in all above scenarios.'.format(pkgnameu),
		url=url,
		download_url=url,
		scripts=['bin/normalisr'],
		# include_package_data=True,
		install_requires=['numpy','scipy','argparse','pandas','sklearn','mygene','goatools'],
		classifiers=['Development Status :: 3 - Alpha',
			'License :: OSI Approved :: BSD License',
			'Environment :: Console',
			'Framework :: Pytest',
			'Intended Audience :: Science/Research',
			'Intended Audience :: Developers',
			'Operating System :: OS Independent',
			'Programming Language :: Python :: 3',
			'Topic :: Scientific/Engineering :: Bio-Informatics'],
		license=license,
		packages=[pkgname],
		package_dir={pkgname:pjoin('src',pkgname)},
	)

pkg_setup()

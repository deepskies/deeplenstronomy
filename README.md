# Welcome to `deeplenstronomy`!

[![status](https://joss.theoj.org/papers/e978dd566d1f290055a02d76288e95e1/status.svg)](https://joss.theoj.org/papers/e978dd566d1f290055a02d76288e95e1)
[![status](https://img.shields.io/badge/arXiv-2102.02830-red)](http://arxiv.org/abs/2102.02830)
[![status](https://img.shields.io/badge/PyPi-0.0.2.3-blue)](https://pypi.org/project/deeplenstronomy/)
[![status](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/deepskies/deeplenstronomy/blob/master/LICENSE)

`deeplenstronomy` is a tool for simulating large datasets for applying deep learning to strong gravitational lensing. 
It works by wrapping the functionalities of [`lenstronomy`](https://github.com/sibirrer/lenstronomy) in a convenient yaml-style interface, allowing users to embrace the astronomer part of their brain rather than their programmer part when generating training datasets.

## Installation

**With conda (Recommended)**

- Step 0: Set up an environment. This can be done straightforwardly with a `conda` installation:

```
conda create -n deeplens python=3.7 jupyter scipy pandas numpy matplotlib astropy h5py PyYAML mpmath future
conda activate deeplens
```

- Step 1: `pip install lenstronomy`
- Step 2: `pip install deeplenstronomy`

**With pip**

- Step 1: `pip install deeplenstronomy`

## [Getting Started and Example Notebooks](https://deepskies.github.io/deeplenstronomy/Notebooks/)

Start by reading the [Getting Started Guide](https://deepskies.github.io/deeplenstronomy/Notebooks/GettingStarted.html) to familiarize yourself with the `deeplenstronomy` style.

After that, check out the example notebooks below:

### Notebooks for `deeplenstronomy` Utilities
- [Creating `deeplenstronomy` Configuration Files](https://deepskies.github.io/deeplenstronomy/Notebooks/ConfigFiles.html)
- [Generating Datasets](https://deepskies.github.io/deeplenstronomy/Notebooks/DeepLenstronomyDemo.html)
- [Visualizing `deeplenstronomy` Images](https://deepskies.github.io/deeplenstronomy/Notebooks/Visualization.html)
- [Utilizing Astronomical Surveys](https://deepskies.github.io/deeplenstronomy/Notebooks/Surveys.html)
- [Defining Your Own Probability Distributions](https://deepskies.github.io/deeplenstronomy/Notebooks/UserDistributions.html)
- [Using Your Own Images as Backgrounds](https://deepskies.github.io/deeplenstronomy/Notebooks/BackgroundsDemo.html)
- [Simulating Time-Series Datasets](https://deepskies.github.io/deeplenstronomy/Notebooks/TimeSeriesDemo.html)

### Notebooks for Applying `deeplenstronomy` to Machine Learning Analyses
- [Using `deeplenstronomy` for Active Learning](https://deepskies.github.io/deeplenstronomy/Notebooks/ActiveUpdateDemo.html)
- [Using `deeplenstronomy` for Classification and Regression](https://deepskies.github.io/deeplenstronomy/Notebooks/Metrics.html)

### Notebooks for Suggested Science Cases
- [A Walkthrough of Using `deeplenstronomy` for Science](https://deepskies.github.io/deeplenstronomy/Notebooks/FullExample.html)


## API Documentation

`deeplenstronomy` is designed so that users only need to work with their personal configuration files and the dataset generatation and image visualization functions.
However, if you would like to view the full API documentation, you can visit the [docs](https://deepskies.github.io/deeplenstronomy/docs/) page.

## Citation

If you use `deeplenstronomy` in your work, please include the following citations:
```
@article{deeplenstronomy,
  doi = {10.21105/joss.02854},
  url = {https://doi.org/10.21105/joss.02854},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {58},
  pages = {2854},
  author = {Robert Morgan and Brian Nord and Simon Birrer and Joshua Yao-Yu Lin and Jason Poh},
  title = {deeplenstronomy: A dataset simulation package for strong gravitational lensing},
  journal = {Journal of Open Source Software}
}

@article{lenstronomy,
    title     =   "lenstronomy: Multi-purpose gravitational lens modelling software package",
    journal   =   "Physics of the Dark Universe",
    volume    =   "22",
    pages     =   "189 - 201",
    year      =   "2018",
    issn      =   "2212-6864",
    doi       =   "10.1016/j.dark.2018.11.002",
    url       =   "http://www.sciencedirect.com/science/article/pii/S2212686418301869",
    author    =   "Simon Birrer and Adam Amara",
    keywords  =   "Gravitational lensing, Software, Image simulations"
}
```

## Contact

If you have any questions or run into any errors with the beta release of `deeplenstronomy`, please don't hesitate to reach out:

Rob Morgan 
<br>
robert [dot] morgan [at] wisc.edu

You can also message me on the DES, DELVE, LSSTC, deepskies, or lenstronomers Slack workspaces





<!---
.. image:: https://badge.fury.io/py/deeplenstronomy.png
    :target: http://badge.fury.io/py/deeplenstronomy

.. image:: https://travis-ci.org/bnord/deeplenstronomy.png?branch=master
    :target: https://travis-ci.org/bnord/deeplenstronomy
--->




# Welcome to `deeplenstronomy`!

[![status](https://joss.theoj.org/papers/e978dd566d1f290055a02d76288e95e1/status.svg)](https://joss.theoj.org/papers/e978dd566d1f290055a02d76288e95e1)

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

## Documentation

Start by reading the [Getting Started Guide](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/GettingStarted.ipynb) to familiarize yourself with the `deeplenstronomy` style.

After that, check out the example notebooks below:

### Notebooks for `deeplenstronomy` Utilities
- [Creating `deeplenstronomy` Configuration Files](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/ConfigFiles.md)
- [Generating Datasets](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/DeepLenstronomyDemo.ipynb)
- [Visualizing `deeplenstronomy` Images](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/Visualization.ipynb)
- [Utilizing Astronomical Surveys](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/Surveys.ipynb)
- [Defining Your Own Probability Distributions](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/UserDistributions.ipynb)
- [Using Your Own Images as Backgrounds](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/BackgroundsDemo.ipynb)
- [Simulating Time-Series Datasets](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/TimeSeriesDemo.ipynb)

### Notebooks for Applying `deeplenstronomy` to Machine Learning Analyses
- [Using `deeplenstronomy` for Active Learning](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/ActiveUpdateDemo.ipynb)
- [Using `deeplenstronomy` for Classification and Regression](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/Metrics.ipynb)

### Notebooks for Suggested Science Cases
- [A Walkthrough of Using `deeplenstronomy` for Science](https://github.com/deepskies/deeplenstronomy/blob/master/Notebooks/FullExample.ipynb)

## Citation

If you use `deeplenstronomy` in your work, please include the following citations:
```
@online{deeplenstronomy,
    author    =   "Robert Morgan and Brian Nord and Simon Birrer and Joshua Yao-Yu Lin and Jason Poh",
    title     =   "deeplenstronomy: A data set simualtion package for strong gravitational lensing",
    year      =   "2020",
    url       =   "https://github.com/deepskies/deeplenstronomy",
    urldate   =   "2020-12-14"
    keywords  =   "Python, astronomy, strong lensing, simulation"
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




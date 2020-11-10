---
title: 'deeplenstronomy: A dataset simulation package for strong gravitational lensing'
tags:
  - Python
  - astronomy
  - strong lensing
  - simulation
authors:
  - name: Robert Morgan^[Corresponding author email: robert.morgan@wisc.edu]
    orcid: 0000-0002-7016-5471
    affiliation: "1, 2" 
  - name: Brian Nord
    orcid: 0000-0001-6706-8972
    affiliation: "3, 4"
  - name: Simon Birrer
    orcid: 0000-0003-3195-5507
    affiliation: 5
  - name: Joshua Yao-Yu Lin
    orcid: 0000-0003-0680-4838
    affiliation: 6
  - name: Jason Poh
    orcid: 0000-0002-5040-093X
    affiliation: 4
affiliations:
  - name: University of Wisconsin-Madison
    index: 1
  - name: Legacy Survey of Space and Time Data Science Fellowship Program
    index: 2
  - name: Fermi National Accelerator Laboratory
    index: 3
  - name: University of Chicago
    index: 4
  - name: Stanford University
    index: 5
  - name: University of Illinois Urbana-Champaign
    index: 6
date: 28 October 2020
bibliography: paper.bib
---

# Background

Astronomical observations and statistical modeling permit the high-fidelity analysis of strong gravitational lensing (SL) systems, which display an astronomical phenomenon in which light from a distant object is deflected by the gravitational field of another object along its path to the observer.
These systems are of great scientific interest, because they provide information about multiple astrophysical and cosmological phenomena, including the nature of dark matter, the expansion rate of the Universe, and characteristics of galaxy populations. 
They also provided critical evidence during tests of the theory of General Relativity. 

Traditional searches have involved time- and effort-intensive visual or manual inspection of images by humans to identify characteristic features  --- like arcs, particular color combinations, and  object orientations ---  of SL systems. 
However, a comprehensive search using the traditional approach is prohibitively expensive for large numbers of images, like those in cosmological surveys --- e.g., SDSS [@2000AJ....120.1579Y], DES [@2018PhRvD..98d3526A], and LSST [@2019ApJ...873..111I]. 
To automate the SL detection process, techniques based in machine learning (ML) are beginning to overtake traditional approaches for scanning  astronomical images. 
In particular, deep learning techniques, like two-dimensional convolutional neural networks have been the focus, but they require large sets of labeled images to train these models. 
Because of the relatively low number of observed SL systems, simulated datasets of images are often needed. 
Thus, the composition and production of these simulated datasets has become an integral part of the field.

One of the premier tools for simulating and analyzing SL systems, `lenstronomy` [@lenstronomy], works by the user specifying the properties of the physical systems, as well as how they’re observed (e.g., telescope and camera) through a `python`-based application programming interface (API) to generate a single image. 
To generate populations of SL systems that are fit for neural network training requires additional infrastructure. 

# Statement of need 

Due to the inherent dependence of the performance of ML approaches on the training data, the 2D-CNN approach to SL detection is in tension with scientific reproducibility without a clear prescription for the simulation of the training data. 
There is a critical need for a tool that simulates full datasets in an efficient and reproducible manner, while enabling the use of all the features of the `lenstronomy` simulation API. 
Additionally, this tool should  simplify user interaction with `lenstronomy` and organize the simulations and associated metadata into convenient data structures for deep learning problems.


# Summary

`deeplenstronomy` generates SL data sets by organizing and expediting user interaction with `lenstronomy`. 
The user creates a single yaml-style configuration file that describes the aspects of the dataset: number of images, properties of the telescope and camera, cosmological parameters, observing conditions, properties of the physical objects, and geometry of the SL systems. 
`deeplenstronomy` parses the configuration file and generates the dataset, producing both the images and the parameters that led to the production of each image as outputs. 
The configuration files can easily be shared, enabling users to easily reproduce each other's training datasets.

`deeplenstronomy` also contains built-in features to help astronomers make their training datasets as realistic as possible. 
These features include the following functionalities: use any stellar light profile or mass profile in `lenstronomy`; simulate a variety of astronomical systems such as single galaxies, foreground stars, galaxy clusters, supernovae, and kilonovae, as well as any combination of those systems; fully control the placements of objects in the simulations; use observing conditions of real astronomical surveys; draw any parameter from any probability distribution; introduce any correlation; inspect and visualize the simulation outputs; incorporate real images into the simulation; and simulate time series data. 
These features and the path from configuration file to full data set are shown in \autoref{fig:flowchart}.

![The `deeplenstronomy` process. Data set properties, camera and telescope properties, observing conditions, object properties (e.g., `lenstronomy` light and mass profiles, point sources, and temporal behavior), the geometry of the SL systems are specified in the main configurations file, and any optional supplemental input files (e.g., probability distributions, covariance matrices, and image backgrounds). `deeplenstronomy` then intreprets the configuration file, calls `lenstronomy` simulation functionalities, and organizes the resulting images and metadata.\label{fig:flowchart}](flowchart.png)


`deeplenstronomy` makes use of multiple open-source software packages: `lenstronomy` is used for all gravitational lensing calculations and image simulation; `numpy` [@numpy] arrays are used internally to store image data and perform vectorized calculations; `pandas` [@pandas] dataframes are utilized for storing simulation metadata and file reading and writing; `scipy` [@scipy] is used for integration and interpolation; `matplotlib` [@matplotlib] functions are used for image visualization; `astropy` [@astropy] is used for cosmological calculations and color image production; `h5py` [@h5py] is utilized for saving images; and `PyYAML` [@pyyaml] is used to manage the configuration file. 
While not used directly, some `python-benedict` [@benedict] functionalities helped to create `deeplenstronomy`s data structures and internal search algorithms. 

`deeplenstronomy` is packaged and disseminated via [PyPI](https://pypi.org/project/deeplenstronomy/). 
Documentation and example notebooks are available on the [`deeplenstronomy` website](https://deepskies.github.io/deeplenstronomy/). 
Any bugs or feature requests can be opened as issues in the [GitHub
repository](https://github.com/deepskies/deeplenstronomy/issues) [@deeplenstronomy].

# Acknowledgements

R. Morgan thanks the LSSTC Data Science Fellowship Program, which is funded by LSSTC, NSF Cybertraining Grant #1829740, the Brinson Foundation, and the Moore Foundation; his participation in the program has benefited this work. 
R. Morgan also thanks the Universities Research Association Fermilab Visiting Scholar Program for funding his work on this project.
We acknowledge contributions from Joao Caldeira during the early stages of this project.

We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who’ve facilitated an environment of open discussion, idea-generation, and collaboration. 
This community was important for the development of this project.

Work supported by the Fermi National Accelerator Laboratory, managed and operated by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy. 
The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for U.S. Government purposes.


# References

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

Astronomical imaging has enabled the observation and analysis of strong gravitational lensing (SL), an astronomical phenomenon in which light from a distant object is deflected by the gravitational field of another object along its path to Earth.
These systems are of great scientific interest and can lead to information about the rate of the Universe's expansion and provide a test for the theory of General Relativity when discovered. 

Traditional searches have involved manual inspection of images to identify the characteristic arcs, colors, and orientations associated with SL.
However, identifying these systems with the traditional approach is challenging due to the large sizes of modern and future astronomical image datasets.
To automate the strong lens detection process, recent analyses have applied two-dimensional convolutional neural networks (2D-CNNs) to astronomical images.
Because of the relatively low number of observed SL systems, large simulated datasets of images are often utilized to train the networks.
Thus, the composition and production of these simulated datasets has become an integral part of the field.

`lenstronomy` [@lenstronomy] is one tool for producing training datasets of SL.
`lenstronomy` works by the user specifying the properties of the camera and physical systems through a `python`-based application programming interface (API) to generate a single image.
It is then up to the user to create a full dataset to train their neural network for SL detection. 

# Statement of need 

Due to the inherent dependence of the performance of machine learning (ML) approaches on the training data utilized, the 2D-CNN approach to SL detection is in tension with scientific reproducibility without a clear prescription for the simulation of the training data.
There is a need for a tool that simulates full datasets in an efficient and reproducible manner, while enabling the use of all the features of the simulation API.
As well, there is a need for this tool to simplify user interaction with `lenstronomy` and organize the simulations and associated metadata into convenient data structures for deep learning problems.

# Summary

`deeplenstronomy` generates SL data sets by organizing and expediting user interaction with `lenstronomy`.
The user creates a single yaml-style configuration file that describes the aspects of the dataset: number of images, properties of the camera, cosmological parameters, observing conditions, properties of the physical objects, and geometry of the SL systems.
`deeplenstronomy` parses the configuration file and generates the dataset, producing both the images and the parameters that led to the production of each image as outputs.
The configuration files can easily be shared, enabling users to easily reproduce each other's training datasets.

`deeplenstronomy` also contains built-in features to help astronomers make their training datasets as realistic as possible.
These features include the following functionalities: use any stellar light profile or mass profile in `lenstronomy`; simulate a variety of astronomical systems such as single galaxies, foreground stars, galaxy clusters, supernovae, and kilonovae, as well as any combination of those systems; fully control the placements of objects in the simulations; use observing conditions of real astronomical surveys; draw any parameter from any probability distribution; introduce any correlation; inspect and visualize the simulation outputs; incorporate real images into the simulation; and simulate time series data.

`deeplenstronomy` makes use of multiple open-source software packages: `lenstronomy` is used for all gravitational lensing calculations and image simulation; `numpy` [@numpy] arrays are used internally to store image data and perform vectorized calculations; `pandas` [@pandas] dataframes are utilized for storing simulation metadata and file reading and writing; `scipy` [@scipy] is used for integration and interpolation; `matplolib` [@matplotlib] functions are used for image visualization; `astropy` [@astropy] is used for cosmological calculations and color image production; `h5py` [@h5py] is utilized for saving images; and `PyYAML` [@pyyaml] is used to manage the configuration file.
While not used directly, some `python-benedict` [@benedict] functionalities helped to create `deeplenstronomy`s data structures and internal search algorithms.

Any bugs or feature requests can be opened as issues in the [GitHub repository](https://github.com/deepskies/deeplenstronomy/issues) [@deeplenstronomy].

R. Morgan thanks the LSSTC Data Science Fellowship Program, which is funded by LSSTC, NSF Cybertraining Grant #1829740, the Brinson Foundation, and the Moore Foundation; his participation in the program has benefited this work.
R. Morgan also thanks the Universities Research Association Fermilab Visiting Scholar Program for funding his work on this project. 


# References

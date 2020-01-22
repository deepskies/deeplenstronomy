# Simulator

Scrits to simulate DES-like image cutouts of lensed quasars. 
Quasar color information is based on SNANA simulated AGN light curves.

### Writen by Rob Morgan and Josh Yao-Yu Lin

## Data

All light curves are located in the [google bucket](https://console.cloud.google.com/storage/browser/deepskies-strong-lenses/quasar_sims).
Download and untar the file `lcs_plus_gal_param.tar.gz` into the current working directory.

## Main Interface: run_simulator.py

`run_simulator.py` takes 2 command line arguments:

1. The number of griz images to be simulated. Max is ~115,000 at present.
2. The object class to be simulated. Choose from `['lenses', 'foregrounds', 'galaxies']`.

### Usage Example

To simulate 10,000 double/quad lensed quasars, the command would be:

`python run_simulator.py 10000 lenses`

## Inner Workings: simulator.py

`run_simulator.py` calls the image simulation functions in `simulator.py` and stores all images in an array.
The functions in `simulator.py` were adapted from `SImulation_full_pipeline_double_quad_non_lens.ipynb`.
Changes include passing light curve, psf, and geometric information to from the input data files directly to the image generation.

## Notes:

This code likely needs to be proofed and made more efficient.






.. :changelog:

History
-------

0.0.1.9 (In development)
+++++++++++++++++++++
* Fix bug in number of lensed point sources

0.0.1.8 (2021-04-26)
+++++++++++++++++++++
* Fix bug in extrapolating nites outside of SEDs for time series

* Require each galaxy to have at least one mass profile

* Require each configuration to have at least two planes

0.0.1.7 (2021-04-01)
+++++++++++++++++++++
* Write simulation input dicts to disk to limit memory usage

* Improve accuracy of calculated magnitudes from SEDs (SNe, KN)

* Track cosmographic time delays in the metadata

* Make image backgrounds compatible with time series

0.0.1.6 (2021-03-16)
+++++++++++++++++++++
* Fix bug in calculation of K-Correction

* Add DES deep field distributiions
  
0.0.1.5 (2021-03-10)
+++++++++++++++++++++
* Fix bug in the number of times a USERDIST gets sampled

* Fix bug in lsst survey mode

* Fix bug in the redshifting calculations for supernovae to prevent NaNs

0.0.1.4 (2021-03-03)
+++++++++++++++++++++
* Fix bug in checking configuration file geometry section

* Speed improvements for timeseries functionalities

* Corner plot functionality for metadata visualization

0.0.1.3 (2021-02-02)
+++++++++++++++++++++

* Introducing the static model for timeseries

* Introducing the peakshift parameter for timeseries

* More accurate treatment of noise for timeseries

0.0.1.2 (2021-01-29)
+++++++++++++++++++++

* Fix bug in saving both sigma_v and theta_E 

* Full API documentation

0.0.1.0 (2020-11-09)
+++++++++++++++++++++

* First official public release

0.0.0.14 (2020-11-09)
+++++++++++++++++++++

* Bug fixes in distributions

* Unit tests

0.0.0.11 (2020-10-27)
+++++++++++++++++++++

* Bug fixes in image backgrounds

* Random seeds

* Search for dataset parameter names

0.0.0.10 (2020-09-30)
+++++++++++++++++++++

* Beta Release

0.0.0.9 (2020-09-30)
++++++++++++++++++++

* Image Backgrounds

* User Distributions

0.0.0.6 (2020-08-17)
++++++++++++++++++++

* Implement time-series functinalities

0.0.0.1 (2020-01-24)
++++++++++++++++++++

* Rebrand to yaml-style configuration file

0.1.0 (2019-01-03)
++++++++++++++++++

* First release on PyPI.

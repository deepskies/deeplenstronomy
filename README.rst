=============================
deeplenstronomy
=============================

# Pipeline Structure
1. Inputs 
   1. YAML file
   2. via function call
2. Generate objects
   1. with some population
   2. of a given type
3. Run Diagnostics over sets of objects


# Elements of Simulations
1. Survey Model (noise and data fidelity): seeing (dist, per band), sky-brightness (dist; corr with seeing; per band?), zero-point (const; per band?), exp time (per band), num exposures, pixel scale (const), read noise, filter set
2. Expected population distribution - where does this come from
3. Injection simulations into real data
4. Sky noise: poisson (from lens, source, uniform sky bkg)





.. image:: https://badge.fury.io/py/deeplenstronomy.png
    :target: http://badge.fury.io/py/deeplenstronomy

.. image:: https://travis-ci.org/bnord/deeplenstronomy.png?branch=master
    :target: https://travis-ci.org/bnord/deeplenstronomy




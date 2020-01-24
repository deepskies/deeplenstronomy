#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.MD').read()
doclink =   """
            Documentation
            -------------

            The full documentation is at http://ImSim.rtfd.org.
            """

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='deeplenstronomy',
    version='0.0.0.1',
    description='wrap lenstronomy for efficient simulation generation',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='DeepSkiesLab',
    author_email='deepskieslab@gmail.com',
    url='https://github.com/deepskies/deeplenstronomy',
    packages=[
        'deeplenstronomy',
    ],
    package_dir={'deeplenstronomy': 'deeplenstronomy'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='deeplenstronomy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

#!/usr/bin/env python

import os
import sys

from os import path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

history = open('HISTORY.rst').read().replace('.. :changelog:', '')


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
    name='deeplenstronomy',
    version='0.0.0.3',
    description='wrap lenstronomy for efficient simulation generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
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

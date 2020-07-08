#!/Users/rmorgan/anaconda3/bin/python

# build a deeplenstronomy wheel

import os

os.chdir('..')
os.system('python setup.py bdist_wheel')
os.system('pip uninstall deeplenstronomy -y')
os.system('pip install dist/deeplenstronomy-0.0.0.5-py2.py3-none-any.whl')

#!/usr/bin/env python

from setuptools import setup, find_packages
import os,sys
sys.path.insert(0, f'{os.path.dirname(__file__)}/tomodrgn')
import tomodrgn
version = tomodrgn.__version__

setup(name='tomodrgn',
      version=version,
      description='tomoDRGN heterogeneity training and analysis',
      author='Barrett Powell',
      author_email='zhonge@mit.edu',
      url='https://github.com/bpowell122/tomodrgn',
      license='GPLv3',
      zip_safe=False,
      packages=find_packages(),
      entry_points={
          "console_scripts": [
            "tomodrgn = tomodrgn.__main__:main",
            ],
      },
      include_package_data = True,
      python_requires='>=3.7',
      install_requires=[
        'torch>=1.0.0',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy>=1.3.1',
        'scikit-learn',
        'seaborn',
        'cufflinks',
        'jupyterlab',
        'umap-learn',
        'ipywidgets',
        'ipyvolume'
        ]
     )

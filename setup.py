#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
import os,sys
sys.path.insert(0, f'{os.path.dirname(__file__)}/tomodrgn')
import tomodrgn
version = tomodrgn.__version__

setup(name='tomodrgn',
      version=version,
      description='tomoDRGN heterogeneity training and analysis',
      author='Barrett Powell',
      author_email='bmp@mit.edu',
      url='https://github.com/bpowell122/tomodrgn',
      license='GPLv3',
      zip_safe=False,
      packages=find_namespace_packages(where='.',
                                       include=['tomodrgn*', 'testing*'],
                                       exclude=['testing.output*']),
      # packages=find_packages(where='.',
      #                        exclude=['*output*']),
      # package_dir={
      #     "tomodrgn": "tomodrgn",
      #     "tomodrgn.commands": "tomodrgn/commands",
      #     "tomodrgn.templates": "tomodrgn/templates",
      #     },
      package_data={"tomodrgn.templates": ["*.ipynb"]},
      entry_points={
          "console_scripts": [
            "tomodrgn = tomodrgn.__main__:main",
            ],
      },
      # include_package_data=True,
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
        'umap-learn',
        'ipywidgets',
        'ipyvolume',
        'plotly',
        'pillow',
        'healpy',
        'typing_extensions>=3.7.4',
        'ipyvolume>=0.6.0a10',
        'traitlets'
        ]
     )

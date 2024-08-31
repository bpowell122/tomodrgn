#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_namespace_packages

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
      python_requires='>=3.10',
      install_requires=[
          'numpy<=2.0',  # not sure if v2's breaking changes affect tomodrgn
          'pandas',
          'torch>=1.13.0',  # nested tensor requires >=1.13 for autograd, if using torch.compile then need >=2.0
          'torchinfo',  # used for visualizing the torch model as torchinfo.summary(model)
          'einops',
          'healpy',
          'matplotlib>=3.5',  # draw_without_rendering
          'seaborn',
          'scipy>=1.3.1',
          'scikit-learn',
          'umap-learn',
          'notebook',
          'ipython',
          'ipywidgets',
          'plotly',
          'typing_extensions>=3.7.4',
          'adjustText',
          'fastcluster',  # added due to warning in analyze_volumes sns.clustermap of large array (n_vols x boxsize**3)
          'sphinx',
          'pydata_sphinx_theme',
          'sphinx_design',
          'sphinx-copybutton',
          'sphinx-simplepdf',
          'importlib_resources',
      ]
      )

import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tomoDRGN'
copyright = '2024, Barrett M Powell, Joseph H Davis'
author = 'Barrett M Powell, Joseph H Davis'
release = metadata.version('tomodrgn')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# significant gratitude to https://github.com/sphinx-doc/sphinx/issues/7912

extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    'sphinx.ext.autodoc',  # automatic documentation generation
    'sphinx.ext.autosummary',  # automatic recursive documentation generation
    'sphinx.ext.viewcode',  # linking module/class/etc documentation to source code
    'sphinx_simplepdf',    # viewing and building PDFs
    'sphinxarg.ext',   # adding argparse documentation to sphinx docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True  # Turn on sphinx.ext.autosummary
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False # Remove namespaces from class/method signatures


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'navigation_with_keys': False,
    'logo': {
        'text': f'tomoDRGN v{metadata.version("tomodrgn")}',  # set the logo display text at top left of navigation bar
    }
}
html_static_path = ['_static']

import os
import sys

sys.path.insert(0, os.path.abspath('../src/lidar_aug'))


class MockClass():
    pass


import lidar_aug

lidar_aug.evaluation.KeysView = MockClass
lidar_aug.evaluation.ItemsView = MockClass
lidar_aug.evaluation.ValuesView = MockClass

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LidarAug'
copyright = '2024, Tom Schammo, Sven Teufel'
author = 'Tom Schammo, Sven Teufel'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

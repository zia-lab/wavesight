# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'wavesight'
copyright = ''
author = 'Juan David Lizarazo Ferro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_css_files = ['custom.css']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_context = {
    'show_copyright': False,
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']


html_theme_options = {
    'prev_next_buttons_location': 'top',
    # 'style_nav_header_background': 'blue',
}
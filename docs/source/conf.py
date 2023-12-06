# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

# -- Project information -----------------------------------------------------

project = 'deepskin - Wound analysis using smartphone images'
copyright = '2023, Nico Curti'
author = 'Nico Curti'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinx_rtd_theme',
              #'rst2pdf.pdfbuilder',
              'nbsphinx',
              'IPython.sphinxext.ipython_console_highlighting',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for PDF output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples# (source start file, target name, title, author, documentclass [howto/manual]).
latex_engine = 'xelatex'
latex_documents = [('index', 'deepskin.tex', u'deepskin - Wound analysis using smartphone images', u'Nico Curti', 'manual'),]
latex_show_pagerefs = True
latex_domain_indices = False

pdf_documents = [('index', u'deepskin', u'deepskin - Wound analysis using smartphone images', u'Nico Curti'),]

nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_kernel_name = 'python3'
nbsphinx_output_prompt = 'Out[%s]:'
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

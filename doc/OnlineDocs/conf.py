#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pyomo documentation build configuration file, created by
# sphinx-quickstart on Mon Dec 12 16:08:36 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# assumes pyutilib source is next to the pyomo source directory
sys.path.insert(0, os.path.abspath('../../../pyutilib'))
# top-level pyomo source directory
sys.path.insert(0, os.path.abspath('../..'))

# -- Rebuild SPY files ----------------------------------------------------
sys.path.insert(0, os.path.abspath('src'))
try:
    print("Regenerating SPY files...")
    from strip_examples import generate_spy_files

    generate_spy_files(os.path.abspath('src'))
    generate_spy_files(
        os.path.abspath(os.path.join('library_reference', 'kernel', 'examples'))
    )
finally:
    sys.path.pop(0)

# -- Options for intersphinx ---------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'Sphinx': ('https://www.sphinx-doc.org/en/stable/', None),
}

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '1.8'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.ifconfig',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    #'sphinx.ext.githubpages',
]

viewcode_follow_imported_members = True
# napoleon_include_private_with_doc = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Pyomo'
copyright = u'2008-2023, Sandia National Laboratories'
author = u'Pyomo Developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import pyomo.version

version = pyomo.version.__version__
# The full version, including alpha/beta/rc tags.
release = pyomo.version.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# If true, doctest flags (comments looking like # doctest: FLAG, ...) at
# the ends of lines and <BLANKLINE> markers are removed for all code
# blocks showing interactive Python sessions (i.e. doctests)
trim_doctest_flags = True

# If true, figures, tables and code-blocks are automatically numbered if
# they have a caption.
numfig = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

html_theme = 'sphinx_rtd_theme'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['theme_overrides.css']

html_favicon = "../logos/pyomo/favicon.ico"


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyomo'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, 'pyomo.tex', 'Pyomo Documentation', 'Pyomo', 'manual')]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'pyomo', 'Pyomo Documentation', [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'pyomo',
        'Pyomo Documentation',
        author,
        'Pyomo',
        'One line description of project.',
        'Miscellaneous',
    )
]

# autodoc_member_order = 'bysource'
# autodoc_member_order = 'groupwise'

# -- Check which conditional dependencies are available ------------------
# Used for skipping certain doctests
from sphinx.ext.doctest import doctest

doctest_default_flags = (
    doctest.ELLIPSIS
    + doctest.NORMALIZE_WHITESPACE
    + doctest.IGNORE_EXCEPTION_DETAIL
    + doctest.DONT_ACCEPT_TRUE_FOR_1
)


class IgnoreResultOutputChecker(doctest.OutputChecker):
    IGNORE_RESULT = doctest.register_optionflag('IGNORE_RESULT')

    def check_output(self, want, got, optionflags):
        if optionflags & self.IGNORE_RESULT:
            return True
        return super().check_output(want, got, optionflags)


doctest.OutputChecker = IgnoreResultOutputChecker

doctest_global_setup = '''
import os, platform, sys
on_github_actions = bool(os.environ.get('GITHUB_ACTIONS', ''))
system_info = (
    sys.platform,
    platform.machine(),
    platform.python_implementation()
)

from pyomo.common.dependencies import (
    attempt_import, numpy_available, scipy_available, pandas_available,
    yaml_available, networkx_available, matplotlib_available,
    pympler_available, dill_available,
)
pint_available = attempt_import('pint', defer_check=False)[1]
from pyomo.contrib.parmest.parmest import parmest_available

import pyomo.environ as _pe # (trigger all plugin registrations)
import pyomo.opt as _opt

# Not using SolverFactory to check solver availability because
# as of June 2020 there is no way to suppress warnings when 
# solvers are not available
ipopt_available = bool(_opt.check_available_solvers('ipopt'))
sipopt_available = bool(_opt.check_available_solvers('ipopt_sens'))
k_aug_available = bool(_opt.check_available_solvers('k_aug'))
dot_sens_available = bool(_opt.check_available_solvers('dot_sens'))
baron_available = bool(_opt.check_available_solvers('baron'))
glpk_available = bool(_opt.check_available_solvers('glpk'))
gurobipy_available = bool(_opt.check_available_solvers('gurobi_direct'))

baron = _opt.SolverFactory('baron')

if numpy_available and scipy_available:
    import pyomo.contrib.pynumero.asl as _asl
    asl_available = _asl.AmplInterface.available()
    import pyomo.contrib.pynumero.linalg.ma27 as _ma27
    ma27_available = _ma27.MA27Interface.available()
    from pyomo.contrib.pynumero.linalg.mumps_interface import mumps_available
else:
    asl_available = False
    ma27_available = False
    mumps_available = False
'''

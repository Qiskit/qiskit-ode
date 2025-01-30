# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

# General options:

project = 'Qiskit Dynamics'
copyright = '2020, Qiskit Development Team'  # pylint: disable=redefined-builtin
author = 'Qiskit Development Team'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = '0.6.0'

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx_autodoc_typehints',
    'reno.sphinxext',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinxcontrib.bibtex',
    "qiskit_sphinx_theme",
    "matplotlib.sphinxext.plot_directive",
]
templates_path = ["_templates"]

numfig = True
numfig_format = {
    'table': 'Table %s'
}
language = 'en'
pygments_style = 'colorful'
add_module_names = False
modindex_common_prefix = ['qiskit_dynamics.']
bibtex_default_style = 'unsrt'
bibtex_bibfiles = ['refs.bib']
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

# html theme options
html_theme = 'qiskit-ecosystem'
html_last_updated_fmt = '%Y/%m/%d'

docs_url_prefix = "qiskit-dynamics"

html_context = {"version_list": ["0.4"]}

html_theme_options = {
    "sidebar_qiskit_ecosystem_member": True,
}

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
intersphinx_mapping = {
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
    "qiskit_experiments": ("https://qiskit-community.github.io/qiskit-experiments/", None),
    "arraylias": ("https://qiskit-community.github.io/arraylias/", None)
}

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
# TODO: swap this with always if tutorial execution is too slow for ci and needs
# a separate job
# nbsphinx_execute = os.getenv('QISKIT_DOCS_BUILD_TUTORIALS', 'never')
nbsphinx_execute = 'always'
nbsphinx_widgets_path = ''
exclude_patterns = ['_build', '**.ipynb_checkpoints']
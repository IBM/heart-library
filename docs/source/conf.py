# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'HEART-library'
copyright = '2024, IBM'
author = 'IBM'
release = '0.6.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx_charts.charts', 'sphinx_design', 'sphinx.ext.viewcode',
    'sphinx.ext.napoleon','myst_parser', 'qiskit_sphinx_theme', 'sphinx_copybutton']

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

html_static_path = ['_static']
html_css_files = ['global_glossary_format.css']
templates_path = ['_templates']
exclude_patterns = []

# This sets the title to e.g. `My Project 4.12`.
html_title = f"{project} {release}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "qiskit-ecosystem"
html_show_sphinx = False
html_show_copyright = False
html_theme_options = {
    "sidebar_hide_name": True,
     "light_css_variables": {
        "color-brand-primary": "#0f62fe",
        "color-brand-content": "#0f62fe",
        "color-link--hover": "#0043ce",
        "color-link-visited": "#8a3ffc",
    },
     "dark_css_variables": {
        "color-brand-primary": "#0f62fe",
        "color-brand-content": "#0f62fe",
        "color-link": "#78a9ff",
        "color-link--hover": "a6c8ff",
        "sd-color-primary-text": "FFFFFF",
    },
    "light_logo": "theme/SVG/HEART-logo-light.svg",
    "dark_logo": "theme/SVG/HEART-logo-dark.svg",
}

def setup(app):
    app.add_css_file('theme/custom.css')  # may also be an URL

# translations_list = [
#     ('en', 'English'),
# ]

# html_context = {
#    "version_list": ["0.5.0"],
# }

# docs_url_prefix = "HEART-library"
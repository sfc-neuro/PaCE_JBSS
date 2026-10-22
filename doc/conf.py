from datetime import datetime, timezone
td = datetime.now(tz=timezone.utc)

project = "PACE"
author = "Silvia Cotroneo and Marijn van Vliet"
copyright = f"2026-{td.year}, {author}."

# This is a nice theme.
html_theme = "furo"

# Disable the left sidebar. Our documentation is so small we don't need it.
html_sidebars = {
    "**": [],
}

# For auto-generating the API docs.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "numpydoc",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

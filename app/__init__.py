"""Messy Desk Lab application package.

Holds the Streamlit app and its headless logic layer. The logic layer
(``app.logic``) imports only the simulation core under ``src/`` plus numpy and
the standard library; it never imports ``streamlit``, ``pygame``, or ``render``.
"""

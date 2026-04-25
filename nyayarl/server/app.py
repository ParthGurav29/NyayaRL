"""
Deprecated API module.

The canonical API is `server/app.py` (mounted by the Space entrypoint).
This file remains as a compatibility shim for older imports.
"""

from server.app import app  # noqa: F401

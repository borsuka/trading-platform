"""Serving the dashboard as part of the API process."""

from app.web.ui import find_ui_directory, mount_ui

__all__ = ["find_ui_directory", "mount_ui"]

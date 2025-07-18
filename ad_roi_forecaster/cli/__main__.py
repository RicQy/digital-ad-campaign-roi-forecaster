#!/usr/bin/env python3
"""
Main entry point for the CLI module.

This allows running the CLI as a module:
    python -m ad_roi_forecaster.cli
"""

from .cli import cli

if __name__ == '__main__':
    cli()

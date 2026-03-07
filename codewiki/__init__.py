"""
CodeWiki: Transform codebases into comprehensive documentation using AI-powered analysis.

This package provides a CLI tool for generating documentation from code repositories.
"""

__version__ = "1.0.1"
__author__ = "CodeWiki Contributors"
__license__ = "MIT"
__all__ = ["cli", "__version__"]


def __getattr__(name: str):
    if name == "cli":
        from codewiki.cli.main import cli

        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

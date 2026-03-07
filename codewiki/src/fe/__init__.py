#!/usr/bin/env python3
"""
CodeWiki Frontend Module

Web interface components for the documentation generation service.
"""

from .models import JobStatus, JobStatusResponse, RepositorySubmission, CacheEntry

__all__ = [
    "app",
    "main",
    "JobStatus",
    "JobStatusResponse",
    "RepositorySubmission",
    "CacheEntry",
    "CacheManager",
    "BackgroundWorker",
    "GitHubRepoProcessor",
    "WebRoutes",
]


def __getattr__(name: str):
    if name in {"app", "main"}:
        from .web_app import app, main

        return {"app": app, "main": main}[name]
    if name == "CacheManager":
        from .cache_manager import CacheManager

        return CacheManager
    if name == "BackgroundWorker":
        from .background_worker import BackgroundWorker

        return BackgroundWorker
    if name == "GitHubRepoProcessor":
        from .github_processor import GitHubRepoProcessor

        return GitHubRepoProcessor
    if name == "WebRoutes":
        from .routes import WebRoutes

        return WebRoutes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

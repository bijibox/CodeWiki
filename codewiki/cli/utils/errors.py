"""
Error handling utilities and exit codes for CLI.

Exit Codes:
  0: Success
  1: General error
  2: Configuration error (missing/invalid credentials)
  3: Repository error (not a git repo, no code files)
  4: LLM API error (including rate limits)
  5: File system error (permissions, disk space)
"""

import sys

from codewiki.cli.utils.logging import CLILogger, create_logger

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_REPOSITORY_ERROR = 3
EXIT_API_ERROR = 4
EXIT_FILESYSTEM_ERROR = 5


class CodeWikiError(Exception):
    """Base exception for CodeWiki CLI errors."""

    def __init__(self, message: str, exit_code: int = EXIT_GENERAL_ERROR):
        self.message = message
        self.exit_code = exit_code
        super().__init__(self.message)


class ConfigurationError(CodeWikiError):
    """Configuration-related errors."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_CONFIG_ERROR)


class RepositoryError(CodeWikiError):
    """Repository-related errors."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_REPOSITORY_ERROR)


class APIError(CodeWikiError):
    """LLM API-related errors."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_API_ERROR)


class FileSystemError(CodeWikiError):
    """File system-related errors."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_FILESYSTEM_ERROR)


def handle_error(error: Exception, verbosity: int = 0, logger: CLILogger | None = None) -> int:
    """
    Handle errors and return appropriate exit code.

    Args:
        error: The exception to handle
        verbosity: Verbosity level
        logger: Optional shared logger facade

    Returns:
        Exit code for the error
    """
    active_logger = logger or create_logger(verbosity=verbosity, name="codewiki.cli.errors")
    if isinstance(error, CodeWikiError):
        active_logger.error(f"Error: {error.message}")
        return error.exit_code
    else:
        active_logger.error(f"Unexpected error: {error}")
        if verbosity >= 1:
            import traceback

            active_logger.error(traceback.format_exc())
        return EXIT_GENERAL_ERROR


def error_with_suggestion(message: str, suggestion: str, exit_code: int = EXIT_GENERAL_ERROR):
    """
    Display error message with actionable suggestion and exit.

    Args:
        message: The error message
        suggestion: Suggested action to resolve the error
        exit_code: Exit code to use
    """
    logger = create_logger(name="codewiki.cli.errors")
    logger.error(message)
    logger.info(suggestion)
    sys.exit(exit_code)


def warning(message: str):
    """Display a warning message."""
    create_logger(name="codewiki.cli.errors").warning(message)


def success(message: str):
    """Display a success message."""
    create_logger(name="codewiki.cli.errors").success(message)


def info(message: str):
    """Display an info message."""
    create_logger(name="codewiki.cli.errors").info(message)

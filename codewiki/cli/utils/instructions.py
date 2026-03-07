"""
Post-generation instructions generator.
"""

from pathlib import Path
from typing import Any

from codewiki.cli.utils.logging import CLILogger, create_logger


def compute_github_pages_url(repo_url: str, repo_name: str) -> str:
    """
    Compute expected GitHub Pages URL from repository URL.

    Args:
        repo_url: GitHub repository URL
        repo_name: Repository name

    Returns:
        Expected GitHub Pages URL
    """
    # Extract owner from GitHub URL
    # e.g., "https://github.com/owner/repo" -> "owner"
    if "github.com" in repo_url:
        parts = repo_url.rstrip("/").split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1].replace(".git", "")
            return f"https://{owner}.github.io/{repo}/"

    return f"https://YOUR_USERNAME.github.io/{repo_name}/"


def get_pr_creation_url(repo_url: str, branch_name: str) -> str:
    """
    Get PR creation URL for GitHub.

    Args:
        repo_url: GitHub repository URL
        branch_name: Branch name

    Returns:
        PR creation URL
    """
    base_url = repo_url.rstrip("/").replace(".git", "")
    return f"{base_url}/compare/{branch_name}"


def display_post_generation_instructions(
    output_dir: Path,
    repo_name: str,
    repo_url: str | None = None,
    branch_name: str | None = None,
    github_pages: bool = False,
    files_generated: list[str] | None = None,
    statistics: dict[str, Any] | None = None,
    logger: CLILogger | None = None,
) -> None:
    """
    Display post-generation instructions.

    Args:
        output_dir: Output directory path
        repo_name: Repository name
        repo_url: GitHub repository URL (optional)
        branch_name: Git branch name (optional)
        github_pages: Whether GitHub Pages HTML was generated
        files_generated: List of generated files
        statistics: Generation statistics
        logger: Shared logger facade
    """
    active_logger = logger or create_logger(name="codewiki.cli.instructions")
    active_logger.blank()
    active_logger.success("Documentation generated successfully!")
    active_logger.blank()

    # Output directory
    active_logger.section("Output directory:")
    active_logger.info(f"  {output_dir}")
    active_logger.blank()

    # Generated files
    if files_generated:
        active_logger.section("Generated files:")
        for file in files_generated[:10]:  # Show first 10
            active_logger.info(f"  - {file}")
        if len(files_generated) > 10:
            active_logger.info(f"  ... and {len(files_generated) - 10} more")
        active_logger.blank()

    # Statistics
    if statistics:
        active_logger.section("Statistics:")
        if "module_count" in statistics:
            active_logger.info(f"  Total modules:     {statistics['module_count']}")
        if "total_files_analyzed" in statistics:
            active_logger.info(f"  Files analyzed:    {statistics['total_files_analyzed']}")
        if "generation_time" in statistics:
            minutes = int(statistics["generation_time"] // 60)
            seconds = int(statistics["generation_time"] % 60)
            active_logger.info(f"  Generation time:   {minutes} minutes {seconds} seconds")
        # if 'total_tokens_used' in statistics:
        #     tokens = statistics['total_tokens_used']
        #     active_logger.info(f"  Tokens used:       ~{tokens:,}")
        active_logger.blank()

    # Next steps
    active_logger.section("Next steps:")
    active_logger.blank()

    active_logger.info("1. Review the generated documentation:")
    active_logger.info(f"   cat {output_dir}/overview.md")
    if github_pages:
        active_logger.info(f"   open {output_dir}/index.html  # View in browser")
    active_logger.blank()

    if branch_name:
        # Git workflow with branch
        active_logger.info("2. Push the documentation branch:")
        active_logger.info(f"   git push origin {branch_name}")
        active_logger.blank()

        if repo_url:
            pr_url = get_pr_creation_url(repo_url, branch_name)
            active_logger.info("3. Create a Pull Request to merge documentation:")
            active_logger.info(f"   {pr_url}")
            active_logger.blank()

            active_logger.info("4. After merge, enable GitHub Pages:")
        else:
            active_logger.info("3. Enable GitHub Pages:")
    else:
        # Direct commit workflow
        active_logger.info("2. Commit the documentation:")
        active_logger.info("   git add docs/")
        active_logger.info('   git commit -m "Add generated documentation"')
        active_logger.blank()

        active_logger.info("3. Push to GitHub:")
        active_logger.info("   git push origin main")
        active_logger.blank()

        active_logger.info("4. Enable GitHub Pages:")

    active_logger.info("   - Go to repository Settings → Pages")
    active_logger.info("   - Source: Deploy from a branch")
    active_logger.info("   - Branch: main, folder: /docs")
    active_logger.blank()

    if repo_url:
        github_pages_url = compute_github_pages_url(repo_url, repo_name)
        active_logger.info("5. Your documentation will be available at:")
        active_logger.info(f"   {github_pages_url}")
        active_logger.blank()


def display_generation_summary(
    success: bool,
    error_message: str | None = None,
    output_dir: Path | None = None,
    logger: CLILogger | None = None,
) -> None:
    """
    Display generation summary (success or failure).

    Args:
        success: Whether generation was successful
        error_message: Error message if failed
        output_dir: Output directory if successful
        logger: Shared logger facade
    """
    active_logger = logger or create_logger(name="codewiki.cli.instructions")
    if success:
        active_logger.blank()
        active_logger.success("Generation completed successfully!")
        if output_dir:
            active_logger.info(f"Documentation saved to: {output_dir}")
        active_logger.blank()
    else:
        active_logger.blank()
        active_logger.error("Generation failed")
        if error_message:
            active_logger.info(error_message)
        active_logger.blank()

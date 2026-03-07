"""
Repository Analyzer Module

This module provides functionality to analyze repository structures and generate
detailed file tree representations with filtering capabilities.
"""

import fnmatch
from pathlib import Path
from typing import Dict, List, Optional
from codewiki.src.be.dependency_analyzer.utils.patterns import (
    DEFAULT_IGNORE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
)


class RepoAnalyzer:
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        # Include patterns: if specified, use ONLY those patterns (replaces defaults)
        self.include_patterns = (
            include_patterns if include_patterns is not None else DEFAULT_INCLUDE_PATTERNS
        )
        # Exclude patterns: if specified, MERGE with default ignore patterns
        self.exclude_patterns = (
            list(DEFAULT_IGNORE_PATTERNS) + exclude_patterns
            if exclude_patterns is not None
            else list(DEFAULT_IGNORE_PATTERNS)
        )

    def analyze_repository_structure(self, repo_dir: str) -> Dict[str, object]:
        file_tree = self._build_file_tree(repo_dir)
        return {
            "file_tree": file_tree,
            "summary": {
                "total_files": self._count_files(file_tree),
                "total_size_kb": self._calculate_size(file_tree),
            },
        }

    def _build_file_tree(self, repo_dir: str) -> Dict[str, object]:
        def build_tree(path: Path, base_path: Path) -> Optional[Dict[str, object]]:
            relative_path = path.relative_to(base_path)
            relative_path_str = str(relative_path)

            # 🚫 Reject symlinks
            if path.is_symlink():
                return None

            # 🚫 Reject escaped paths (e.g., symlinks pointing outside)
            try:
                if not path.resolve().is_relative_to(base_path.resolve()):
                    return None
            except AttributeError:
                if not str(path.resolve()).startswith(str(base_path.resolve())):
                    return None

            if self._should_exclude_path(relative_path_str, path.name):
                return None

            if path.is_file():
                if not self._should_include_file(relative_path_str, path.name):
                    return None

                size = path.stat().st_size
                return {
                    "type": "file",
                    "name": path.name,
                    "path": relative_path_str,
                    "extension": path.suffix,
                    "_size_bytes": size,
                }

            elif path.is_dir():
                children: list[Dict[str, object]] = []
                try:
                    for child in sorted(path.iterdir()):
                        child_tree = build_tree(child, base_path)
                        if child_tree is not None:
                            children.append(child_tree)
                except PermissionError:
                    pass

                if children or str(relative_path) == ".":
                    return {
                        "type": "directory",
                        "name": path.name,
                        "path": relative_path_str,
                        "children": children,
                    }
                return None

            # Other types (sockets, devices, etc.)
            return None

        root_tree = build_tree(Path(repo_dir), Path(repo_dir))
        if root_tree is None:
            return {
                "type": "directory",
                "name": Path(repo_dir).name,
                "path": ".",
                "children": [],
            }
        return root_tree

    def _should_exclude_path(self, path: str, filename: str) -> bool:
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
            if pattern.endswith("/") and path.startswith(pattern.rstrip("/")):
                return True
            if path.startswith(pattern + "/") or path == pattern:
                return True
            if pattern in path.split("/"):
                return True
        return False

    def _should_include_file(self, path: str, filename: str) -> bool:
        if not self.include_patterns:
            return True
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _count_files(self, tree: Dict[str, object]) -> int:
        if tree["type"] == "file":
            return 1
        children = tree.get("children", [])
        if not isinstance(children, list):
            return 0
        return sum(self._count_files(child) for child in children if isinstance(child, dict))

    def _calculate_size(self, tree: Dict[str, object]) -> float:
        if tree["type"] == "file":
            size_bytes = tree.get("_size_bytes", 0)
            if isinstance(size_bytes, (int, float)):
                return float(size_bytes) / 1024
            return 0.0
        children = tree.get("children", [])
        if not isinstance(children, list):
            return 0.0
        return sum(self._calculate_size(child) for child in children if isinstance(child, dict))

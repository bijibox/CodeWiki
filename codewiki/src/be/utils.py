import contextlib
import io
import base64
import re
from pathlib import Path
import logging
import traceback
from typing import Iterator, List, Mapping, Protocol, Sequence, Tuple

import requests
import tiktoken

from codewiki.src.config import DEFAULT_MERMAID_VALIDATOR

logger = logging.getLogger(__name__)


class HasFilePath(Protocol):
    file_path: str


# ------------------------------------------------------------
# ---------------------- Complexity Check --------------------
# ------------------------------------------------------------


def is_complex_module(
    components: Mapping[str, HasFilePath],
    core_component_ids: Sequence[str],
) -> bool:
    files = set()
    for component_id in core_component_ids:
        if component_id in components:
            files.add(components[component_id].file_path)

    result = len(files) > 1

    return result


# ------------------------------------------------------------
# ---------------------- Token Counting ---------------------
# ------------------------------------------------------------

enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text.
    """
    length = len(enc.encode(text))
    # logger.debug(f"Number of tokens: {length}")
    return length


# ------------------------------------------------------------
# ---------------------- Mermaid Validation -----------------
# ------------------------------------------------------------


@contextlib.contextmanager
def suppress_output() -> Iterator[None]:
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


async def validate_mermaid_diagrams(
    md_file_path: str,
    relative_path: str,
    mermaid_validator: str = DEFAULT_MERMAID_VALIDATOR,
) -> str:
    """
    Validate all Mermaid diagrams in a markdown file.

    Args:
        md_file_path: Path to the markdown file to check
        relative_path: Relative path to the markdown file
    Returns:
        "All mermaid diagrams are syntax correct" if all diagrams are valid,
        otherwise returns error message with details about invalid diagrams
    """

    try:
        # Read the markdown file
        file_path = Path(md_file_path)
        if not file_path.exists():
            return f"Error: File '{md_file_path}' does not exist"

        content = file_path.read_text(encoding="utf-8")

        # Extract all mermaid code blocks
        mermaid_blocks = extract_mermaid_blocks(content)

        if not mermaid_blocks:
            return "No mermaid diagrams found in the file"

        # Validate each mermaid diagram sequentially to avoid segfaults
        errors: list[str] = []
        for i, (line_start, diagram_content) in enumerate(mermaid_blocks, 1):
            error_msg = await validate_single_diagram(
                diagram_content,
                i,
                line_start,
                mermaid_validator=mermaid_validator,
            )
            if error_msg:
                errors.append("\n")
                errors.append(error_msg)

        # if errors:
        #     logger.debug(f"Mermaid syntax errors found in file: {md_file_path}: {errors}")

        if errors:
            return (
                "Mermaid syntax errors found in file: " + relative_path + "\n" + "\n".join(errors)
            )
        else:
            return "All mermaid diagrams in file: " + relative_path + " are syntax correct"

    except Exception as e:
        return f"Error processing file: {str(e)}"


def extract_mermaid_blocks(content: str) -> List[Tuple[int, str]]:
    """
    Extract all mermaid code blocks from markdown content.

    Returns:
        List of tuples containing (line_number, diagram_content)
    """
    mermaid_blocks = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for mermaid code block start
        if line == "```mermaid" or line.startswith("```mermaid"):
            start_line = i + 1
            diagram_lines = []
            i += 1

            # Collect lines until we find the closing ```
            while i < len(lines):
                if lines[i].strip() == "```":
                    break
                diagram_lines.append(lines[i])
                i += 1

            if diagram_lines:  # Only add non-empty diagrams
                diagram_content = "\n".join(diagram_lines)
                mermaid_blocks.append((start_line, diagram_content))

        i += 1

    return mermaid_blocks


async def validate_single_diagram(
    diagram_content: str,
    diagram_num: int,
    line_start: int,
    mermaid_validator: str = DEFAULT_MERMAID_VALIDATOR,
) -> str:
    """
    Validate a single mermaid diagram.

    Args:
        diagram_content: The mermaid diagram content
        diagram_num: Diagram number for error reporting
        line_start: Starting line number in the file

    Returns:
        Error message if invalid, empty string if valid
    """
    try:
        if mermaid_validator == "mermaid_parser_py":
            core_error = await validate_single_diagram_with_mermaid_parser(diagram_content)
        elif mermaid_validator == "mermaid_ink_api":
            core_error = await validate_single_diagram_with_mermaid_ink_api(diagram_content)
        else:
            return (
                f"  Diagram {diagram_num}: Exception during validation - "
                f"Unsupported mermaid validator: {mermaid_validator}"
            )
    except Exception as e:
        return f"  Diagram {diagram_num}: Exception during validation - {str(e)}"

    # Check if response indicates a parse error
    if core_error:
        # Extract line number from parse error and calculate actual line in markdown file
        line_match = re.search(r"line (\d+)", core_error)
        if line_match:
            error_line_in_diagram = int(line_match.group(1))
            actual_line_in_file = line_start + error_line_in_diagram
            newline = "\n"
            return f"Diagram {diagram_num}: Parse error on line {actual_line_in_file}:{newline}{newline.join(core_error.split(newline)[1:])}"
        else:
            return f"Diagram {diagram_num}: {core_error}"

    return ""  # No error


async def validate_single_diagram_with_mermaid_parser(diagram_content: str) -> str:
    import os
    import sys

    with suppress_output():
        from mermaid_parser.parser import parse_mermaid_py

    try:
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            await parse_mermaid_py(diagram_content)
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        return ""
    except Exception as e:
        error_str = str(e)
        error_pattern = r"Error:(.*?)(?=Stack Trace:|$)"
        match = re.search(error_pattern, error_str, re.DOTALL)

        if match:
            return match.group(0).strip()

        logger.error(f"Unable to parse mermaid-parser-py error output\n{error_str}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_str)


async def validate_single_diagram_with_mermaid_ink_api(diagram_content: str) -> str:
    encoded_diagram = base64.urlsafe_b64encode(diagram_content.encode("utf-8")).decode("ascii")
    response = requests.get(f"https://mermaid.ink/svg/{encoded_diagram}", timeout=20)

    if response.status_code == 200:
        return ""

    if response.status_code == 400:
        return response.text.strip()

    response.raise_for_status()
    return ""


if __name__ == "__main__":
    # Test with the provided file
    import asyncio

    test_file = "output/docs/SWE_agent-docs/agent_hooks.md"
    result = asyncio.run(validate_mermaid_diagrams(test_file, "agent_hooks.md"))
    print(result)

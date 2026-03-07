from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Protocol

from codewiki.src.utils import file_manager

if TYPE_CHECKING:
    from codewiki.src.be.dependency_analyzer.models.core import Node

DEFAULT_PROMPT_NAME = "en"
PROMPTS_PACKAGE = "codewiki"
PROMPTS_BASE_PATH = ("templates", "prompts")
PROMPT_FILE_MAP = {
    "system_prompt": "system_prompt.md",
    "leaf_system_prompt": "leaf_system_prompt.md",
    "user_prompt": "user_prompt.md",
    "repo_overview_prompt": "repo_overview_prompt.md",
    "module_overview_prompt": "module_overview_prompt.md",
    "cluster_repo_prompt": "cluster_repo_prompt.md",
    "cluster_module_prompt": "cluster_module_prompt.md",
    "filter_folders_prompt": "filter_folders_prompt.md",
}
PROMPT_ADDITION_FILE_MAP = {
    "prompt_additions_section": "additions/section.md",
    "prompt_addition_doc_type_api": "additions/doc_type_api.md",
    "prompt_addition_doc_type_architecture": "additions/doc_type_architecture.md",
    "prompt_addition_doc_type_user_guide": "additions/doc_type_user_guide.md",
    "prompt_addition_doc_type_developer": "additions/doc_type_developer.md",
    "prompt_addition_doc_type_generic": "additions/doc_type_generic.md",
    "prompt_addition_focus_modules": "additions/focus_modules.md",
    "prompt_addition_custom_instructions": "additions/custom_instructions.md",
}
REQUIRED_PROMPT_FILE_MAP = {
    **PROMPT_FILE_MAP,
    **PROMPT_ADDITION_FILE_MAP,
}


class PromptTemplateError(ValueError):
    """Raised when prompt templates cannot be loaded or are incomplete."""


class PromptTemplateSet(Protocol):
    """Raw template strings loaded from a prompt set."""

    @property
    def prompt_name(self) -> str: ...

    @property
    def system_prompt(self) -> str: ...

    @property
    def leaf_system_prompt(self) -> str: ...

    @property
    def user_prompt(self) -> str: ...

    @property
    def repo_overview_prompt(self) -> str: ...

    @property
    def module_overview_prompt(self) -> str: ...

    @property
    def cluster_repo_prompt(self) -> str: ...

    @property
    def cluster_module_prompt(self) -> str: ...

    @property
    def filter_folders_prompt(self) -> str: ...

    @property
    def prompt_additions_section(self) -> str: ...

    @property
    def prompt_addition_doc_type_api(self) -> str: ...

    @property
    def prompt_addition_doc_type_architecture(self) -> str: ...

    @property
    def prompt_addition_doc_type_user_guide(self) -> str: ...

    @property
    def prompt_addition_doc_type_developer(self) -> str: ...

    @property
    def prompt_addition_doc_type_generic(self) -> str: ...

    @property
    def prompt_addition_focus_modules(self) -> str: ...

    @property
    def prompt_addition_custom_instructions(self) -> str: ...


@dataclass(frozen=True)
class FilePromptTemplateSet:
    """Prompt template set loaded from package resources."""

    prompt_name: str
    system_prompt: str
    leaf_system_prompt: str
    user_prompt: str
    repo_overview_prompt: str
    module_overview_prompt: str
    cluster_repo_prompt: str
    cluster_module_prompt: str
    filter_folders_prompt: str
    prompt_additions_section: str
    prompt_addition_doc_type_api: str
    prompt_addition_doc_type_architecture: str
    prompt_addition_doc_type_user_guide: str
    prompt_addition_doc_type_developer: str
    prompt_addition_doc_type_generic: str
    prompt_addition_focus_modules: str
    prompt_addition_custom_instructions: str

    @classmethod
    def from_name(cls, prompt_name: str) -> "FilePromptTemplateSet":
        prompt_dir = _prompts_base_dir().joinpath(prompt_name)
        if not prompt_dir.is_dir():
            available = ", ".join(available_prompt_names()) or "<none>"
            raise PromptTemplateError(
                f"Unknown prompt set '{prompt_name}'. Available prompt sets: {available}"
            )
        return cls.from_directory(prompt_name, prompt_dir)

    @classmethod
    def from_directory(cls, prompt_name: str, prompt_dir: Any) -> "FilePromptTemplateSet":
        missing_files = [
            filename
            for filename in REQUIRED_PROMPT_FILE_MAP.values()
            if not prompt_dir.joinpath(filename).is_file()
        ]
        if missing_files:
            missing = ", ".join(sorted(missing_files))
            raise PromptTemplateError(
                f"Prompt set '{prompt_name}' is missing required prompt files: {missing}"
            )

        prompt_values = {
            attr_name: prompt_dir.joinpath(filename).read_text(encoding="utf-8").strip()
            for attr_name, filename in REQUIRED_PROMPT_FILE_MAP.items()
        }
        return cls(prompt_name=prompt_name, **prompt_values)


@dataclass(frozen=True)
class PromptBuilder:
    """Runtime helper that builds all prompts from a loaded template set."""

    templates: PromptTemplateSet

    @property
    def prompt_name(self) -> str:
        return self.templates.prompt_name

    def build_user_prompt(
        self,
        module_name: str,
        core_component_ids: list[str],
        components: Dict[str, "Node"],
        module_tree: "ModuleTree",
    ) -> str:
        return self.templates.user_prompt.format(
            module_name=module_name,
            formatted_core_component_codes=self._format_core_component_codes(
                core_component_ids, components
            ),
            module_tree=self._format_module_tree(module_tree, module_name),
        )

    def build_cluster_prompt(
        self,
        potential_core_components: str,
        module_tree: "ModuleTree | None" = None,
        module_name: str | None = None,
    ) -> str:
        current_module_tree = module_tree or {}
        if not current_module_tree:
            return self.templates.cluster_repo_prompt.format(
                potential_core_components=potential_core_components
            )
        return self.templates.cluster_module_prompt.format(
            potential_core_components=potential_core_components,
            module_tree=self._format_module_tree(current_module_tree, module_name),
            module_name=module_name,
        )

    def build_system_prompt(self, module_name: str, custom_instructions: str | None = None) -> str:
        return self.templates.system_prompt.format(
            module_name=module_name,
            custom_instructions=self._build_custom_instruction_section(custom_instructions),
        ).strip()

    def build_leaf_system_prompt(
        self, module_name: str, custom_instructions: str | None = None
    ) -> str:
        return self.templates.leaf_system_prompt.format(
            module_name=module_name,
            custom_instructions=self._build_custom_instruction_section(custom_instructions),
        ).strip()

    def build_repo_overview_prompt(self, repo_name: str, repo_structure: str) -> str:
        return self.templates.repo_overview_prompt.format(
            repo_name=repo_name,
            repo_structure=repo_structure,
        )

    def build_module_overview_prompt(self, module_name: str, repo_structure: str) -> str:
        return self.templates.module_overview_prompt.format(
            module_name=module_name,
            repo_structure=repo_structure,
        )

    def build_filter_folders_prompt(self, project_name: str, files: str) -> str:
        return self.templates.filter_folders_prompt.format(
            project_name=project_name,
            files=files,
        )

    def build_prompt_addition(self, agent_instructions: Mapping[str, Any] | None) -> str:
        if not agent_instructions:
            return ""

        additions: list[str] = []
        doc_type = agent_instructions.get("doc_type")
        if isinstance(doc_type, str) and doc_type:
            additions.append(self._build_doc_type_addition(doc_type))

        focus_modules = agent_instructions.get("focus_modules")
        if isinstance(focus_modules, list):
            filtered_focus_modules = [module for module in focus_modules if isinstance(module, str)]
            if filtered_focus_modules:
                additions.append(
                    self.templates.prompt_addition_focus_modules.format(
                        focus_modules=", ".join(filtered_focus_modules)
                    )
                )

        custom_instructions = agent_instructions.get("custom_instructions")
        if isinstance(custom_instructions, str) and custom_instructions:
            additions.append(
                self.templates.prompt_addition_custom_instructions.format(
                    custom_instructions=custom_instructions
                )
            )

        if not additions:
            return ""

        return self.templates.prompt_additions_section.format(instructions="\n".join(additions))

    def _format_module_tree(
        self, module_tree: "ModuleTree", current_module_name: str | None = None
    ) -> str:
        lines: list[str] = []

        def _walk(tree: ModuleTree, indent: int = 0) -> None:
            for key, value in tree.items():
                suffix = " (current module)" if key == current_module_name else ""
                lines.append(f"{'  ' * indent}{key}{suffix}")
                lines.append(
                    f"{'  ' * (indent + 1)} Core components: {', '.join(value['components'])}"
                )
                if isinstance(value.get("children"), dict) and value["children"]:
                    lines.append(f"{'  ' * (indent + 1)} Children:")
                    _walk(value["children"], indent + 2)

        _walk(module_tree, 0)
        return "\n".join(lines)

    def _format_core_component_codes(
        self, core_component_ids: list[str], components: Dict[str, "Node"]
    ) -> str:
        grouped_components: dict[str, list[str]] = {}
        for component_id in core_component_ids:
            if component_id not in components:
                continue
            component = components[component_id]
            grouped_components.setdefault(component.relative_path, []).append(component_id)

        sections: list[str] = []
        for path, component_ids_in_file in grouped_components.items():
            sections.append(f"# File: {path}\n")
            sections.append("## Core Components in this file:\n")

            for component_id in component_ids_in_file:
                sections.append(f"- {component_id}\n")

            extension = Path(path).suffix
            language = EXTENSION_TO_LANGUAGE.get(extension, "")
            sections.append(f"\n## File Content:\n```{language}\n")

            try:
                sections.append(
                    file_manager.load_text(components[component_ids_in_file[0]].file_path)
                )
            except (FileNotFoundError, IOError) as e:
                sections.append(f"# Error reading file: {e}\n")

            sections.append("```\n\n")

        return "".join(sections)

    def _build_custom_instruction_section(self, custom_instructions: str | None) -> str:
        if not custom_instructions:
            return ""
        return f"\n\n<CUSTOM_INSTRUCTIONS>\n{custom_instructions}\n</CUSTOM_INSTRUCTIONS>"

    def _build_doc_type_addition(self, doc_type: str) -> str:
        doc_type_templates = {
            "api": self.templates.prompt_addition_doc_type_api,
            "architecture": self.templates.prompt_addition_doc_type_architecture,
            "user-guide": self.templates.prompt_addition_doc_type_user_guide,
            "developer": self.templates.prompt_addition_doc_type_developer,
        }
        template = doc_type_templates.get(doc_type.lower())
        if template is not None:
            return template
        return self.templates.prompt_addition_doc_type_generic.format(doc_type=doc_type)


def _prompts_base_dir():
    return resources.files(PROMPTS_PACKAGE).joinpath(*PROMPTS_BASE_PATH)


def available_prompt_names() -> list[str]:
    prompts_dir = _prompts_base_dir()
    if not prompts_dir.is_dir():
        return []
    return sorted(entry.name for entry in prompts_dir.iterdir() if entry.is_dir())


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".md": "markdown",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".tsx": "typescript",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".php": "php",
    ".phtml": "php",
    ".inc": "php",
}


ModuleEntry = dict[str, Any]
ModuleTree = dict[str, ModuleEntry]

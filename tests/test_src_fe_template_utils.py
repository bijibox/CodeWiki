from types import SimpleNamespace

from codewiki.src.fe.template_utils import (
    render_job_list,
    render_navigation,
    render_template,
)


def test_render_template_escapes_html_by_default():
    rendered = render_template("Hello {{ name }}", {"name": "<b>CodeWiki</b>"})

    assert rendered == "Hello &lt;b&gt;CodeWiki&lt;/b&gt;"


def test_render_navigation_returns_empty_string_for_empty_tree():
    assert render_navigation({}) == ""


def test_render_navigation_marks_active_page_and_renders_sections():
    module_tree = {
        "core_module": {
            "components": ["service"],
            "children": {
                "child_page": {"components": ["handler"]},
            },
        }
    }

    rendered = render_navigation(module_tree, current_page="core_module.md")

    assert "Core Module" in rendered
    assert 'href="/core_module.md"' in rendered
    assert 'class="nav-item active"' in rendered
    assert "Child Page" in rendered


def test_render_job_list_includes_progress_and_completed_job_link():
    jobs = [
        SimpleNamespace(
            repo_url="https://github.com/example/repo",
            status="completed",
            progress="Finished",
            docs_path="/tmp/docs",
            job_id="job-123",
        ),
        SimpleNamespace(
            repo_url="https://github.com/example/other",
            status="processing",
            progress="Cloning repository",
            docs_path=None,
            job_id="job-456",
        ),
    ]

    rendered = render_job_list(jobs)

    assert "https://github.com/example/repo" in rendered
    assert "Completed" in rendered
    assert "Cloning repository" in rendered
    assert 'href="/docs/job-123"' in rendered


def test_render_job_list_returns_empty_string_for_empty_jobs():
    assert render_job_list([]) == ""

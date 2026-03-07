from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.src.be.dependency_analyzer.topo_sort import (
    build_graph_from_components,
    dependency_first_dfs,
    detect_cycles,
    resolve_cycles,
    topological_sort,
)


def make_node(node_id: str, component_type: str = "class", depends_on=None) -> Node:
    return Node(
        id=node_id,
        name=node_id.split(".")[-1],
        component_type=component_type,
        file_path=f"/repo/{node_id.replace('.', '/')}.py",
        relative_path=f"{node_id.replace('.', '/')}.py",
        depends_on=set(depends_on or []),
    )


def test_detect_cycles_returns_strongly_connected_components():
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"a"},
        "d": set(),
    }

    cycles = detect_cycles(graph)

    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_resolve_cycles_breaks_at_least_one_edge_in_cycle():
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"a"},
    }

    resolved = resolve_cycles(graph)

    assert resolved != graph
    assert detect_cycles(resolved) == []


def test_topological_sort_returns_dependencies_first():
    graph = {
        "service": {"repository", "logger"},
        "repository": {"logger"},
        "logger": set(),
    }

    order = topological_sort(graph)

    assert order.index("logger") < order.index("repository")
    assert order.index("repository") < order.index("service")


def test_dependency_first_dfs_returns_dependencies_before_dependents():
    graph = {
        "service": {"repository", "logger"},
        "repository": {"logger"},
        "logger": set(),
    }

    order = dependency_first_dfs(graph)

    assert order == ["logger", "repository", "service"]


def test_build_graph_from_components_ignores_external_dependencies():
    components = {
        "service": make_node("service", depends_on={"repository", "external"}),
        "repository": make_node("repository", depends_on={"logger"}),
        "logger": make_node("logger"),
    }

    graph = build_graph_from_components(components)

    assert graph == {
        "service": {"repository"},
        "repository": {"logger"},
        "logger": set(),
    }

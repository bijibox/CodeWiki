from datetime import datetime, timedelta

from codewiki.src.fe.cache_manager import CacheManager
from codewiki.src.fe.models import CacheEntry


def test_cache_manager_adds_and_loads_entries_from_disk(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path), cache_expiry_days=30)

    manager.add_to_cache("https://github.com/example/repo", "/tmp/docs/repo")

    reloaded = CacheManager(cache_dir=str(tmp_path), cache_expiry_days=30)
    cache_key = reloaded.get_repo_hash("https://github.com/example/repo")

    assert cache_key in reloaded.cache_index
    assert reloaded.cache_index[cache_key].docs_path == "/tmp/docs/repo"


def test_cache_manager_get_cached_docs_returns_path_and_updates_last_accessed(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path), cache_expiry_days=30)
    manager.add_to_cache("https://github.com/example/repo", "/tmp/docs/repo")

    cache_key = manager.get_repo_hash("https://github.com/example/repo")
    original_access = manager.cache_index[cache_key].last_accessed

    docs_path = manager.get_cached_docs("https://github.com/example/repo")

    assert docs_path == "/tmp/docs/repo"
    assert manager.cache_index[cache_key].last_accessed >= original_access


def test_cache_manager_removes_expired_entry_when_accessed(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path), cache_expiry_days=7)
    repo_url = "https://github.com/example/repo"
    cache_key = manager.get_repo_hash(repo_url)

    old_time = datetime.now() - timedelta(days=10)
    manager.cache_index[cache_key] = CacheEntry(
        repo_url=repo_url,
        repo_url_hash=cache_key,
        docs_path="/tmp/docs/repo",
        created_at=old_time,
        last_accessed=old_time,
    )

    assert manager.get_cached_docs(repo_url) is None
    assert cache_key not in manager.cache_index


def test_cache_manager_cleanup_expired_cache_keeps_recent_entries(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path), cache_expiry_days=7)

    expired_key = manager.get_repo_hash("https://github.com/example/expired")
    fresh_key = manager.get_repo_hash("https://github.com/example/fresh")
    now = datetime.now()

    manager.cache_index[expired_key] = CacheEntry(
        repo_url="https://github.com/example/expired",
        repo_url_hash=expired_key,
        docs_path="/tmp/docs/expired",
        created_at=now - timedelta(days=8),
        last_accessed=now - timedelta(days=8),
    )
    manager.cache_index[fresh_key] = CacheEntry(
        repo_url="https://github.com/example/fresh",
        repo_url_hash=fresh_key,
        docs_path="/tmp/docs/fresh",
        created_at=now - timedelta(days=1),
        last_accessed=now - timedelta(days=1),
    )

    manager.cleanup_expired_cache()

    assert expired_key not in manager.cache_index
    assert fresh_key in manager.cache_index


def test_cache_manager_repo_hash_is_stable_and_short(tmp_path):
    manager = CacheManager(cache_dir=str(tmp_path))

    first_hash = manager.get_repo_hash("https://github.com/example/repo")
    second_hash = manager.get_repo_hash("https://github.com/example/repo")

    assert first_hash == second_hash
    assert len(first_hash) == 16

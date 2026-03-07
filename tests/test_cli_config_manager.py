from pathlib import Path

from pytest import MonkeyPatch

from codewiki.cli.config_manager import ConfigManager


class _FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def test_configuration_to_dict_includes_fallback_model() -> None:
    from codewiki.cli.models.config import Configuration

    config = Configuration(
        base_url="http://example.com/v1",
        main_model="main-model",
        cluster_model="cluster-model",
        fallback_model="fallback-model",
    )

    assert config.to_dict()["fallback_model"] == "fallback-model"


def test_config_manager_persists_fallback_model(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    fake_keyring = _FakeKeyring()
    config_dir = tmp_path / ".codewiki"
    config_file = config_dir / "config.json"

    monkeypatch.setattr("codewiki.cli.config_manager.CONFIG_DIR", config_dir)
    monkeypatch.setattr("codewiki.cli.config_manager.CONFIG_FILE", config_file)
    monkeypatch.setattr(
        "codewiki.cli.config_manager.keyring.get_password", fake_keyring.get_password
    )
    monkeypatch.setattr(
        "codewiki.cli.config_manager.keyring.set_password", fake_keyring.set_password
    )
    monkeypatch.setattr(
        "codewiki.cli.config_manager.keyring.delete_password", fake_keyring.delete_password
    )

    manager = ConfigManager()
    manager.save(
        api_key="secret-api-key",
        base_url="http://example.com/v1",
        main_model="main-model",
        cluster_model="cluster-model",
        fallback_model="fallback-model",
    )

    reloaded = ConfigManager()
    assert reloaded.load() is True

    config = reloaded.get_config()
    assert config is not None
    assert config.fallback_model == "fallback-model"
    assert '"fallback_model": "fallback-model"' in config_file.read_text(encoding="utf-8")

import pytest

from localstack import config
from localstack.utils.analytics.metrics import (
    Counter,
    MultiLabelCounter,
    UnlabeledCounter,
    get_metric_registry,
)


@pytest.fixture(autouse=True)
def enable_analytics(monkeypatch):
    """Makes sure that all tests in this package are executed with analytics enabled."""
    monkeypatch.setattr(target=config, name="DISABLE_EVENTS", value=False)


@pytest.fixture(autouse=False)
def disable_analytics(monkeypatch):
    """Makes sure that all tests in this package are executed with analytics enabled."""
    monkeypatch.setattr(target=config, name="DISABLE_EVENTS", value=True)


@pytest.fixture(scope="function", autouse=True)
def reset_metric_registry() -> None:
    """Ensures each test starts with a fresh MetricRegistry."""
    registry = get_metric_registry()
    registry.registry.clear()  # Reset all registered metrics before each test


@pytest.fixture
def counter() -> UnlabeledCounter:
    return Counter(name="test_counter")


@pytest.fixture
def multilabel_counter() -> MultiLabelCounter:
    return Counter(name="test_multilabel_counter", labels=["status"])

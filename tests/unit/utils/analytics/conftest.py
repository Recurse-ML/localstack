import pytest

from localstack import config
from localstack.runtime.current import get_current_runtime, set_current_runtime
from localstack.utils.analytics.metrics import (
    Counter,
    _LabeledCounter,
    _SimpleCounter,
    get_metric_registry,
)


@pytest.fixture(autouse=True)
def enable_analytics(monkeypatch):
    """Makes sure that all tests in this package are executed with analytics enabled."""
    monkeypatch.setattr(target=config, name="DISABLE_EVENTS", value=False)


@pytest.fixture(scope="function", autouse=False)
def disable_analytics(monkeypatch):
    """Makes sure that all tests in this package are executed with analytics enabled."""
    monkeypatch.setattr(target=config, name="DISABLE_EVENTS", value=True)


@pytest.fixture(scope="function", autouse=True)
def reset_metric_registry() -> None:
    """Ensures each test starts with a fresh MetricRegistry."""
    registry = get_metric_registry()
    registry.registry.clear()  # Reset all registered metrics before each test


@pytest.fixture
def counter() -> _SimpleCounter:
    return Counter(name="test_counter")


@pytest.fixture
def labeled_counter() -> _LabeledCounter:
    return Counter(name="test_multilabel_counter", labels=["status"])


class MockComponents:
    name = "mock-product"


class MockRuntime:
    components = MockComponents()


@pytest.fixture(autouse=True)
def mock_runtime():
    try:
        # don't do anything if a runtime is set
        get_current_runtime()
        yield
    except ValueError:
        # set a mock runtime if no runtime is set
        set_current_runtime(MockRuntime())
        yield
        set_current_runtime(None)

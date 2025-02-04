from __future__ import annotations

import logging
import datetime
from collections import defaultdict
import threading
from typing import List, Tuple, Optional, Union, Dict

from localstack import config
from localstack.runtime import hooks
from localstack.utils.analytics import get_session_id
from localstack.utils.analytics.events import Event, EventMetadata
from localstack.utils.analytics.publisher import AnalyticsClientPublisher

LOG = logging.getLogger(__name__)

# Counters have to register with the registry
collector_registry: dict[str, Counter] = dict()

# TODO: introduce some base abstraction for the counters after gather some initial experience working with it
#  we could probably do intermediate aggregations over time to avoid unbounded counters for very long LS sessions
#  for now, we can recommend to use config.DISABLE_EVENTS=1


class MockCounter:
    """Mock implementation of the Counter class, used when events are disabled."""

    def labels(self, **kwargs: str) -> MockCounter:
        """Returns itself for chained calls, allowing no-op metric operations."""
        return self

    def inc(self, value: int = 1) -> None:
        """Ignores increment operations when events are disabled."""
        pass

    def reset(self, ) -> None:
        pass

    @staticmethod
    def collect() -> list[dict[str, int | str]]:
        """Returns an empty list since no metrics are collected in mock mode."""
        return []


class Counter:
    """
        A thread-safe counter for tracking metrics with optional labels.

        Supports both:
        - **Labeled counters** (via `.labels()`).
        - **Unlabeled counters** (direct `.inc()` calls).

        Attributes:
            _lock (threading.Lock): Ensures thread-safe operations.
            _values (defaultdict): Stores counter values, keyed by label tuples.
            _name (Optional[str]): The metric name.
            _full_name (str): The fully qualified metric name.
            _namespace (Optional[str]): The namespace for the metric.
            _labels (List[str]): The assigned label keys.
            _labels_origin (Dict[str, str]): Maps original label names to generic `label_X` keys.
    """
    _lock: threading.Lock
    _values: defaultdict[Tuple[Optional[str], ...], int]
    _name: Optional[str]
    _full_name: str
    _namespace: Optional[str]
    _labels: Optional[List[str]]
    _labels_origin: Dict[str, str]

    def __new__(cls, name: str = "", labels: Optional[List[str]] = None, namespace: str = "") -> Union[Counter, MockCounter]:
        """Returns a real or mock instance based on the `DISABLE_EVENTS` config."""
        if config.DISABLE_EVENTS:
            return MockCounter()
        return super(Counter, cls).__new__(cls)

    def __init__(self, name: str = "", labels: Optional[List[str]] = None, namespace: str = ""):
        """
        Initializes a counter.

        Args:
            name (str): The metric name.
            labels (Optional[List[str]]): List of labels (max 5). If not provided, the counter is unlabeled.
            namespace (str): The namespace for the metric.
        """
        self._lock = threading.Lock()
        self._values = defaultdict(int)

        if not name and not namespace:
            raise ValueError("Either 'name' or 'namespace' must be provided.")

        self._name = name.strip() if name else None
        self._namespace = namespace.strip() if namespace else None

        # Construct the full metric name
        self._full_name = "_".join(filter(None, [self._namespace, self._name])).strip(":")
        if not self._full_name:
            raise ValueError("Counter must have a valid name.")

        if labels:
            if len(labels) > 5:
                raise ValueError("A maximum of 5 labels are allowed.")
            self._labels_origin = {label_origin: f"label_{i + 1}" for i, label_origin in enumerate(labels or [])}
            self._labels = list(self._labels_origin.values())
        else:
            self._labels_origin = {}
            self._labels = []

        # Avoid duplicate counters in the registry
        if self._full_name in collector_registry:
            raise ValueError(f"Counter '{self._full_name}' already exists.")

        collector_registry[self._full_name] = self

    @property
    def full_name(self) -> str:
        """Returns the fully qualified metric name."""
        return self._full_name

    def labels(self, **kwargs: str) -> LabeledCounter:
        """
        Returns a metric instance for specific label values.

        Args:
            kwargs (str): A dictionary of label values (e.g., `status="error"`).

        Returns:
            LabeledCounter: A labeled metric instance.

        Raises:
            ValueError: If the counter does not support labels or incorrect labels are provided.
        """
        if not self._labels:
            raise ValueError("This counter does not support labels.")

        if len(kwargs) != len(self._labels_origin.keys()):
            raise ValueError(f"Expected labels {self._labels_origin.keys()}, got {list(kwargs.keys())}")

        labels = tuple(kwargs.get(label_key, None) for label_key in self._labels_origin.keys())
        return LabeledCounter(counter=self, labels=labels)

    def inc(self, value: int = 1, label_key: Optional[Tuple[Optional[str], ...]] = None) -> None:
        """
        Increments the counter.

        Args:
            value (int): The amount to increment (must be positive).
            label_key: Tuple of label values (only required for labeled counters).

        Raises:
            ValueError: If the value is not a positive number.
            ValueError: If incrementing a labeled counter without labels.
        """
        if value <= 0:
            raise ValueError("Metrics Counter: increment value must be positive.")

        if self._labels and label_key is None:
            raise ValueError("This counter requires labels, use .labels() instead.")

        # Use an empty tuple for non-labeled counters
        key = label_key if label_key is not None else ()

        with self._lock:
            self._values[key] += value

    def reset(self, label_key: Optional[Tuple[Optional[str], ...]] = None) -> None:
        """
        Resets the counter to zero.

        Args:
            label_key: Tuple of label values (only required for labeled counters).

        Raises:
            ValueError: If resetting a labeled counter without labels.
        """
        if self._labels and label_key is None:
            raise ValueError("This counter requires labels, use .labels() instead.")

        # Use an empty tuple for non-labeled counters
        key = label_key if label_key is not None else ()

        with self._lock:
            self._values[key] = 0

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        """
        Collects and returns metric data in a JSON-friendly format.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of collected metrics.
        """
        with self.get_lock():
            collected_data = []
            for labels, value in self._values.items():
                label_dict = dict(zip(self._labels, labels))

                # labels original names
                label_origin_dict = {f"origin_{i + 1}": orig for i, orig in enumerate(self._labels_origin.keys()) or {}}

                collected_data.append({
                    "name": self._full_name,
                    "value": value,
                    # Example: If labels=["service", "status"], and values=("sqs", "error"),
                    # it would generate: {"label_1": "sqs", "label_2": "error"}
                    **label_dict,
                    # Example: If labels=["service", "status"], it would generate:
                    # {"origin_1": "service", "origin_2": "status"}
                    **label_origin_dict
                })

            return collected_data

    def get_lock(self) -> threading.Lock:
        """Provides access to the internal lock."""
        return self._lock


class LabeledCounter:
    """
    A labeled instance of a `Counter`.

    This allows calling `.inc(value)` or `.reset()` on a specific set of label values.

    Attributes:
        _counter (Counter): Reference to the main counter.
        _labels (Tuple[Optional[str], ...]): The assigned label values.
    """
    _counter: Counter
    _labels: tuple[Optional[str], ...]

    def __init__(self, counter: Counter, labels: Tuple[Optional[str], ...]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: int = 1) -> None:
        """
        Increments the labeled counter.

        Args:
            value (int): The amount to increment (must be positive).

        Raises:
            ValueError: If the increment value is not positive.
        """
        self._counter.inc(label_key=self._labels, value=value)

    def reset(self) -> None:
        """Resets the labeled counter to zero."""
        self._counter.reset(label_key=self._labels)


def collect_metrics() -> List[Dict[str, Union[str, int]]]:
    """
    Collects usage metrics from all registered counters.

    Returns:
        List[Dict[str, Union[str, int]]]: A flat list of usage metrics.
    """
    collected_metrics = []
    for collector in collector_registry.values():
        collected_metrics.extend(collector.collect())
    return collected_metrics


@hooks.on_infra_shutdown()
def publish_metrics():
    """
    Collects all the registered metrics and immediately sends them to the analytics service.
    Skips execution if event tracking is disabled (`config.DISABLE_EVENTS`).

    This function is automatically triggered on infrastructure shutdown.
    """
    if config.DISABLE_EVENTS:
        return

    metadata = EventMetadata(
        session_id=get_session_id(),
        client_time=str(datetime.datetime.now()),
    )

    collected_metrics = collect_metrics()

    if collected_metrics:
        publisher = AnalyticsClientPublisher()
        publisher.publish([Event(name="metrics", metadata=metadata, payload=collected_metrics)])


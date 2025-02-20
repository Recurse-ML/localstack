from __future__ import annotations

import datetime
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, overload

from localstack import config
from localstack.runtime import hooks
from localstack.utils.analytics import get_session_id
from localstack.utils.analytics.events import Event, EventMetadata
from localstack.utils.analytics.publisher import AnalyticsClientPublisher

LOG = logging.getLogger(__name__)


class MetricRegistry:
    """
    A Singleton class responsible for managing all registered metrics.
    Provides methods for retrieving and collecting metrics.
    """

    _instance = None
    _mutex: threading.Lock = threading.Lock()

    def __new__(cls):
        # avoid locking if the instance already exist
        if cls._instance is None:
            with cls._mutex:
                # Prevents race conditions when multiple threads enter the first check simultaneously
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_registry"):
            self._registry = dict()

    @property
    def registry(self) -> Dict[str, "Metric"]:
        return self._registry

    def register(self, metric: Metric) -> None:
        """
        Registers a new metric.

        :param metric: The metric instance to register.
        :type metric: Metric
        :raises TypeError: If the provided metric is not an instance of `Metric`.
        :raises ValueError: If a metric with the same name already exists.
        """
        if not isinstance(metric, Metric):
            raise TypeError("Only subclasses of `Metric` can be registered.")

        if metric.full_name in self._registry:
            raise ValueError(f"Metric '{metric.full_name}' already exists.")

        self._registry[metric.full_name] = metric

    def collect(self) -> Dict[str, List[Dict[str, Union[str, int]]]]:
        """
        Collects all registered metrics.

        :return: A dictionary containing all collected metrics.
        :rtype: Dict[str, List[Dict[str, Union[str, int]]]]
        """
        return {
            "metrics": [
                metric
                for metric_instance in self._registry.values()
                for metric in metric_instance.collect()
            ]
        }


def get_metric_registry() -> MetricRegistry:
    return MetricRegistry()


class Metric(ABC):
    """
    Base class for all metrics (e.g., Counter, Gauge).

    Each subclass must implement the `collect()` method.
    """

    _full_name: str

    @property
    def full_name(self) -> str:
        """
        Retrieves the fully qualified metric name.

        :return: The full name of the metric.
        :rtype: str
        """
        return self._full_name

    @full_name.setter
    def full_name(self, value: str) -> None:
        """
        Validates and sets the full metric name.

        :param value: The fully qualified name to be set.
        :type value: str
        :raises ValueError: If the name is empty or invalid.
        """
        if not value or value.strip() == "":
            raise ValueError("Metric must have a valid name.")
        self._full_name = value

    @abstractmethod
    def collect(self) -> List[Dict[str, Union[str, int]]]:
        """
        Collects and returns metric data. Subclasses must implement this to return collected metric data.

        :return: A list of collected metrics.
        :rtype: List[Dict[str, Union[str, int]]]
        """
        pass


class _SimpleCounter(Metric):
    """
    A thread-safe counter for tracking occurrences of an event without labels.
    """

    _mutex: threading.Lock
    _namespace: Optional[str]
    _name: str
    _full_name: str
    _count: int

    @property
    def mutex(self) -> threading.Lock:
        """
        Provides thread-safe access to the internal lock.
        """
        return self._mutex

    def __init__(self, name: str, namespace: Optional[str] = ""):
        if not name:
            raise ValueError("Name is required and cannot be empty.")

        self._mutex = threading.Lock()
        self._name = name.strip()
        self._namespace = namespace.strip() if namespace else ""
        self._full_name = "_".join(filter(None, [self._namespace, self._name])).strip("_")
        self._count = 0
        get_metric_registry().register(self)

    def increment(self, value: int = 1) -> None:
        """Increments the counter unless events are disabled."""
        if config.DISABLE_EVENTS:
            return

        if value <= 0:
            raise ValueError("Increment value must be positive.")

        with self._mutex:
            self._count += value

    def reset(self) -> None:
        """Resets the counter to zero unless events are disabled."""
        if config.DISABLE_EVENTS:
            return

        with self._mutex:
            self._count = 0

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        """Collects the metric unless events are disabled."""
        if config.DISABLE_EVENTS:
            return list()

        with self._mutex:
            if (
                self._count == 0
            ):  # Entries with a count value of 0 will not be included in the final payload.
                return list()
            return [{"namespace": self._namespace, "name": self._full_name, "value": self._count}]


class _LabeledCounter(Metric):
    """
    A labeled counter that tracks occurrences of an event across different label combinations.
    """

    _count_by_labels: defaultdict[Tuple[str, ...], int]
    _label_values: tuple[Optional[str], ...]
    _labels: List[str]

    def __init__(self, name: str, labels: List[str] = list, namespace: Optional[str] = ""):
        if not name:
            raise ValueError("Name is required and cannot be empty.")

        if any(not label.strip() for label in labels):
            raise ValueError("Labels must be non-empty strings.")

        if len(labels) > 8:
            raise ValueError("A maximum of 8 labels are allowed.")

        self._mutex = threading.Lock()
        self._name = name.strip()
        self._namespace = namespace.strip() if namespace else ""
        self._full_name = "_".join(filter(None, [self._namespace, self._name])).strip("_")

        self._labels = labels
        self._label_values = tuple()
        self._count_by_labels = defaultdict(int)

        get_metric_registry().register(self)

    @property
    def mutex(self) -> threading.Lock:
        """
        Provides thread-safe access to the internal lock.
        """
        return self._mutex

    @property
    def count_by_labels(self) -> defaultdict[Tuple[str, ...], int]:
        return self._count_by_labels

    def labels(self, **kwargs: str) -> _LabeledCounterProxy:
        """
        Assign label values to the counter instance.

        :param kwargs:
            Key-value pairs representing labels and their respective values.

        :return:
            The updated instance of `MultiLabelCounter` with assigned labels.

        :rtype: MultiLabelCounter

        :raises ValueError:
            - If the number of provided labels does not match the expected count.
            - If any of the provided labels are empty strings.
        """
        self._label_values = tuple(label_value.strip() for label_value in kwargs.values())

        if len(kwargs) != len(self._label_values):
            raise ValueError(f"Expected labels {self._label_values}, got {list(kwargs.values())}")

        if any(not label for label in self._label_values):
            raise ValueError("Label values must be non-empty strings.")

        return _LabeledCounterProxy(counter=self, label_values=self._label_values)

    def _as_list(self) -> List[Dict[str, Union[str, int]]]:
        num_labels = len(self._labels)

        static_key_label_value = [f"label_{i + 1}_value" for i in range(num_labels)]
        static_key_label = [f"label_{i + 1}" for i in range(num_labels)]

        collected_metrics = []

        for label_values, count in self._count_by_labels.items():
            if count == 0:
                continue  # entries with a count value of 0 will not be included in the list.

            if len(label_values) != num_labels:
                raise ValueError(
                    f"Label count mismatch: expected {num_labels} labels {self._labels}, "
                    f"but got {len(label_values)} values {label_values}."
                )

            collected_metrics.append(
                {
                    "namespace": self._namespace,
                    "name": self._full_name,
                    "metric_value": count,
                    **dict(zip(static_key_label_value, label_values)),
                    **dict(zip(static_key_label, self._labels)),
                }
            )

        return collected_metrics

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        if config.DISABLE_EVENTS:
            return list()

        with self._mutex:
            return self._as_list()


class _LabeledCounterProxy:
    """A proxy for a labeled counter, enforcing scoped label values."""

    def __init__(self, counter: _LabeledCounter, label_values: Tuple[str, ...]):
        self._counter = counter
        self._label_values = label_values

    def increment(self, value: int = 1) -> None:
        """Increments the counter for the assigned labels unless events are disabled."""
        if config.DISABLE_EVENTS:
            return

        if value <= 0:
            raise ValueError("Increment value must be positive.")

        with self._counter.mutex:
            self._counter.count_by_labels[self._label_values] += value

    def reset(self) -> None:
        """Resets the counter to zero for the assigned labels unless events are disabled."""
        if config.DISABLE_EVENTS:
            return

        with self._counter.mutex:
            self._counter.count_by_labels[self._label_values] = 0


class Counter:
    """
    A flexible entry point that returns the appropriate counter type.
    """

    @overload
    def __new__(cls, name: str, namespace: Optional[str] = "") -> _SimpleCounter:
        return _SimpleCounter(namespace=namespace, name=name)

    @overload
    def __new__(
        cls, name: str, labels: List[str], namespace: Optional[str] = ""
    ) -> _LabeledCounter:
        return _LabeledCounter(namespace=namespace, name=name, labels=labels)

    def __new__(
        cls, name: str, namespace: Optional[str] = "", labels: Optional[List[str]] = None
    ) -> Union[_SimpleCounter, _LabeledCounter]:
        if labels:
            return _LabeledCounter(namespace=namespace, name=name, labels=labels)
        return _SimpleCounter(namespace=namespace, name=name)


@hooks.on_infra_shutdown()
def publish_metrics():
    """
    Collects all the registered metrics and immediately sends them to the analytics service.
    Skips execution if event tracking is disabled (`config.DISABLE_EVENTS`).

    This function is automatically triggered on infrastructure shutdown.
    """
    if config.DISABLE_EVENTS:
        return

    collected_metrics = get_metric_registry().collect()
    if not collected_metrics["metrics"]:  # Skip publishing if no metrics remain after filtering
        return

    metadata = EventMetadata(
        session_id=get_session_id(),
        client_time=str(datetime.datetime.now()),
    )

    if collected_metrics:
        publisher = AnalyticsClientPublisher()
        publisher.publish([Event(name="ls_metrics", metadata=metadata, payload=collected_metrics)])

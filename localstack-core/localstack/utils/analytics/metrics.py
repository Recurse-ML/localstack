from __future__ import annotations

import datetime
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from localstack import config
from localstack.runtime import hooks
from localstack.utils.analytics import get_session_id
from localstack.utils.analytics.events import Event, EventMetadata
from localstack.utils.analytics.publisher import AnalyticsClientPublisher

LOG = logging.getLogger(__name__)


class MetricRegistry:
    """
    A Singleton class responsible for managing all registered metrics.

    - Stores references to `Metric` instances.
    - Provides methods for retrieving and collecting metrics.
    """

    _instance: Optional[MetricRegistry] = None  # Singleton instance
    _registry: Dict[str, Metric]

    @property
    def registry(self) -> Dict[str, Metric]:
        return self._registry

    def __new__(cls) -> MetricRegistry:
        """
        Ensures that only one instance of `MetricRegistry` exists.

        :return: A instance of `MetricRegistry`.
        """
        if not cls._instance:
            cls._instance = super(MetricRegistry, cls).__new__(cls)
            cls._instance._registry = dict()  # Registry initialized here
        return cls._instance

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
    """
    Retrieves the `MetricRegistry` instance.

    :return: The `MetricRegistry` instance.
    :rtype: MetricRegistry
    """
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


class BaseCounter(Metric):
    """
    A base class for counters (both labeled and unlabeled).
    Provides common functionalities such as locking, initialization, and registration.
    """

    _mutex: threading.Lock
    _namespace: Optional[str]
    _name: str
    _full_name: str

    @property
    def mutex(self) -> threading.Lock:
        """
        Provides thread-safe access to the internal lock.

        :return: The threading lock used for synchronization.
        :rtype: threading.Lock
        """
        return self._mutex

    def __init__(self, name: str = "", namespace: Optional[str] = ""):
        """
        Initializes a counter with a name and an optional namespace.

        :param name: The metric name.
        :type name: str
        :param namespace: An optional prefix that is prepended to the metric name to provide logical grouping.
        If set, the full metric name will follow the format: `namespace_name`
        :type namespace: Optional[str]
        :raises ValueError: If `name` is None or empty string.
        """
        self._mutex = threading.Lock()

        if not name:
            raise ValueError("name is required and cannot be empty")

        self._name = name.strip()
        self._namespace = namespace.strip() if namespace else ""

        # Construct the full metric name
        self.full_name = "_".join(filter(None, [self._namespace, self._name])).strip("_")

    @abstractmethod
    def increment(self, value: int = 1) -> None:
        """
        Increments the counter.

        :param value: The amount to increment (must be positive).
        :type value: int
        :raises ValueError: If the increment value is not positive.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the counter to zero."""
        pass

    @abstractmethod
    def collect(self) -> List[Dict[str, Union[str, int]]]:
        """
        Collects and returns metric data in a JSON-friendly format.

        :return: A list of collected metrics.
        :rtype: List[Dict[str, Union[str, int]]]
        """
        pass


class MockCounter(BaseCounter):
    """Mock implementation of the Counter class, used when events are disabled."""

    def __init__(self, name: str = "counter", namespace: Optional[str] = "mock"):
        super().__init__(name=name, namespace=namespace)

    def labels(self, **kwargs: str) -> MockCounter:
        """Returns itself for chained calls, allowing no-op metric operations."""
        return self

    def increment(self, value: int = 1) -> None:
        """Ignores increment operations when events are disabled."""
        pass

    def reset(
        self,
    ) -> None:
        pass

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        """Returns an empty list since no metrics are collected in mock mode."""
        return list()


class UnlabeledCounter(BaseCounter):
    """
    A thread-safe counter for tracking occurrences of an event without labels.
    """

    _count: int

    def __init__(self, name: str = "", namespace: Optional[str] = ""):
        """
        Initializes an unlabeled counter.

        :param name: The metric name.
        :type name: str
        :param namespace: An optional prefix that is prepended to the metric name to provide logical grouping.
        If set, the full metric name will follow the format: `namespace_name`
        :type namespace: str
        """
        super().__init__(name=name, namespace=namespace)
        self._count = 0
        get_metric_registry().register(self)

    def increment(self, value: int = 1) -> None:
        if value <= 0:
            raise ValueError("Increment value must be positive.")
        with self._mutex:
            self._count += value

    def reset(self) -> None:
        with self._mutex:
            self._count = 0

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        with self._mutex:
            if self._count == 0:  # Exclude zero-value metrics
                return []
            return [{"namespace": self._namespace, "name": self._full_name, "value": self._count}]


class MultiLabelCounter(BaseCounter):
    """
    A labeled counter that tracks occurrences of an event across different label combinations.
    """

    metric_value_by_label: defaultdict[Tuple[str, ...], int]
    _label_values: tuple[Optional[str], ...]
    _labels: List[str]

    def __init__(self, name: str = "", labels: List[str] = list, namespace: Optional[str] = ""):
        """
        Initializes a labeled counter.

        :param name: The metric name.
        :type name: str
        :param labels: List of label names.
        :type labels: List[str]
        :param namespace: The prefix for the metric.
        :type namespace: Optional[str]
        :raises ValueError: If more than 5 labels are provided.
        """
        super().__init__(name=name, namespace=namespace)

        if any(not label.strip() for label in labels):
            raise ValueError("Labels must be non-empty strings.")

        if len(labels) > 5:
            raise ValueError("A maximum of 5 labels are allowed.")

        self._metric_value_by_label = defaultdict(int)
        self._labels = labels
        self._label_values = tuple()

        get_metric_registry().register(self)

    def labels(self, **kwargs: str) -> MultiLabelCounter:
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

        return self

    def increment(self, value: int = 1) -> None:
        if value <= 0:
            raise ValueError("Increment value must be positive.")

        if not self._label_values:
            raise ValueError("Labels must be set using `.labels()` before incrementing.")

        with self._mutex:
            self._metric_value_by_label[self._label_values] += value

    def reset(self) -> None:
        if self._label_values is None:
            raise ValueError("Labels must be set using .labels() before resetting.")

        with self._mutex:
            self._metric_value_by_label[self._label_values] = 0

    def as_dict_list(self) -> List[Dict[str, Union[str, int]]]:
        num_labels = len(self._labels)

        static_key_label_value = [f"label_{i + 1}_value" for i in range(num_labels)]
        static_key_label = [f"label_{i + 1}" for i in range(num_labels)]

        collected_metrics = []

        for label_values, metric_value in self._metric_value_by_label.items():
            if metric_value == 0:
                continue  # Exclude zero-value metrics

            if len(label_values) != num_labels:
                raise ValueError(
                    f"Label count mismatch: expected {num_labels} labels {self._labels}, "
                    f"but got {len(label_values)} values {label_values}."
                )

            collected_metrics.append(
                {
                    "namespace": self._namespace,
                    "name": self._full_name,
                    "metric_value": metric_value,
                    **dict(
                        zip(static_key_label_value, label_values)
                    ),  # maps 'label_1_value': 'label_value'
                    **dict(zip(static_key_label, self._labels)),  # maps 'label_1': 'label'
                }
            )

        return collected_metrics

    def collect(self) -> List[Dict[str, Union[str, int]]]:
        with self._mutex:
            return self.as_dict_list()


class Counter:
    """
    A flexible entry point for creating counter instances.
    """

    def __new__(
        cls, name: str = "", namespace: Optional[str] = "", labels: Optional[List[str]] = None
    ):
        if config.DISABLE_EVENTS:
            return MockCounter()
        else:
            if labels:
                return MultiLabelCounter(namespace=namespace, name=name, labels=labels)
            return UnlabeledCounter(namespace=namespace, name=name)


@hooks.on_infra_start()
def initialize_metric_registry() -> None:
    """Initializes the `CollectorRegistry` at infrastructure startup."""
    MetricRegistry()


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

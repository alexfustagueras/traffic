"""
Prediction algorithms for flight trajectory forecasting.

This module provides a registry-based system for trajectory prediction methods,
allowing for easy extension and maintenance of prediction algorithms.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from ...core.flight import Flight
from ...core.traffic import Traffic

logger = logging.getLogger(__name__)


class PredictorBase(ABC):
    """Base class for all trajectory prediction methods.

    All prediction algorithms should inherit from this class and implement
    the predict method. Predictors are automatically registered when imported.
    """

    # Class attribute to store method name
    method_name: str = ""

    def __init_subclass__(cls, **kwargs):
        """Automatically register predictor subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.method_name:
            PredictionRegistry.register(cls.method_name, cls)

    @abstractmethod
    def predict(self, flight: Flight) -> Union[Flight, Traffic]:
        """Generate trajectory predictions for a flight.

        Args:
            flight: Flight object with historical trajectory data

        Returns:
            Flight object with predicted trajectory if n_samples=1,
            or Traffic object containing multiple predicted trajectories
        """
        pass


class PredictionRegistry:
    """Registry for prediction methods.

    Provides centralized management of prediction algorithms, allowing
    dynamic registration and discovery of prediction methods.
    """

    _registry: Dict[str, Type[PredictorBase]] = {}

    @classmethod
    def register(
        cls, method_name: str, predictor_class: Type[PredictorBase]
    ) -> None:
        """Register a prediction method.

        Args:
            method_name: Name of the prediction method
            predictor_class: Predictor class to register
        """
        if not issubclass(predictor_class, PredictorBase):
            raise TypeError(
                f"Predictor class {predictor_class} must inherit from PredictorBase"
            )

        cls._registry[method_name] = predictor_class
        logger.debug(f"Registered prediction method: {method_name}")

    @classmethod
    def get_predictor_class(cls, method_name: str) -> Type[PredictorBase]:
        """Get predictor class for a method name.

        Args:
            method_name: Name of the prediction method

        Returns:
            Predictor class

        Raises:
            ValueError: If method is not registered
        """
        if method_name not in cls._registry:
            available_methods = list(cls._registry.keys())
            raise ValueError(
                f"Unknown prediction method: {method_name}. "
                f"Available methods: {available_methods}"
            )
        return cls._registry[method_name]

    @classmethod
    def create_predictor(cls, method_name: str, **kwargs) -> PredictorBase:
        """Create a predictor instance.

        Args:
            method_name: Name of the prediction method
            **kwargs: Parameters to pass to the predictor constructor

        Returns:
            Predictor instance
        """
        predictor_class = cls.get_predictor_class(method_name)
        try:
            return predictor_class(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create predictor '{method_name}': {e}"
            ) from e

    @classmethod
    def list_methods(cls) -> list[str]:
        """List all registered prediction methods."""
        return list(cls._registry.keys())


# Import existing predictors to register them
try:
    from .bn import BNPredict
    from .cfm import CFMPredict
    from .cfm_intent import CFMIntentPredict
    from .flightplan import FlightPlanPredict
    from .linearextrapolation import LinearExtrapolation
except ImportError as e:
    logger.warning(f"Some prediction modules could not be imported: {e}")
    # Don't fail if optional prediction modules aren't available


# Backward compatibility: keep the old protocol for now
from typing import Protocol, runtime_checkable


@runtime_checkable
class PredictBase(Protocol):
    def predict(self, flight: Flight) -> Union[Flight, Traffic]: ...

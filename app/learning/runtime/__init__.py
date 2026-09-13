"""
Continuous production learning runtime.

The runtime is designed to remain active continuously while
training only when eligible client data is available.
"""

from .data_gateway import ClientDataGateway
from .runtime_service import LearningRuntimeService
from .training_trigger import TrainingTrigger

__all__ = [
    "ClientDataGateway",
    "LearningRuntimeService",
    "TrainingTrigger",
]
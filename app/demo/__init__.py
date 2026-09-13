"""
Client Demonstration Layer.

This package provides a client-facing demonstration over the
existing Restaurant Demand AI intelligence stack.

Part A deliberately does not create or train a new ML model.
It consumes the existing synthetic historical dataset and produces
a production-style commercial intelligence output.
"""

from .schemas import ClientDemoRequest, ClientDemoResult
from .service import ClientDemoService

__all__ = [
    "ClientDemoRequest",
    "ClientDemoResult",
    "ClientDemoService",
]
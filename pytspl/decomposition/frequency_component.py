"""Enum class for the types of frequency components.

The frequency components are:
- Harmonic
- Curl
- Gradient
"""

from enum import Enum


class FrequencyComponent(Enum):
    """The frequency components that can be extracted from a
    edge flows."""

    HARMONIC = "harmonic"
    CURL = "curl"
    GRADIENT = "gradient"

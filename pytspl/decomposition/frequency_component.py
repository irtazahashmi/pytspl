"""Enum class for the types of frequency components.

The frequency components are:
- Gradient
- Curl
- Harmonic
"""

from enum import Enum


class FrequencyComponent(Enum):
    """The frequency components that can be extracted from
    edge flows."""

    GRADIENT = "gradient"
    CURL = "curl"
    HARMONIC = "harmonic"

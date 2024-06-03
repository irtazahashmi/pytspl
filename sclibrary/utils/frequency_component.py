"""Enum for the types of frequency components.

The frequency components are:
- Harmonic
- Curl
- Gradient
"""

from enum import Enum


class FrequencyComponent(Enum):
    """Enum for the types of frequency components."""

    HARMONIC = "harmonic"
    CURL = "curl"
    GRADIENT = "gradient"

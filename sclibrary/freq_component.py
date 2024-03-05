from enum import Enum


class FrequencyComponent(Enum):
    """
    Enum for different types of frequency components.
    """

    HARMONIC = "harmonic"
    CURL = "curl"
    GRADIENT = "gradient"

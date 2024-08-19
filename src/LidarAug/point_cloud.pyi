from enum import Enum


class IntensityRange(Enum):
    """
    Defines options for maximum intensity values.
    Intensity goes from [0; MAX_INTENSITY], where MAX_INTENSITY is either 1 or
    255.
    """
    MAX_INTENSITY_1: int
    MAX_INTENSITY_255: int

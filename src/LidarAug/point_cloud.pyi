from enum import Enum


class IntensityRange(Enum):
    """
    Defines options for maximum intensity values.
    Intensity goes from [0; MAX_INTENSITY], where MAX_INTENSITY is either 1 or
    255.
    """
    MAX_INTENSITY_1: int
    MAX_INTENSITY_255: int


def set_max_intensity(val: IntensityRange) -> None:
    """
    Set the global state tracker for the maximum intensity.

    :param val: is the new maximum intensity (member of `IntensityRange`).
    """
    ...


def get_max_intensity() -> int:
    """
    Get the current value of the maximum intensity global state tracker.

    :return: an int representing the maximum intensity value.
    """
    ...

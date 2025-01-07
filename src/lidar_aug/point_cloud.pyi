import typing
from enum import Enum

__all__ = ['IntensityRange', 'get_max_intensity', 'set_max_intensity']


class IntensityRange(Enum):
    """
    Defines options for maximum intensity values.
    Intensity goes from [0; MAX_INTENSITY], where MAX_INTENSITY is either 1 or 255.

    Members:

      MAX_INTENSITY_1

      MAX_INTENSITY_255
    """
    MAX_INTENSITY_1: typing.ClassVar[
        IntensityRange]  # value = <IntensityRange.MAX_INTENSITY_1: 1>
    MAX_INTENSITY_255: typing.ClassVar[
        IntensityRange]  # value = <IntensityRange.MAX_INTENSITY_255: 255>
    __members__: typing.ClassVar[dict[
        str,
        IntensityRange]]  # value = {'MAX_INTENSITY_1': <IntensityRange.MAX_INTENSITY_1: 1>, 'MAX_INTENSITY_255': <IntensityRange.MAX_INTENSITY_255: 255>}

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: int) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


def get_max_intensity() -> int:
    """
    Get the current value of the maximum intensity global state tracker.

    :return: an int representing the maximum intensity value.
    """
    ...


def set_max_intensity(val: IntensityRange) -> None:
    """
    Set the global state tracker for the maximum intensity.

    :param val: is the new maximum intensity (member of `IntensityRange`).
    """
    ...

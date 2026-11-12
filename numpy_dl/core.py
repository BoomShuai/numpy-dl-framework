"""Minimal explicit parameters; this is not an automatic differentiation engine."""
from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class Parameter:
    data: np.ndarray
    grad: object = None


class Module:
    def parameters(self):
        return {}

    def buffers(self):
        return {}

    def zero_grad(self):
        for parameter in self.parameters().values():
            parameter.grad = None


def pair(value, name, minimum=1):
    values = (value, value) if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, (tuple, list)) or len(values) != 2 or any(type(v) is not int or v < minimum for v in values):
        raise ValueError(name + " must contain two valid integers")
    return tuple(values)


def floating(value, ndim=None):
    value = np.asarray(value)
    if value.dtype.kind != "f" or value.size == 0 or ndim is not None and value.ndim != ndim:
        raise ValueError("Expected a non-empty floating point array with valid dimensions")
    return value

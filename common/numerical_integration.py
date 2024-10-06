import abc
from functools import singledispatchmethod
from typing import Iterable, TypeVar, Generic
import numpy as np


OutputType = TypeVar("OutputType", float, np.array)
"""
Valid output types for models.
"""


class Model(abc.ABC, Generic[OutputType]):
    """
    Represents a model with numerical outputs.
    """

    @abc.abstractmethod
    def difference_eq(self, prior_output: OutputType) -> OutputType:
        """
        Calculates the value of the model's difference equation.

        Args:
            prior_output: The previous output of the model.

        Returns:
            The estimated next output, divided by the timestep.

        """


class ModelSolver(abc.ABC, Generic[OutputType]):
    """
    Solves a model using a numerical method.
    """

    def __init__(self, model: Model[OutputType], *, initial_value: OutputType):
        """
        Args:
            model: The model to solve.
            initial_value: The initial value of the model.

        """
        self._model = model
        self._initial_value = initial_value

    @singledispatchmethod
    def _copy(self, x: np.array) -> OutputType:
        """
        Returns a copy of the input if necessary. This can be important
        because Numpy arrays are modified in-place by default, which can
        cause issues if we are using one to represent the model output at
        each timestep.

        Args:
            x: The input.

        Returns:
            A copy of the input if needed to avoid modifying the original.

        """
        return x.copy()

    @_copy.register
    def _(self, x: float) -> float:
        return x

    @abc.abstractmethod
    def outputs(self, timestep: float) -> Iterable[OutputType]:
        """
        Computes the output of the model at each time.

        Args:
            timestep: The time delta between each output.

        Yields:
            The value of each output, starting from time zero.

        """


class EulerSolver(ModelSolver):
    """
    Solver that uses the Euler method of approximation.
    """

    def outputs(self, timestep: float) -> Iterable[OutputType]:
        output = self._initial_value

        while True:
            output = self._copy(output)
            yield output

            output += self._model.difference_eq(output) * timestep


class ImprovedEulerSolver(ModelSolver):
    """
    Solver that uses the improved Euler method of approximation.
    """

    def outputs(self, timestep: float) -> Iterable[OutputType]:
        output = self._initial_value

        while True:
            output = self._copy(output)
            yield output

            k1 = self._model.difference_eq(output) * timestep
            k2 = self._model.difference_eq(output + k1) * timestep
            output += (k1 + k2) / 2


class RK4Solver(ModelSolver):
    """
    Solver that uses the Runge-Kutta 4 method of approximation.
    """

    def outputs(self, timestep: float) -> Iterable[OutputType]:
        output = self._initial_value

        while True:
            output = self._copy(output)
            yield output

            k1 = self._model.difference_eq(output) * timestep
            k2 = self._model.difference_eq(output + k1 / 2.0) * timestep
            k3 = self._model.difference_eq(output + k2 / 2.0) * timestep
            k4 = self._model.difference_eq(output + k3) * timestep
            output += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

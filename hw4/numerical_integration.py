import abc
import math
from shutil import which
from typing import Iterable
from itertools import islice

from tabulate import tabulate


class Model(abc.ABC):
    """
    Represents a model with a single output.
    """

    @abc.abstractmethod
    def outputs(self, timestep: float) -> Iterable[float]:
        """
        Computes the output of the model at each time.

        Args:
            timestep: The time delta between each output.

        Yields:
            The value of each output, starting from time zero.

        """

    @abc.abstractmethod
    def difference_eq(self, prior_output: float) -> float:
        """
        Calculates the value of the model's difference equation.

        Args:
            prior_output: The previous output of the model.

        Returns:
            The estimated next output, divided by the timestep.

        """


class ExponentialGrowthModel(Model):
    """
    Models the exponential growth of a population.
    """

    def __init__(self, *, initial_size: int, growth_rate: float):
        """
        Args:
            initial_size: The initial size of the population.
            growth_rate: The growth rate of the population.

        """
        self.__initial_size = initial_size
        self.__growth_rate = growth_rate

    def outputs(self, timestep: float) -> Iterable[float]:
        t = 0.0
        while True:
            yield self.__initial_size * math.exp(self.__growth_rate * t)
            t += timestep

    def difference_eq(self, prior_output: float) -> float:
        return self.__growth_rate * prior_output


class EulerModel(Model):
    """
    Model that uses the Euler method of approximation.
    """

    def __init__(self, model: Model, *, initial_size: int):
        """
        Args:
            model: The model to approximate.
            initial_size: The initial size of the population.

        """
        self.__model = model
        self.__initial_size = initial_size

    def outputs(self, timestep: float) -> Iterable[float]:
        output = self.__initial_size

        while True:
            yield output
            output += self.__model.difference_eq(output) * timestep

    def difference_eq(self, prior_output: float) -> float:
        return self.__model.difference_eq(prior_output)


class ImprovedEulerModel(Model):
    """
    Model that uses the improved Euler method of approximation.
    """

    def __init__(self, model: Model, *, initial_size: int):
        """
        Args:
            model: The model to approximate.
            initial_size: The initial size of the population.

        """
        self.__model = model
        self.__initial_size = initial_size

    def outputs(self, timestep: float) -> Iterable[float]:
        output = self.__initial_size

        while True:
            yield output

            k1 = self.__model.difference_eq(output) * timestep
            k2 = self.__model.difference_eq(output + k1) * timestep
            output += (k1 + k2) / 2

    def difference_eq(self, prior_output: float) -> float:
        return self.__model.difference_eq(prior_output)


class RK4Model(Model):
    """
    Model that uses the Runge-Kutta 4 method of approximation.
    """

    def __init__(self, model: Model, *, initial_size: int):
        """
        Args:
            model: The model to approximate.
            initial_size: The initial size of the population.

        """
        self.__model = model
        self.__initial_size = initial_size

    def outputs(self, timestep: float) -> Iterable[float]:
        output = self.__initial_size

        while True:
            yield output

            k1 = self.__model.difference_eq(output) * timestep
            k2 = self.__model.difference_eq(output + k1 / 2) * timestep
            k3 = self.__model.difference_eq(output + k2 / 2) * timestep
            k4 = self.__model.difference_eq(output + k3) * timestep
            output += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def difference_eq(self, prior_output: float) -> float:
        return self.__model.difference_eq(prior_output)


def main():
    n_0 = 10
    k = 0.1
    dt = 0.1

    analytical = ExponentialGrowthModel(initial_size=n_0, growth_rate=k)
    euler = EulerModel(analytical, initial_size=n_0)
    improved_euler = ImprovedEulerModel(analytical, initial_size=n_0)
    rk4 = RK4Model(analytical, initial_size=n_0)
    models = [analytical, euler, improved_euler, rk4]

    # Get the output iterators for a limited number of steps.
    end_step = int(5 / dt) + 1
    start_step = max(end_step - 100, 0)
    models = [islice(m.outputs(dt), start_step, end_step) for m in models]

    # Compute population sizes.
    table = []
    for t, outputs in enumerate(zip(*models)):
        table.append([t * dt] + list(outputs))
    print(
        tabulate(
            table, headers=["Time", "Analytical", "Euler", "RK2", "RK4"], floatfmt=".4f"
        )
    )


if __name__ == "__main__":
    main()

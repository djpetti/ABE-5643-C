import math
from itertools import islice
from typing import Iterable

from tabulate import tabulate

from common.numerical_integration import (
    EulerSolver,
    ImprovedEulerSolver,
    RK4Solver, Model
)


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


def main():
    n_0 = 10
    k = 0.1
    dt = 0.1

    analytical = ExponentialGrowthModel(initial_size=n_0, growth_rate=k)
    euler = EulerSolver(analytical, initial_value=n_0)
    improved_euler = ImprovedEulerSolver(analytical, initial_value=n_0)
    rk4 = RK4Solver(analytical, initial_value=n_0)
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

from itertools import islice
from typing import Iterable

import pandas as pd
import numpy as np
from tabulate import tabulate


class AgeClassSimulation:
    """
    Implements an age-class simulation.
    """

    def __init__(self, life_table: pd.DataFrame):
        """
        Args:
            life_table: The life table for the organism being simulated. It
                is assumed to be indexed by age-class. It must have the
                following columns:
                - survival_rate: Fraction of organisms that survive to the next
                    age class.
                - birth_rate: Per-capita birth rate for that age class.
        """
        self.__life_table = life_table

    def simulate(self, *, initial_population: np.array) -> Iterable[np.array]:
        """
        Iteratively simulates the population over time.

        Args:
            initial_population: The initial population for each age-class.

        Yields:
            The population for each age-class at that timestep.

        """
        population = initial_population.copy()

        while True:
            yield population

            # Figure out how many survive.
            survivors = population * self.__life_table.survival_rate
            # Figure out how many are born.
            births = population @ self.__life_table.birth_rate
            # We can't have part of an organism...
            survivors = survivors.to_numpy().astype(int)
            births = int(births)

            # Everything now moves up an age-class.
            population = np.roll(survivors, 1)
            population[0] = births


def main() -> None:
    # Create the life-table.
    life_table = pd.DataFrame(
        {
            "survival_rate": [0.1104, 0.1042, 0.1391, 0.1250, 0.0],
            "birth_rate": [0.0, 3.13, 42.53, 100.98, 118.75],
        },
        index=[0, 1, 2, 3, 4],
    )
    initial_population = np.array([10000, 1104, 115, 16, 2])

    simulation = AgeClassSimulation(life_table)

    pop_table = [
        [i] + p.tolist()
        for i, p in enumerate(
            islice(simulation.simulate(initial_population=initial_population), 13)
        )
    ]
    print(tabulate(pop_table, headers=["Year", "0", "1", "2", "3", "4"]))


if __name__ == "__main__":
    main()

import math

from tabulate import tabulate


class ExponentialGrowthModel:
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

    def population_at_time(self, t: float) -> float:
        """
        Uses the exponential growth model to calculate the population size at a
        given time.

        Args:
            t: The time to get the population size at.

        Returns:
            The estimated population size.

        """
        return self.__initial_size * math.exp(self.__growth_rate * t)


def main():
    model = ExponentialGrowthModel(initial_size=2, growth_rate=0.092)

    # Compute population sizes.
    table = []
    for t in range(51):
        table.append([t, model.population_at_time(t)])
    print(tabulate(table, headers=["Time", "Population"]))


if __name__ == "__main__":
    main()

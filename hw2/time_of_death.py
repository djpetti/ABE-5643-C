import math
from tabulate import tabulate


class CoolingModel:
    """
    Model that implements Newton's law of cooling.
    """

    def __init__(self, *, initial_temp: float, environment_temp: float, k: float):
        """
        Args:
            initial_temp: The initial temperature of the object.
            environment_temp: The temperature of the environment.
            k: The cooling rate.

        """
        self.__initial_temp = initial_temp
        self.__environment_temp = environment_temp
        self.__k = k

    def temp_at_time(self, t: float) -> float:
        """
        Returns the temperature of the object at time t.

        Args:
            t: The time.

        Returns:
            The temperature of the object at time t.

        """
        return self.__environment_temp + (
            self.__initial_temp - self.__environment_temp
        ) * math.exp(-self.__k * t)


def main() -> None:
    # Average low temperature in Gainesville in January.
    environment_temp = 8.89
    model = CoolingModel(
        initial_temp=37.0, environment_temp=environment_temp, k=0.06
    )

    # Simulate for a 24-hour period.
    table = []
    for t in range(25):
        table.append([t, model.temp_at_time(t)])
    print(tabulate(table))


if __name__ == "__main__":
    main()
from itertools import islice

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from common.numerical_integration import Model, RK4Solver


sns.set_theme()


class LotkaVolterraModel(Model[np.array]):
    """
    Implements the Lotka-Volterra model.
    """

    def __init__(
        self,
        *,
        prey_growth_rate: float,
        predation_efficiency: float,
        conversion_efficiency: float,
        death_rate: float,
    ):
        self.__prey_growth = prey_growth_rate
        self.__predation_efficiency = predation_efficiency
        self.__conversion_efficiency = conversion_efficiency
        self.__predator_death = death_rate

    @property
    def prey_growth_rate(self) -> float:
        return self.__prey_growth

    @property
    def predation_efficiency(self) -> float:
        return self.__predation_efficiency

    @property
    def conversion_efficiency(self) -> float:
        return self.__conversion_efficiency

    @property
    def death_rate(self) -> float:
        return self.__predator_death

    def difference_eq(self, prior_output: np.array) -> np.array:
        """
        Args:
            prior_output: An array of `[num prey, num predators]`

        Returns:
            The same array for the next timestep.

        """
        prey, predator = prior_output
        return np.array(
            [
                self.__prey_growth * prey
                - self.__predation_efficiency * prey * predator,
                self.__conversion_efficiency * prey * predator
                - self.__predator_death * predator,
            ]
        )


def _plot(results: pd.DataFrame) -> None:
    """
    Plots the predator-prey numbers over time.

    Args:
        results: A dataframe containing the model output data.

    """
    # Set the figure size
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot num_predators on the first y-axis (left side)
    sns.lineplot(
        results, x="time", y="predators", label="Predators", color="red", ax=ax1
    )
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Number of Predators", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Create a second y-axis (right side) for num_prey
    ax2 = ax1.twinx()
    sns.lineplot(results, x="time", y="prey", label="Prey", color="blue", ax=ax2)
    ax2.set_ylabel("Number of Prey", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Set the title
    plt.title("Predator vs Prey Population Over Time")

    # Add legends (manual, since Seaborn’s legend doesn't automatically support twin axes)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Display the plot
    plt.tight_layout()
    plt.show()


def _plot_densities(results: pd.DataFrame, model: LotkaVolterraModel) -> None:
    """
    Plots the densities of the two populations against each-other.

    Args:
        results: A dataframe containing the model output data.
        model: The original model.

    """
    # Set the figure size
    plt.figure(figsize=(10, 6))

    sns.scatterplot(results, x="predators", y="prey", label="Density")
    plt.xlabel("Number of Predators")
    plt.ylabel("Number of Prey")
    plt.title("Density Plot of Predator vs Prey Population")

    # Plot the isoclines.
    prey_iso = model.prey_growth_rate / model.predation_efficiency
    predator_iso = model.death_rate / model.conversion_efficiency
    plt.axhline(y=predator_iso, color="gray", linestyle="--", label="Prey Isocline")
    plt.axvline(x=prey_iso, color="gray", linestyle="--", label="Predator Isocline")

    # Display the plot
    plt.tight_layout()
    plt.show()


def main() -> None:
    # How long to run the simulation for.
    run_time = 100
    # Timestep to use for integration.
    timestep = 0.1

    model = LotkaVolterraModel(
        prey_growth_rate=0.1,
        predation_efficiency=0.002,
        conversion_efficiency=0.0002,
        death_rate=0.2,
    )
    solver = RK4Solver(model, initial_value=np.array([1500.0, 50.0]))

    # Run the simulation.
    num_iterations = int(run_time // timestep)
    results = solver.outputs(timestep)
    results = list(islice(results, num_iterations))
    # Create an index for x-axis (e.g., time steps or generations)
    time_steps = np.arange(len(results)) * timestep
    results = pd.DataFrame(results, columns=["prey", "predators"])
    results["time"] = time_steps

    _plot(results)
    _plot_densities(results, model)


if __name__ == "__main__":
    main()

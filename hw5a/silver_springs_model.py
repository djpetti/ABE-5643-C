"""
Implements the Silver Springs compartment model.
"""

from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compartment_model import (
    CompartmentModel,
    Flow,
    ConstantFlow,
    SourceProportionalFlow,
)
from common.numerical_integration import RK4Solver


sns.set_theme()


class LightFlow(Flow):
    """
    Simulates the flow of light into the system.
    """

    def __init__(self, mean_photo: float, fluctuation: float, sink: str):
        """

        Args:
            mean_photo: The mean annual photosynthesis.
            fluctuation: The range of the annual fluctuation around the mean.
            sink: The name of the sink compartment this is connected to.

        """
        super().__init__(sink=sink, source=None)

        self.__mean_photo = mean_photo
        self.__range = fluctuation

        # Keeps track of the current week of the year.
        self.__week = 0

    def set_week(self, week: int) -> None:
        """
        Sets the current week of the year to use when calculating the amount of
        light energy input.

        Args:
            week: The week number of the simulation.

        """
        self.__week = week

    def difference_eq(self, prior_output: np.array) -> np.array:
        light = self.__mean_photo + self.__range * np.sin(
            2 * np.pi * (self.__week - 11) / 52
        )

        return np.array([light])


def _make_model() -> Tuple[CompartmentModel, LightFlow]:
    """
    Returns:
        The Silver Springs model, and the light input flow.

    """
    model = CompartmentModel()

    model.add_compartment("producers")
    model.add_compartment("herbivores")
    model.add_compartment("carnivores")
    model.add_compartment("top_carnivores")
    model.add_compartment("decomposers")

    # Forcing functions
    light_flow = LightFlow(mean_photo=400, fluctuation=175, sink="producers")
    model.add_flow(light_flow)
    model.add_flow(ConstantFlow(486 / 52, sink="herbivores"))

    # Feeding
    model.add_flow(
        SourceProportionalFlow(1.094 / 52, source="producers", sink="herbivores")
    )
    model.add_flow(
        SourceProportionalFlow(1.798 / 52, source="herbivores", sink="carnivores")
    )
    model.add_flow(
        SourceProportionalFlow(0.339 / 52, source="carnivores", sink="top_carnivores")
    )

    # Mortality
    model.add_flow(
        SourceProportionalFlow(1.310 / 52, source="producers", sink="decomposers")
    )
    model.add_flow(
        SourceProportionalFlow(5.141 / 52, source="herbivores", sink="decomposers")
    )
    model.add_flow(
        SourceProportionalFlow(0.742 / 52, source="carnivores", sink="decomposers")
    )
    model.add_flow(
        SourceProportionalFlow(0.889 / 52, source="top_carnivores", sink="decomposers")
    )

    # Respiration
    model.add_flow(SourceProportionalFlow(4.545 / 52, source="producers"))
    model.add_flow(SourceProportionalFlow(8.873 / 52, source="herbivores"))
    model.add_flow(SourceProportionalFlow(5.097 / 52, source="carnivores"))
    model.add_flow(SourceProportionalFlow(1.444 / 52, source="top_carnivores"))
    model.add_flow(SourceProportionalFlow(184.0 / 52, source="decomposers"))

    # Export
    model.add_flow(SourceProportionalFlow(0.94 / 52, source="producers"))

    return model, light_flow


def _plot(results: pd.DataFrame) -> None:
    """
    Plots the predator-prey numbers over time.

    Args:
        results: A dataframe containing the model output data.

    """
    # Melt the DataFrame to long format
    results_melted = results.melt(
        id_vars="week", var_name="category", value_name="biomass"
    )

    # Create the line plot
    fig = sns.lineplot(x="week", y="biomass", hue="category", data=results_melted)
    fig.set(yscale="log")

    # Add labels and title
    plt.xlabel("Week")
    plt.ylabel("Biomass (kcal/m^2)")
    plt.title("Silver Springs Model")

    # Display the plot
    plt.tight_layout()
    plt.show()


def main() -> None:
    model, light_flow = _make_model()
    solver = RK4Solver(model, initial_value=np.array([2635.0, 213.0, 62.0, 9.0, 25.0]))

    # Simulate the model for 3 years.
    timestep = 0.1
    times = np.arange(3 * 52 / timestep) * timestep
    results = []
    for time, output in zip(times, solver.outputs(timestep)):
        light_flow.set_week(time)
        results.append(output)
    results = pd.DataFrame(results, columns=model.compartment_names)
    results["week"] = times

    _plot(results)


if __name__ == "__main__":
    main()

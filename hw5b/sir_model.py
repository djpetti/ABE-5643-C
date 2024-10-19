from itertools import islice
from typing import Any

import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from common.compartment_model import (
    BasicCompartmentModel,
    SourceSinkProportionalFlow,
    SourceProportionalFlow,
)
from common.numerical_integration import RK4Solver


sns.set_theme()


class SirModel(BasicCompartmentModel):
    """
    Implements an SIR model of epidemics.
    """

    def __init__(self, *, transmission_rate: float, removal_rate: float):
        """
        Args:
            transmission_rate: The coefficient of transmission, i.e. the rate at
                which susceptibles become infected.
            removal_rate: The coefficient of removal, i.e. the rate at which the
                infected recover.

        """
        super().__init__()

        # Create the model.
        self.add_compartment("susceptible")
        self.add_compartment("infected")
        self.add_compartment("recovering")

        self.add_flow(
            SourceSinkProportionalFlow(
                transmission_rate,
                source="susceptible",
                sink="infected",
            )
        )
        self.add_flow(
            SourceProportionalFlow(removal_rate, source="infected", sink="recovering")
        )


def _run_model(initial_conditions: np.array, **kwargs: Any) -> pd.DataFrame:
    """
    Runs the model once.

    Args:
        initial_conditions: The initial conditions to use.
        **kwargs: Will be forwarded to the model constructor.

    Returns:
        The model output.

    """
    # Create the model.
    model = SirModel(**kwargs)

    solver = RK4Solver(model, initial_value=initial_conditions)
    # Run the model over 160 time units.
    timestep = 0.1
    times = np.arange(0, 160 + timestep, timestep)
    outputs = np.array(list(islice(solver.outputs(timestep), len(times)))).astype(int)
    outputs = pd.DataFrame(outputs, columns=["susceptible", "infected", "recovering"])
    outputs["time"] = times

    return outputs


def ex_1() -> None:
    initial_conditions = np.array([19.0, 1.0, 0.0])
    outputs = _run_model(initial_conditions, transmission_rate=0.01, removal_rate=0.02)

    # Plot the results.
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="time", y="susceptible", data=outputs, label="Susceptible", color="blue"
    )
    sns.lineplot(x="time", y="infected", label="Infected", data=outputs, color="red")
    sns.lineplot(
        x="time", y="recovering", label="Recovering", data=outputs, color="green"
    )

    plt.title("SIR Model Simulation")
    plt.xlabel("Time")
    plt.ylabel("Population Count")
    plt.show()


def ex_5() -> None:
    s_values = [12, 20, 40, 80]
    model_args = dict(transmission_rate=0.01, removal_rate=0.12)
    all_outputs = [
        _run_model(np.array([s, 1, 0], dtype=float), **model_args) for s in s_values
    ]

    # Plot the results.
    plt.figure(figsize=(10, 6))

    for s, output in zip(s_values, all_outputs):
        sns.scatterplot(x="susceptible", y="infected", data=output, label=f"S={s}")
    plt.xlabel("Number of Susceptibles")
    plt.ylabel("Number of Infecteds")
    plt.title("Density Plot of Susceptible vs Infected Populations")

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ex_5()

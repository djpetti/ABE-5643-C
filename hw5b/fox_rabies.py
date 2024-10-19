from itertools import islice
from typing import Any, List, Iterable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.compartment_model import (
    BasicCompartmentModel,
    SourceProportionalFlow,
    SinkProportionalFlow,
    Flow,
)
from common.numerical_integration import RK4Solver


sns.set_theme()


class InfectionFlow(Flow):
    """
    A special flow to model the infection of susceptible foxes.
    """

    def __init__(self, rate: float, carrier_name: str = "carriers", **kwargs: Any):
        """
        Args:
            rate: The rate constant for this flow.
            carrier_name: The name of the carrier compartment.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(**kwargs)

        self.__rate = rate
        self.__carrier_name = carrier_name

    @property
    def inputs(self) -> List[str]:
        return [self.source, self.__carrier_name]

    def difference_eq(self, prior_output: np.array) -> np.array:
        return self._symmetric_flow(self.__rate * np.prod(prior_output))


class CarryingCapacityFlow(Flow):
    """
    A flow that models the carrying capacity.
    """

    def __init__(
        self,
        *,
        carrying_capacity: float,
        growth_rate: float,
        population_names: Iterable[str] = ("susceptible", "infected", "carriers"),
        **kwargs
    ):
        """
        Args:
            carrying_capacity: The carrying capacity.
            growth_rate: The population growth rate.
            population_names: The names of the compartments whose values
                collectively constitute the total population.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(**kwargs)

        self.__carrying_capacity = carrying_capacity
        self.__growth_rate = growth_rate
        self.__population_names = list(population_names)

    @property
    def inputs(self) -> List[str]:
        return [self.source] + self.__population_names

    def difference_eq(self, prior_output: np.array) -> np.array:
        # Calculate the total size of the population.
        population_size = np.sum(prior_output[1:])
        source_value = prior_output[0]

        pop_limit_rate = self.__growth_rate * population_size / self.__carrying_capacity
        return self._symmetric_flow(source_value * pop_limit_rate)


class RabiesModel(BasicCompartmentModel):
    """
    Model of rabies spread in a population of foxes.
    """

    def __init__(
        self,
        *,
        transmission_rate: float,
        birth_rate: float,
        death_rate: float,
        rabid_transition_rate: float,
        rabies_death_rate: float,
        population_size: int,
        carrying_capacity: int
    ):
        """
        Args:
            transmission_rate: The rate at which susceptible foxes get
                infected.
            birth_rate: The rate at which foxes are born.
            death_rate: The rate at which foxes die normally.
            rabid_transition_rate: The rate at which infected foxes become
                rabid.
            rabies_death_rate: The rate at which rabid foxes die due to rabies.
            population_size: The total population size.
            carrying_capacity: The carrying capacity of the environment.

        """
        super().__init__()

        self.add_compartment("susceptible")
        self.add_compartment("infected")
        self.add_compartment("carriers")

        # Fox births
        self.add_flow(SinkProportionalFlow(birth_rate, sink="susceptible"))
        # Normal fox deaths
        self.add_flow(SourceProportionalFlow(death_rate, source="susceptible"))
        self.add_flow(SourceProportionalFlow(death_rate, source="infected"))
        self.add_flow(SourceProportionalFlow(death_rate, source="carriers"))

        # Infection of susceptible foxes.
        self.add_flow(
            InfectionFlow(transmission_rate, source="susceptible", sink="infected")
        )

        # Population limits due to carrying capacity.
        growth_rate = birth_rate - death_rate
        self.add_flow(
            CarryingCapacityFlow(
                carrying_capacity=carrying_capacity,
                growth_rate=growth_rate,
                source="susceptible",
            )
        )
        self.add_flow(
            CarryingCapacityFlow(
                carrying_capacity=carrying_capacity,
                growth_rate=growth_rate,
                source="infected",
            )
        )
        self.add_flow(
            CarryingCapacityFlow(
                carrying_capacity=carrying_capacity,
                growth_rate=growth_rate,
                source="carriers",
            )
        )

        # Progression of the disease to the rabid stage
        self.add_flow(
            SourceProportionalFlow(
                rabid_transition_rate, source="infected", sink="carriers"
            )
        )
        # Rabies-related mortality
        self.add_flow(SourceProportionalFlow(rabies_death_rate, source="carriers"))


def main() -> None:
    # Create the model.
    model = RabiesModel(
        birth_rate=0.00274,
        death_rate=0.00137,
        transmission_rate=0.21833,
        rabid_transition_rate=0.033562,
        rabies_death_rate=0.2,
        population_size=2,
        carrying_capacity=2,
    )

    solver = RK4Solver(model, initial_value=np.array([1.9, 0.1, 0]))
    # Run the model over 40 years.
    timestep = 1
    times = np.arange(0, 40 * 365 + timestep, timestep)
    outputs = np.array(list(islice(solver.outputs(timestep), len(times))))
    outputs = pd.DataFrame(outputs, columns=["susceptible", "infected", "carriers"])
    outputs["total"] = (
        outputs["susceptible"] + outputs["infected"] + outputs["carriers"]
    )
    outputs["time"] = times

    # Plot the results.
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="time", y="total", data=outputs, label="Total")
    sns.lineplot(x="time", y="carriers", label="Carriers", data=outputs)

    plt.title("Fox Rabies Model Simulation")
    plt.xlabel("Time")
    plt.ylabel("Population Density (foxes/km^2)")
    plt.show()


if __name__ == "__main__":
    main()

"""
Tools for implementing compartment models.
"""

import abc
from typing import List, Any

import numpy as np

from common.numerical_integration import Model


class Flow(Model, abc.ABC):
    """
    Models the flow from one compartment to another.
    """

    def __init__(self, source: str | None = None, sink: str | None = None):
        """
        Args:
            source: The source compartment, if there is one.
            sink: The sink compartment, if there is one.
        """
        self.__source = source
        self.__sink = sink

    @property
    def source(self) -> str | None:
        """
        Returns:
            The source compartment, if there is one.

        """
        return self.__source

    @property
    def sink(self) -> str | None:
        """
        Returns:
            The sink compartment, if there is one.

        """
        return self.__sink

    def _symmetric_flow(self, rate: float) -> np.array:
        """
        Helper function to convert a single "symmetric" flow rate into proper
        inflow and outflow rates for the sink and source, respectively.

        Args:
            rate: The rate.

        Returns:
            Rates for the source and/or sink, depending on which are present
            for this flow.

        """
        output_rate = []
        if self.source is not None:
            output_rate.append(-rate)
        if self.sink is not None:
            output_rate.append(rate)

        return np.array(output_rate)

    @abc.abstractmethod
    def difference_eq(self, prior_output: np.array) -> np.array:
        """
        Args:
            prior_output: The prior values of the source and sink
                compartment, or just the source or sink, depending on which
                are provided. Should be a vector of either 1 or 2 values.

        Returns:
            The rate of change of the source and/or sink compartment values.

        """


class ConstantFlow(Flow):
    """
    Simulates a flow that is constant.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any):
        """
        Args:
            rate: The constant flow rate.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        Notes:
            Either the source or sink is expected to be None.

        """
        super().__init__(*args, **kwargs)

        self.__rate = self._symmetric_flow(rate)

    def difference_eq(self, prior_output: np.array) -> np.array:
        return self.__rate


class SourceProportionalFlow(Flow):
    """
    Simulates a flow that is proportional to the value of the source compartment.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any):
        """
        Args:
            rate: The proportional flow rate.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        self.__rate = rate

    def difference_eq(self, prior_output: np.array) -> np.array:
        source = prior_output[0]
        return self._symmetric_flow(self.__rate * source)


class CompartmentModel(Model):
    """
    Implements a compartment model.
    """

    def __init__(self):
        # Keeps track of the order that compartments were added in.
        self.__compartment_order: List[str] = []
        # Maps compartments to flows that have that compartment as a source.
        self.__flow_sources: dict[str | None, List[Flow]] = {None: []}

    def add_compartment(self, name: str) -> None:
        """
        Adds a compartment to the model, or resets the value of an existing one.

        Args:
            name: The name of the compartment.

        """
        self.__compartment_order.append(name)
        self.__flow_sources[name] = []

    def add_flow(self, flow: Flow) -> None:
        """
        Adds a new flow between compartments.

        Args:
            flow: The flow to add.

        """
        if flow.source is not None and flow.source not in self.__flow_sources:
            raise ValueError(f"Unknown source compartment {flow.source}.")
        if flow.sink is not None and flow.sink not in self.__flow_sources:
            raise ValueError(f"Unknown sink compartment {flow.sink}.")

        self.__flow_sources[flow.source].append(flow)

    @property
    def compartment_names(self) -> List[str]:
        """
        Returns:
            The names of the compartments in the model, in the order that
            they were added.

        """
        return self.__compartment_order[:]

    def difference_eq(self, prior_output: np.array) -> np.array:
        """
        Calculates the value of the model's difference equation.

        Args:
            prior_output: The previous value of all the compartments,
                specified in the order that the compartments were added.

        Returns:
            The next value of all the compartments, divided by the timestep.

        """
        # Index prior outputs by compartment names for easy access.
        prior_output = {n: v for n, v in zip(self.__compartment_order, prior_output)}

        # Use the flows to calculate change rates for connected compartments.
        compartment_rates = {}
        for compartment, flows in self.__flow_sources.items():
            for flow in flows:
                flow_inputs = []
                if flow.source is not None:
                    flow_inputs.append(prior_output[flow.source])
                if flow.sink is not None:
                    flow_inputs.append(prior_output[flow.sink])

                flow_rates = flow.difference_eq(np.array(flow_inputs))

                if flow.source is not None:
                    compartment_rates.setdefault(flow.source, 0)
                    compartment_rates[flow.source] += flow_rates[0]
                if flow.sink is not None:
                    compartment_rates.setdefault(flow.sink, 0)
                    compartment_rates[flow.sink] += flow_rates[-1]

        return np.array([compartment_rates[c] for c in self.__compartment_order])

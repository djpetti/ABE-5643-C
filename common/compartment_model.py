"""
Tools for implementing compartment models.
"""

import abc
from typing import List, Any, cast, Iterable

import numpy as np

from .numerical_integration import Model, OutputType


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

    @property
    def inputs(self) -> List[str]:
        """
        Returns:
            The names of the compartments whose values are expected as inputs
            to `difference_eq()`, in the order they are expected.

        """
        expected_inputs = []
        if self.source is not None:
            expected_inputs.append(self.source)
        if self.sink is not None:
            expected_inputs.append(self.sink)

        return expected_inputs

    def _symmetric_flow(self, rate: float) -> np.array:
        """
        Helper function to convert a single "symmetric" flow rate into proper
        inflow and outflow rates for the sink and source, respectively.

        Args:
            rate: The rate (moving from the source to the sink).

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
            prior_output: The prior values of the compartmens specified by
            `inputs`, in that order.

        Returns:
            The rate of change of the source and/or sink compartment values.

        """


class LinearFlow(Flow, abc.ABC):
    """
    A flow that can be described by a linear ODE. This allows us to use some
    more efficient computational methods.
    """

    @property
    def is_dynamic(self) -> bool:
        """
        If true, then the rate "constants" for this flow are time-dependent.
        Otherwise, it the model is free to assume that they never change.

        """
        return False

    @property
    @abc.abstractmethod
    def rate_constants(self) -> np.array:
        """
        Computes the rate constants to use for this flow. Specifically, assuming
        X_s is the value of the source compartment, and X_d the value of the
        destination compartment, we assume that the effect of this flow can
        be described by:

        dX_d/dt = k1 * X_s + k2 * X_d + k3
        dX_s/dt = -dX_d/dt

        Returns:
            The rate constants k1, k2, and k3

        """

    def difference_eq(self, prior_output: np.array) -> np.array:
        compartment_values = np.array([0, 0, 1], dtype=float)
        if self.source is not None:
            compartment_values[0] = prior_output[0]
        if self.sink is not None:
            compartment_values[1] = prior_output[-1]

        output = np.array(compartment_values) @ self.rate_constants

        return self._symmetric_flow(output)


class ConstantFlow(LinearFlow):
    """
    Simulates a flow that is constant.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any):
        """
        Args:
            rate: The constant flow rate.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        self.__rate_constants = np.array([0, 0, rate], dtype=float)

    @property
    def rate_constants(self) -> np.array:
        return self.__rate_constants


class SourceProportionalFlow(LinearFlow):
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

        self.__rate_constants = np.array([rate, 0, 0], dtype=float)

    @property
    def rate_constants(self) -> np.array:
        return self.__rate_constants


class SinkProportionalFlow(LinearFlow):
    """
    Simulates a flow that is proportional to the value of the sink compartment.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any):
        """
        Args:
            rate: The proportional flow rate.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        self.__rate_constants = np.array([0, rate, 0], dtype=float)

    @property
    def rate_constants(self) -> np.array:
        return self.__rate_constants


class SourceSinkProportionalFlow(Flow):
    """
    Simulates a flow that is proportional to the values of both the source
    and sink compartments.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any):
        """
        Args:
            rate: The proportional flow rate.
            *args: Will be passed on to the superclass constructor.
            **kwargs: Will be passed on to the superclass constructor.
        """
        super().__init__(*args, **kwargs)

        self.__rate = rate

    def difference_eq(self, prior_output: np.array) -> np.array:
        rate = np.prod(prior_output) * self.__rate
        return self._symmetric_flow(rate)


class CompartmentModel(Model, abc.ABC):
    """
    Base class for all compartment models.
    """

    def __init__(self):
        # Keeps track of the order that compartments were added in.
        self._compartment_order: List[str] = []
        # Maps compartments to flows that have that compartment as a source.
        self._flow_sources: dict[str | None, List[Flow]] = {None: []}

    def add_compartment(self, name: str) -> None:
        """
        Adds a compartment to the model, or resets the value of an existing one.

        Args:
            name: The name of the compartment.

        """
        self._compartment_order.append(name)
        self._flow_sources[name] = []

    def _do_add_flow(self, flow: Flow) -> None:
        """
        Adds a new flow between compartments.

        Args:
            flow: The flow to add.

        """
        if flow.source is not None and flow.source not in self._flow_sources:
            raise ValueError(f"Unknown source compartment {flow.source}.")
        if flow.sink is not None and flow.sink not in self._flow_sources:
            raise ValueError(f"Unknown sink compartment {flow.sink}.")

        self._flow_sources[flow.source].append(flow)

    def add_flow(self, flow: LinearFlow) -> None:
        """
        Adds a new flow between compartments.

        Args:
            flow: The flow to add.

        """
        self._do_add_flow(flow)

    @property
    def compartment_names(self) -> List[str]:
        """
        Returns:
            The names of the compartments in the model, in the order that
            they were added.

        """
        return self._compartment_order[:]


class BasicCompartmentModel(CompartmentModel):
    """
    Implements a standard compartment model.
    """

    def add_flow(self, flow: Flow) -> None:
        self._do_add_flow(flow)

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
        prior_output = {n: v for n, v in zip(self._compartment_order, prior_output)}

        # Use the flows to calculate change rates for connected compartments.
        compartment_rates = {}
        for _, flows in self._flow_sources.items():
            for flow in flows:
                flow_inputs = [prior_output[c] for c in flow.inputs]
                flow_rates = flow.difference_eq(np.array(flow_inputs))

                if flow.source is not None:
                    compartment_rates.setdefault(flow.source, 0)
                    compartment_rates[flow.source] += flow_rates[0]
                if flow.sink is not None:
                    compartment_rates.setdefault(flow.sink, 0)
                    compartment_rates[flow.sink] += flow_rates[-1]

        return np.array([compartment_rates[c] for c in self._compartment_order])


class MatrixCompartmentModel(CompartmentModel):
    """
    Does the same thing as a `BasicCompartmentModel`, but uses matrices
    internally which dramatically speeds up computation. However,
    the disadvantage is that it is less flexible, and can only accept
    `LinearFlows`.
    """

    def __init__(self):
        super().__init__()

        # Stores flows that are time-dependent.
        self.__dynamic_flows = []

        # Stores the static components of the A matrix.
        self.__static_a_matrix = None
        # If true, we need to recompute the A matrix because the model changed.
        self.__recompute_a_matrix = True

    def add_compartment(self, name: str) -> None:
        super().add_compartment(name)
        self.__recompute_a_matrix = True

    def add_flow(self, flow: LinearFlow) -> None:
        super().add_flow(flow)
        self.__recompute_a_matrix = True

        if flow.is_dynamic:
            self.__dynamic_flows.append(flow)

    def __compute_a_matrix_for_flows(self, flows: Iterable[LinearFlow]) -> np.array:
        """
        Computes the A matrix given the specified flows.

        Args:
            flows: The flows.

        Returns:
            The computed A-matrix.

        """
        # We add an extra row and column here that will just be used as a
        # scratch space when computing the matrix. We'll get rid of it when
        # we're done.
        a_matrix = np.zeros(
            (len(self._compartment_order) + 1, len(self._compartment_order) + 2),
            dtype=float,
        )

        # First, some housekeeping: we need a quick way to map compartment
        # names to rows and columns in the matrix.
        compartment_indices = {
            name: idx for (idx, name) in enumerate(self._compartment_order)
        }

        for flow in flows:
            flow = cast(LinearFlow, flow)
            source_rate, sink_rate, bias_rate = flow.rate_constants

            # If there is no source or sink, this will direct updates to
            # the scratch space in the A matrix.
            source_i = compartment_indices.get(flow.source, -1)
            sink_i = compartment_indices.get(flow.sink, -1)

            # Source-dependent flow out of the source.
            a_matrix[source_i, source_i] -= source_rate
            # Sink-dependent flow out of the source.
            a_matrix[source_i, sink_i] -= sink_rate
            # Constant flow out of the source.
            a_matrix[source_i, -2] -= bias_rate

            # Source-dependent flow into the sink.
            a_matrix[sink_i, source_i] += source_rate
            # Sink-dependent flow into the sink.
            a_matrix[sink_i, sink_i] += sink_rate
            # Constant flow into the sink.
            a_matrix[sink_i, -2] += bias_rate

        # Remove the scratch space.
        return a_matrix[:-1, :-1]

    def __compute_a_matrix(self) -> np.array:
        """
        Computes the A matrix to use for the model.

        Returns:
            The computed A matrix. The rows and columns will be in the same
            order as the corresponding compartments were added to the model.
            One extra column will be added at the end to account for flow
            rates that are not dependent on any compartment values.

        """
        # We always need to recompute the dynamic portion.
        dynamic_a = self.__compute_a_matrix_for_flows(self.__dynamic_flows)

        if self.__recompute_a_matrix:
            # We need to compute the static part too.
            flows = (
                cast(LinearFlow, f) for fs in self._flow_sources.values() for f in fs
            )
            flows = filter(lambda f: not f.is_dynamic, flows)
            self.__static_a_matrix = self.__compute_a_matrix_for_flows(flows)
            self.__recompute_a_matrix = False

        return dynamic_a + self.__static_a_matrix

    def difference_eq(self, prior_output: OutputType) -> OutputType:
        a_matrix = self.__compute_a_matrix()

        # Augment the prior output to make the constant rate terms work.
        augmented_prior = np.append(prior_output, [1])
        return a_matrix @ augmented_prior

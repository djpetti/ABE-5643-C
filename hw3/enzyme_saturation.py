import seaborn as sns
from matplotlib import pyplot as plt


sns.set()


class MichaelisMentenModel:
    """
    Implements the Michaelis-Menten model of enzyme saturation.
    """

    def __init__(self, *, k_m: float, max_velocity: float):
        """
        Args:
            k_m: The Michaelis-Menten constant.
            max_velocity: The maximum velocity of the reaction.

        """
        self.__k_m = k_m
        self.__max_velocity = max_velocity

    def initial_velocity(self, substrate: float) -> float:
        """
        Calculates the initial velocity of the reaction.

        Args:
            substrate: The concentration of the substrate.

        Returns:
            The initial velocity of the reaction.

        """
        return self.__max_velocity * substrate / (self.__k_m + substrate)


def main() -> None:
    model = MichaelisMentenModel(k_m=10, max_velocity=0.1)

    concentrations = list(range(81))
    velocities = [model.initial_velocity(c) for c in concentrations]

    fig = sns.lineplot(x=concentrations, y=velocities)
    fig.set_title("Initial Reaction Velocity")
    plt.xlabel("Substrate Concentration (mM)")
    plt.ylabel("Initial Reaction Velocity (mM/s)")
    plt.show()


if __name__ == "__main__":
    main()
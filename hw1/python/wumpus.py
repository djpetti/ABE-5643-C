from dataclasses import dataclass
from typing import Tuple, Iterable
import random
import enum


class PlayerDied(Exception):
    """
    Raised when the player dies.
    """


class World:
    """
    Simulates a Wumpus world.
    """

    @dataclass
    class Room:
        """
        Represents a room in the Wumpus world.

        Attributes:
            has_wumpus: Whether the Wumpus is in this room.
            has_gold: Whether the gold is in this room.
            has_pit: Whether there is a pit in this room.

        """

        has_wumpus: bool
        has_gold: bool
        has_pit: bool

    class Direction(enum.IntEnum):
        """
        Represents the directions that the player can move.
        """

        UP = enum.auto()
        DOWN = enum.auto()
        LEFT = enum.auto()
        RIGHT = enum.auto()

    def __init__(self):
        # This represents the map of the world.
        self.__world = []
        for _ in range(4):
            self.__world.append([self.Room(False, False, False) for _ in range(4)])

        self.__init_world()

        # The current location of the player.
        self.__player_location = (3, 0)
        # Whether the player has shot the arrow.
        self.__shot_arrow = False
        # Whether the Wumpus is alive.
        self.__wumpus_alive = True
        # Whether the player is holding the gold.
        self.__player_has_gold = False

    def __random_room(self) -> Room:
        """
        Selects a room randomly.

        Returns:
            The selected room.

        """
        row = random.choice(self.__world)
        return random.choice(row)

    def __init_world(self) -> None:
        """
        Randomly initializes the game world.
        """
        # Choose a new Wumpus location.
        wumpus_room = self.__random_room()
        wumpus_room.has_wumpus = True

        # Choose a new gold location.
        gold_room = self.__random_room()
        gold_room.has_gold = True

        # Choose new pit locations.
        num_pits = random.randint(0, 4)
        for _ in range(num_pits):
            pit_room = self.__random_room()
            pit_room.has_pit = True

    def __get_neighborhood(self, location: Tuple[int, int]) -> Iterable[Room]:
        """
        Gets the rooms surrounding a particular room.

        Args:
            location: The location to get the neighborhood for.

        Yields:
            The surrounding rooms.

        """
        row, col = location
        for i, j in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if i < 0 or i > 3 or j < 0 or j > 3:
                continue
            yield self.__world[i][j]

    def __move_player(self, direction: Direction) -> bool:
        """
        Moves the player in the specified direction.

        Args:
            direction: The direction to move the player.

        Returns:
            True if the player moved successfully, false if they hit a wall.

        """
        row, col = self.__player_location
        if direction == self.Direction.UP:
            row = max(0, row - 1)
        elif direction == self.Direction.DOWN:
            row = min(3, row + 1)
        elif direction == self.Direction.LEFT:
            col = max(0, col - 1)
        elif direction == self.Direction.RIGHT:
            col = min(3, col + 1)

        move_successful = (row, col) != self.__player_location
        self.__player_location = (row, col)
        return move_successful

    def __check_player_alive(self) -> None:
        """
        Checks that the player is still alive.

        """
        row, col = self.__player_location
        room = self.__world[row][col]

        if room.has_pit:
            raise PlayerDied("You fell into a pit and died!")
        if room.has_wumpus and self.__wumpus_alive:
            raise PlayerDied("You were eaten by the Wumpus!")

    def check_breeze(self) -> bool:
        """
        Checks if there is a breeze in the player's location.

        Returns:
            True if there is a breeze.

        """
        for room in self.__get_neighborhood(self.__player_location):
            if room.has_pit:
                return True
        return False

    def check_stench(self) -> bool:
        """
        Checks if there is a stench in the player's location.

        Returns:
            True if there is a stench.

        """
        for room in self.__get_neighborhood(self.__player_location):
            if room.has_wumpus:
                return True
        return False

    def check_glitter(self) -> bool:
        """
        Checks if there is gold in the player's location.

        Returns:
            True if there is gold.

        """
        row, col = self.__player_location
        return self.__world[row][col].has_gold

    def move(self, direction: Direction) -> bool:
        """
        Moves the player in a given direction.

        Args:
            direction: The direction to move in.

        Returns:
            True if the player moved successfully, false if they hit a wall.

        """
        move_successful = self.__move_player(direction)
        self.__check_player_alive()
        return move_successful

    def shoot(self, direction: Direction) -> bool:
        """
        Shoots an arrow in a given direction.

        Args:
            direction: The direction to shoot the arrow.

        Returns:
            True if the arrow hit the Wumpus, false otherwise.

        """
        self.__shot_arrow = True
        row, col = self.__player_location

        if direction in {self.Direction.UP, self.Direction.DOWN}:
            axis = [r[col] for r in self.__world]
            pos = row
        else:
            axis = self.__world[row]
            pos = col

        if direction in {self.Direction.UP, self.Direction.LEFT}:
            increment = -1
        else:
            increment = 1

        # Follow the arrow's path until it hits something.
        while 0 <= pos < len(axis):
            room = axis[pos]
            if room.has_wumpus:
                # We hit the wumpus.
                self.__wumpus_alive = False
                return True

            pos += increment

        return False

    def grab(self) -> None:
        """
        Grabs the gold in the player's location.

        """
        row, col = self.__player_location
        self.__player_has_gold = self.__world[row][col].has_gold
        self.__world[row][col].has_gold = False

    @property
    def at_entrance(self) -> bool:
        """
        Returns:
            True if the player is at the entrance to the cave.

        """
        return self.__player_location == (3, 0)

    @property
    def has_gold(self) -> bool:
        """
        Returns:
            True if the player has the gold.

        """
        return self.__player_has_gold

    @property
    def shot_arrow(self) -> bool:
        """
        Returns:
            True if the player has shot the arrow.

        """
        return self.__shot_arrow

    @property
    def wumpus_alive(self) -> bool:
        """
        Returns:
            True if the Wumpus is alive.

        """
        return self.__wumpus_alive


class Action(enum.Enum):
    """
    Represents player actions.
    """

    MOVE = "Move to another room"
    SHOOT = "Shoot your arrow"
    GRAB = "Grab the gold"
    CLIMB = "Climb out of the cave"


def _choose_action(
    enable_grab: bool = False, enable_climb: bool = False, enable_shoot: bool = True
) -> Action:
    """
    Shows a menu to the user allowing them to choose an action.

    Args:
        enable_grab: Whether to enable the grab action.
        enable_climb: Whether to enable the climb action.
        enable_shoot: Whether to enable the shoot action.

    Returns:
        The action selected by the user.

    """
    print("===================================")
    print("What do you do?\n")

    enabled_actions = set(Action)
    if not enable_grab:
        enabled_actions.discard(Action.GRAB)
    if not enable_climb:
        enabled_actions.discard(Action.CLIMB)
    if not enable_shoot:
        enabled_actions.discard(Action.SHOOT)

    # Print the possible actions.
    action_list = list(Action)
    for i, action in enumerate(action_list):
        if action in enabled_actions:
            print(f"\t{i}: {action.value}")

    action_id = int(input("\nChoice: "))
    action_id = max(action_id, 0)
    action_id = min(action_id, len(action_list) - 1)
    return action_list[action_id]


def _choose_direction() -> World.Direction:
    """
    Shows a menu to the user allowing them to choose a direction.

    Returns:
        The direction selected by the user.

    """
    print("===================================")
    print("Which direction?\n")

    direction_list = list(World.Direction)
    for i, direction in enumerate(direction_list):
        print(f"\t{i}: {direction.name}")

    direction_id = int(input("\nChoice: "))
    direction_id = max(direction_id, 0)
    direction_id = min(direction_id, len(direction_list) - 1)
    return direction_list[direction_id]


def main() -> None:
    """
    Play the game.

    """
    world = World()
    print(
        "Welcome to the Wumpus world. This is how we decide who gets tenure"
        " from now on."
    )
    print("You climb down into the Wumpus's cave. It is very dark.\n")

    while True:
        if world.check_breeze():
            print("> You feel a breeze.")
        if world.check_stench():
            print("> You smell something terrible.")
        if world.check_glitter():
            print("> You see the glimmer of gold!")

        # Get the player's action.
        action = _choose_action(
            enable_grab=world.check_glitter(),
            enable_climb=world.at_entrance,
            enable_shoot=not world.shot_arrow,
        )

        if action == action.MOVE:
            direction = _choose_direction()
            try:
                if not world.move(direction):
                    print("> You hit a wall.")
            except PlayerDied as death_message:
                print(f"> {death_message}")
                break

        if action == action.CLIMB:
            if world.has_gold:
                print("Congratulations! You can pay your student loans!")
            else:
                print("You leave the cave without the gold. Coward.")
            break

        if action == action.SHOOT:
            direction = _choose_direction()
            if world.shoot(direction):
                print("> You hear the hideous scream of a dying Wumpus!")

        if action == action.GRAB:
            world.grab()
            print("> You now have the gold!")


if __name__ == "__main__":
    main()

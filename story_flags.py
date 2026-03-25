"""Story flag definitions — episodes from game start through Brock.

Each StoryFlag is one RL episode. The curriculum trains them in order.
Detection uses game memory reads — no manual flag flipping needed.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StoryFlag:
    """A single story milestone that defines one RL episode."""
    id: str                          # unique identifier
    name: str                        # human-readable name
    description: str                 # what the player needs to do
    check: Callable[[dict], bool]    # takes game_state dict, returns True if completed
    prerequisite: str | None = None  # id of flag that must be mastered first
    max_steps: int = 2000            # episode timeout
    optimal_steps: int | None = None # set from first successful run


def _check_left_house(state: dict) -> bool:
    """Player has left their house and is in Pallet Town."""
    return state["map_name"] == "PALLET_TOWN"


def _check_entered_oaks_lab(state: dict) -> bool:
    """Player entered Oak's Lab."""
    return state["map_name"] == "OAKS_LAB"


def _check_got_starter(state: dict) -> bool:
    """Player has a Pokemon in their party."""
    return state["party_count"] >= 1


def _check_reached_viridian(state: dict) -> bool:
    """Player reached Viridian City."""
    return state["map_name"] == "VIRIDIAN_CITY"


def _check_got_oaks_parcel(state: dict) -> bool:
    """Player picked up Oak's Parcel from Viridian Mart."""
    return state.get("has_oaks_parcel", False)


def _check_delivered_parcel(state: dict) -> bool:
    """Player delivered the parcel — detected by having Pokedex.

    In FireRed, delivering the parcel immediately leads to getting the Pokedex.
    We detect the combined event since the parcel disappears from inventory.
    """
    return state.get("has_pokedex", False)


def _check_reached_route2(state: dict) -> bool:
    """Player reached Route 2 (north of Viridian)."""
    return state["map_name"] == "ROUTE_2"


def _check_entered_viridian_forest(state: dict) -> bool:
    """Player entered Viridian Forest."""
    return state["map_name"] == "VIRIDIAN_FOREST"


def _check_exited_viridian_forest(state: dict) -> bool:
    """Player exited Viridian Forest to the north side (Route 2 north entrance)."""
    # They're on Route 2 AND north of the forest exit (y < 20ish)
    # OR they're already in Pewter City
    if state["map_name"] == "PEWTER_CITY":
        return True
    if state["map_name"] == "ROUTE_2" and state["player_y"] < 20:
        return True
    return False


def _check_reached_pewter(state: dict) -> bool:
    """Player reached Pewter City."""
    return state["map_name"] == "PEWTER_CITY"


def _check_entered_pewter_gym(state: dict) -> bool:
    """Player entered Pewter Gym."""
    return state["map_name"] == "PEWTER_GYM"


def _check_beat_brock(state: dict) -> bool:
    """Player earned the Boulder Badge (badge bit 0)."""
    return (state["badges"] & 0x01) != 0


# --- Flag chain: start → Brock ---

STORY_FLAGS: list[StoryFlag] = [
    StoryFlag(
        id="leave_house",
        name="Leave House",
        description="Exit player's house to Pallet Town",
        check=_check_left_house,
        prerequisite=None,
        max_steps=1000,
    ),
    StoryFlag(
        id="enter_oaks_lab",
        name="Enter Oak's Lab",
        description="Walk to Oak's Lab in Pallet Town",
        check=_check_entered_oaks_lab,
        prerequisite="leave_house",
        max_steps=300,
    ),
    StoryFlag(
        id="get_starter",
        name="Get Starter Pokemon",
        description="Choose a starter Pokemon from Oak's Lab",
        check=_check_got_starter,
        prerequisite="enter_oaks_lab",
        max_steps=500,
    ),
    StoryFlag(
        id="reach_viridian",
        name="Reach Viridian City",
        description="Walk north through Route 1 to Viridian City",
        check=_check_reached_viridian,
        prerequisite="get_starter",
        max_steps=1500,
    ),
    StoryFlag(
        id="get_oaks_parcel",
        name="Get Oak's Parcel",
        description="Enter Viridian Mart and receive Oak's Parcel",
        check=_check_got_oaks_parcel,
        prerequisite="reach_viridian",
        max_steps=500,
    ),
    StoryFlag(
        id="deliver_parcel",
        name="Deliver Parcel & Get Pokedex",
        description="Return to Oak's Lab and deliver the parcel to get the Pokedex",
        check=_check_delivered_parcel,
        prerequisite="get_oaks_parcel",
        max_steps=2000,
    ),
    StoryFlag(
        id="reach_route2",
        name="Reach Route 2",
        description="Head north from Viridian City to Route 2",
        check=_check_reached_route2,
        prerequisite="deliver_parcel",
        max_steps=1000,
    ),
    StoryFlag(
        id="enter_viridian_forest",
        name="Enter Viridian Forest",
        description="Enter Viridian Forest from Route 2",
        check=_check_entered_viridian_forest,
        prerequisite="reach_route2",
        max_steps=500,
    ),
    StoryFlag(
        id="exit_viridian_forest",
        name="Exit Viridian Forest",
        description="Navigate through Viridian Forest to the north exit",
        check=_check_exited_viridian_forest,
        prerequisite="enter_viridian_forest",
        max_steps=3000,
    ),
    StoryFlag(
        id="reach_pewter",
        name="Reach Pewter City",
        description="Walk from the forest exit to Pewter City",
        check=_check_reached_pewter,
        prerequisite="exit_viridian_forest",
        max_steps=500,
    ),
    StoryFlag(
        id="enter_pewter_gym",
        name="Enter Pewter Gym",
        description="Find and enter the Pewter City Gym",
        check=_check_entered_pewter_gym,
        prerequisite="reach_pewter",
        max_steps=500,
    ),
    StoryFlag(
        id="beat_brock",
        name="Beat Brock",
        description="Defeat Brock and earn the Boulder Badge",
        check=_check_beat_brock,
        prerequisite="enter_pewter_gym",
        max_steps=3000,
    ),
]

# Quick lookup by id
FLAG_BY_ID: dict[str, StoryFlag] = {f.id: f for f in STORY_FLAGS}

# Ordered chain for curriculum
FLAG_ORDER: list[str] = [f.id for f in STORY_FLAGS]

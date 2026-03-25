"""Structured overworld navigation for Pokemon FireRed."""

from dataclasses import dataclass, field
from enum import Enum
import random

from claude_client import classify_navigation_intent, select_building_door, select_interactable_target, select_npc_target, select_route_exit
from exploration import (
    edge_exit_action,
    facing_direction,
    is_counter_talk_position,
    is_passable,
    path_to_adjacent_object,
    path_to_map_edge,
    path_to_nearest_door,
    path_to_nearest_grass,
    path_to_target_tile,
)


class NavState(str, Enum):
    FREE_EXPLORE = "FREE_EXPLORE"
    SELECT_BUILDING_TARGET = "SELECT_BUILDING_TARGET"
    PATH_TO_BUILDING = "PATH_TO_BUILDING"
    ENTER_BUILDING = "ENTER_BUILDING"
    PATH_TO_ROUTE_EXIT = "PATH_TO_ROUTE_EXIT"
    EXIT_ROUTE = "EXIT_ROUTE"
    TRAIN_FIND_GRASS = "TRAIN_FIND_GRASS"
    TRAIN_IN_GRASS = "TRAIN_IN_GRASS"
    BUILDING_PATH_TO_NPC = "BUILDING_PATH_TO_NPC"
    BUILDING_TALK_TO_NPC = "BUILDING_TALK_TO_NPC"
    PATH_TO_OBJECT = "PATH_TO_OBJECT"
    INTERACT_OBJECT = "INTERACT_OBJECT"
    BUILDING_LEAVE = "BUILDING_LEAVE"


@dataclass
class WorldSnapshot:
    game_state: str
    map_id: int
    map_name: str
    player_x: int
    player_y: int
    in_battle: bool
    in_dialogue: bool
    collision: tuple[int, int, list[str]] | None
    objects: list[dict] | None
    bg_events: list[dict] | None
    connections: list[dict] | None
    player_facing: int | None
    current_objective: str
    strategy_objective: str
    party: list[dict]


@dataclass
class NavContext:
    state: NavState = NavState.FREE_EXPLORE
    intent: str | None = None
    target_pos: tuple[int, int] | None = None
    target_object_id: int | None = None
    last_map_id: int | None = None
    last_position: tuple[int, int] | None = None
    stalled_ticks: int = 0
    selected_door: tuple[int, int] | None = None
    selected_door_origin_map: int | None = None
    selected_goal_type: str | None = None
    selected_door_reason: str = ""
    known_doors: dict[str, str] = field(default_factory=dict)
    rejected_doors: dict[str, set[str]] = field(default_factory=dict)
    force_leave_building: bool = False
    pending_exit_tile: tuple[int, int] | None = None
    selected_route_exit: tuple[int, int] | None = None
    selected_route_reason: str = ""
    rejected_route_exits: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    llm_intent: str | None = None
    llm_intent_reason: str = ""
    llm_intent_key: str | None = None
    last_building_result: str = ""
    selected_npc_reason: str = ""
    selected_object_reason: str = ""
    target_object_kind: str | None = None
    target_object_last_pos: tuple[int, int] | None = None
    moving_npc_cooldowns: dict[int, int] = field(default_factory=dict)

    def note_progress(self, world: WorldSnapshot) -> None:
        current_pos = (world.player_x, world.player_y)
        if world.map_id != self.last_map_id:
            self.target_pos = None
            self.target_object_id = None
            self.stalled_ticks = 0
            self.pending_exit_tile = None
            self.selected_route_exit = None
            self.selected_route_reason = ""
            self.llm_intent = None
            self.llm_intent_reason = ""
            self.llm_intent_key = None
            self.last_building_result = ""
            self.selected_npc_reason = ""
            self.selected_object_reason = ""
            self.target_object_kind = None
            self.target_object_last_pos = None
            self.moving_npc_cooldowns.clear()
        elif current_pos == self.last_position:
            self.stalled_ticks += 1
        else:
            self.stalled_ticks = 0
        if self.moving_npc_cooldowns:
            updated = {}
            for local_id, turns in self.moving_npc_cooldowns.items():
                if turns > 1:
                    updated[local_id] = turns - 1
            self.moving_npc_cooldowns = updated
        self.last_map_id = world.map_id
        self.last_position = current_pos


def is_building_map(map_name: str) -> bool:
    tags = (
        "POKECENTER",
        "MART",
        "GYM",
        "SCHOOL",
        "MUSEUM",
        "HOUSE",
        "LAB",
        "ROOM",
        "ENTRANCE",
        "COTTAGE",
        "CENTER",
        "LOBBY",
        "DAYCARE",
        "BUILDING",
        "FAN_CLUB",
        "GAME_CORNER",
        "STORE",
        "CONDOMINIUMS",
        "RESTAURANT",
        "HOTEL",
        "DOJO",
        "HARBOR",
        "OFFICE",
    )
    return any(tag in map_name for tag in tags)


def is_outdoor_map(map_name: str) -> bool:
    tags = ("ROUTE", "CITY", "TOWN", "ISLAND", "FOREST", "SAFARI_ZONE", "PLATEAU")
    return any(tag in map_name for tag in tags)


def _edge_passable_count(collision: tuple[int, int, list[str]] | None) -> int:
    if not collision:
        return 0
    grid_w, grid_h, rows = collision
    seen = set()
    count = 0
    for x in range(grid_w):
        seen.add((x, 0))
        seen.add((x, grid_h - 1))
    for y in range(grid_h):
        seen.add((0, y))
        seen.add((grid_w - 1, y))
    for x, y in seen:
        if is_passable(rows[y][x]):
            count += 1
    return count


def is_probably_building(map_name: str,
                         collision: tuple[int, int, list[str]] | None = None) -> bool:
    """Infer indoor maps using names first, then collision edge openness."""
    if is_building_map(map_name):
        return True
    if is_outdoor_map(map_name):
        return False
    if map_name.startswith("MAP_"):
        return _edge_passable_count(collision) <= 2
    return _edge_passable_count(collision) <= 2


def is_probably_outdoor(map_name: str,
                        collision: tuple[int, int, list[str]] | None = None) -> bool:
    if is_outdoor_map(map_name):
        return True
    if is_building_map(map_name):
        return False
    return _edge_passable_count(collision) >= 4


def _hp_ratio(party: list[dict]) -> float:
    if not party:
        return 1.0
    lead = party[0]
    max_hp = max(lead.get("max_hp", 0), 1)
    return lead.get("hp", 0) / max_hp


def _combined_objective_text(world: WorldSnapshot) -> str:
    return f"{world.current_objective} {world.strategy_objective}".lower()


def _primary_objective_text(world: WorldSnapshot) -> str:
    text = (world.current_objective or "").strip().lower()
    if text:
        return text
    return (world.strategy_objective or "").strip().lower()


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _has_interaction_intent(text: str) -> bool:
    interaction_phrases = (
        "talk to",
        "speak to",
        "ask ",
        "interact",
        "press a",
        "nurse",
        "counter",
        "clerk",
    )
    return _has_any(text, interaction_phrases)


def _current_area_candidates(map_name: str) -> set[str]:
    parts = map_name.lower().split("_")
    candidates = {" ".join(parts)}
    if len(parts) >= 2:
        candidates.add(" ".join(parts[:2]))
    if len(parts) >= 3:
        candidates.add(" ".join(parts[:3]))
    if parts and parts[0] == "route" and len(parts) >= 2:
        candidates.add(f"route {parts[1]}")
    return {candidate.strip() for candidate in candidates if candidate.strip()}


def _objective_mentions_current_area(world: WorldSnapshot) -> bool:
    """Return True when the current objective text names the current area."""
    text = _primary_objective_text(world)
    if not text:
        return False

    candidates = _current_area_candidates(world.map_name)
    return any(candidate in text for candidate in candidates if candidate.strip())


def _objective_mentions_other_area(world: WorldSnapshot) -> bool:
    """Return True when the objective names a different common area."""
    text = _primary_objective_text(world)
    if not text:
        return False

    current_candidates = _current_area_candidates(world.map_name)
    area_names = (
        "pallet town",
        "viridian city",
        "pewter city",
        "cerulean city",
        "vermilion city",
        "lavender town",
        "celadon city",
        "fuchsia city",
        "saffron city",
        "cinnabar island",
        "indigo plateau",
        "viridian forest",
        "mt. moon",
        "diglett cave",
        "ss anne",
        "pokemon tower",
        "silph co",
        "victory road",
        "route 1",
        "route 2",
        "route 3",
        "route 4",
        "route 5",
        "route 6",
        "route 7",
        "route 8",
        "route 9",
        "route 10",
        "route 11",
        "route 12",
        "route 13",
        "route 14",
        "route 15",
        "route 16",
        "route 17",
        "route 18",
        "route 19",
        "route 20",
        "route 21",
        "route 22",
        "route 23",
        "route 24",
        "route 25",
    )
    for area_name in area_names:
        if area_name in text and area_name not in current_candidates:
            return True
    return False


def _has_local_destination_intent(world: WorldSnapshot) -> bool:
    """Return True when the objective implies a local building or NPC task."""
    text = _primary_objective_text(world)
    if not text:
        return False

    local_target_words = (
        "lab", "mart", "gym", "house", "center", "pokecenter", "pokemon center",
        "building", "school", "museum", "store", "room", "office",
        "deliver", "receive", "get the pokedex", "get pokedex", "parcel",
        "professor oak", "oak", "talk to", "speak to",
    )
    if not _has_any(text, local_target_words):
        return False
    if _objective_mentions_current_area(world):
        return True
    return not _objective_mentions_other_area(world)


def _should_stay_in_route_travel(world: WorldSnapshot) -> bool:
    """Keep route-exit stickiness only while the objective is still primarily about travel."""
    text = _primary_objective_text(world)
    if not text:
        return False
    if _has_local_destination_intent(world):
        return False
    return _has_any(text, (
        "walk south", "walk north", "walk east", "walk west",
        "head south", "head north", "head east", "head west",
        "go south", "go north", "go east", "go west",
        "leave", "exit", "route", "back to",
    ))


def _should_keep_talking(ctx: NavContext, world: WorldSnapshot) -> bool:
    if ctx.selected_goal_type in ("heal", "shop"):
        return _goal_matches_map(ctx.selected_goal_type, world.map_name)
    text = _primary_objective_text(world)
    if _has_any(text, (
        "walk south", "walk north", "walk east", "walk west",
        "head south", "head north", "head east", "head west",
        "go south", "go north", "go east", "go west",
        "back to", "leave", "exit", "outside",
    )):
        return False
    return _has_interaction_intent(text) or (
        is_probably_building(world.map_name, world.collision)
        and _objective_mentions_current_area(world)
        and _has_any(text, ("deliver", "receive", "get", "show", "give"))
    )


def _infer_nav_intent_fallback(world: WorldSnapshot) -> str | None:
    """Interpret current task into a structured overworld intent."""
    if world.game_state != "overworld" or world.in_battle or world.in_dialogue or not world.collision:
        return None

    text = _primary_objective_text(world)
    hp_ratio = _hp_ratio(world.party)
    low_hp = hp_ratio <= 0.35
    critical_hp = hp_ratio <= 0.2

    train_keywords = ("train", "grind", "level", "wild", "grass", "encounter")
    talk_keywords = ("talk to", "speak to", "ask ", "interact", "nurse", "counter", "clerk")
    object_keywords = ("pick up", "inspect", "trigger", "sign", "item", "object")
    building_keywords = ("heal", "pokecenter", "pokemon center", "mart", "gym", "lab", "house", "building")
    leave_keywords = ("leave", "exit", "outside", "head out", "go outside", "walk outside", "next area")
    travel_keywords = (
        "route", "walk south", "walk north", "walk east", "walk west",
        "head south", "head north", "head east", "head west",
        "go south", "go north", "go east", "go west",
        "pallet town", "viridian city", "pewter city", "cerulean city",
        "vermilion city", "lavender town", "saffron city", "celadon city",
        "fuchsia city", "cinnabar", "indigo plateau",
    )

    explicit_heal = _has_any(text, ("heal", "pokecenter", "pokemon center", "nurse"))
    explicit_building = _has_any(text, building_keywords) or _has_interaction_intent(text)
    explicit_train = _has_any(text, train_keywords)
    explicit_object = _has_any(text, object_keywords)
    explicit_travel = _has_any(text, travel_keywords + leave_keywords)
    local_destination = _has_local_destination_intent(world)
    no_party = len(world.party) == 0

    # Explicit task wording should win unless HP is truly critical.
    if explicit_train:
        return "train"
    if no_party and _has_trigger_or_item_interactable(world):
        return "interact_with_object"
    if explicit_object:
        return "interact_with_object"
    if explicit_heal:
        if "POKECENTER" in world.map_name:
            return "talk_to_npc"
        if is_probably_building(world.map_name, world.collision):
            return "leave_building"
        return "go_to_building"
    if local_destination:
        if is_probably_building(world.map_name, world.collision):
            return "talk_to_npc"
        return "go_to_building"
    if explicit_building and not _objective_mentions_other_area(world):
        if is_probably_building(world.map_name, world.collision):
            return "talk_to_npc"
        return "go_to_building"
    if explicit_travel and is_probably_outdoor(world.map_name, world.collision):
        return "go_to_route_exit"
    if explicit_building and not explicit_travel:
        if is_probably_building(world.map_name, world.collision):
            return "talk_to_npc"
        return "go_to_building"

    if critical_hp:
        if "POKECENTER" in world.map_name:
            return "talk_to_npc"
        if is_probably_building(world.map_name, world.collision):
            return "leave_building"
        return "go_to_building"

    if is_probably_building(world.map_name, world.collision):
        if _has_any(text, talk_keywords):
            return "talk_to_npc"
        if _has_any(text, train_keywords + leave_keywords):
            return "leave_building"
        if "POKECENTER" in world.map_name or "MART" in world.map_name:
            return "talk_to_npc"
        return "leave_building"

    if _has_any(text, building_keywords + talk_keywords):
        return "go_to_building"
    if _has_any(text, leave_keywords) and is_probably_outdoor(world.map_name, world.collision):
        return "go_to_route_exit"
    if low_hp and is_probably_building(world.map_name, world.collision):
        return "leave_building"

    return None


def infer_nav_intent(ctx: NavContext, world: WorldSnapshot) -> str | None:
    """Use the LLM to choose the next overworld intent, then fall back to simple rules."""
    if world.game_state != "overworld" or world.in_battle or world.in_dialogue or not world.collision:
        return None

    objective_key = "|".join((
        world.map_name,
        world.current_objective or "",
        world.strategy_objective or "",
        "indoor" if is_probably_building(world.map_name, world.collision) else "outdoor",
    ))
    should_refresh = (
        ctx.llm_intent_key != objective_key
        or ctx.llm_intent is None
        or ctx.stalled_ticks >= 2
    )
    if should_refresh:
        choice = classify_navigation_intent(
            map_name=world.map_name,
            current_objective=world.current_objective,
            strategy_objective=world.strategy_objective,
            indoor=is_probably_building(world.map_name, world.collision),
            hp_ratio=_hp_ratio(world.party),
            party_count=len(world.party),
            visible_doors=sum(1 for row in world.collision[2] for tile in row if tile == "D"),
            visible_objects=len(world.objects or []),
            visible_summary=_visible_interactable_summary(world),
            current_state=ctx.state.value,
            previous_intent=ctx.intent,
        )
        ctx.llm_intent_key = objective_key
        ctx.llm_intent = None
        ctx.llm_intent_reason = ""
        if choice:
            selected = choice.get("intent")
            if selected and selected != "none":
                if (
                    selected == "talk_to_npc"
                    and not is_probably_building(world.map_name, world.collision)
                ):
                    text = _primary_objective_text(world)
                    building_goal_words = (
                        "lab", "laboratory", "mart", "gym", "house", "center",
                        "pokecenter", "pokemon center", "building", "museum", "school",
                    )
                    if _has_any(text, building_goal_words):
                        selected = "go_to_building"
                    elif not _has_interaction_intent(text):
                        selected = "none"
                if (
                    len(world.party) == 0
                    and _has_trigger_or_item_interactable(world)
                    and selected in {"go_to_route_exit", "leave_building", "talk_to_npc", "none"}
                ):
                    selected = "interact_with_object"
                ctx.llm_intent = selected
                ctx.llm_intent_reason = choice.get("reason", "")

    if ctx.llm_intent:
        return ctx.llm_intent
    return _infer_nav_intent_fallback(world)


def _tile_at(world: WorldSnapshot, x: int, y: int) -> str | None:
    if not world.collision:
        return None
    grid_w, grid_h, rows = world.collision
    if 0 <= x < grid_w and 0 <= y < grid_h:
        return rows[y][x]
    return None


def _adjacent_tiles(world: WorldSnapshot) -> list[tuple[str, int, int, str | None]]:
    px, py = world.player_x, world.player_y
    results = []
    for direction, dx, dy in (("Up", 0, -1), ("Down", 0, 1), ("Left", -1, 0), ("Right", 1, 0)):
        tx, ty = px + dx, py + dy
        results.append((direction, tx, ty, _tile_at(world, tx, ty)))
    return results


def _find_adjacent_tile(world: WorldSnapshot, tile_kind: str) -> str | None:
    for direction, _tx, _ty, tile in _adjacent_tiles(world):
        if tile == tile_kind:
            return direction
    return None


def _choose_object_target(ctx: NavContext, world: WorldSnapshot) -> dict | None:
    objects = [obj for obj in (world.objects or []) if "Pokeball" not in obj.get("label", "")]
    if not objects:
        return None
    eligible_objects = [
        obj for obj in objects
        if ctx.moving_npc_cooldowns.get(obj.get("local_id", -1), 0) <= 0
    ]
    if eligible_objects:
        objects = eligible_objects

    if world.map_name.endswith("POKECENTER_1F") or "MART" in world.map_name:
        selected = sorted(
            objects,
            key=lambda obj: (obj["y"], abs(obj["x"] - world.player_x) + abs(obj["y"] - world.player_y)),
        )[0]
        ctx.selected_npc_reason = "Service-building counter NPC chosen before generic building randomization."
        return selected

    if is_probably_building(world.map_name, world.collision):
        min_interactions = min(obj.get("interaction_count", 0) for obj in objects)
        lowest_interaction = [
            obj for obj in objects
            if obj.get("interaction_count", 0) == min_interactions
        ]
        selected = random.choice(lowest_interaction)
        ctx.selected_npc_reason = (
            "Randomly chosen among the lowest-interaction NPCs in the building "
            f"({min_interactions} prior talks)."
        )
        return selected

    candidates = []
    for obj in objects:
        distance = abs(obj["x"] - world.player_x) + abs(obj["y"] - world.player_y)
        candidates.append({
            "local_id": obj["local_id"],
            "label": obj.get("label", "NPC"),
            "x": obj["x"],
            "y": obj["y"],
            "distance": distance,
            "interaction_count": obj.get("interaction_count", 0),
        })

    choice = select_npc_target(
        map_name=world.map_name,
        current_objective=world.current_objective,
        strategy_objective=world.strategy_objective,
        player_pos=(world.player_x, world.player_y),
        candidates=sorted(candidates, key=lambda cand: (cand["interaction_count"], cand["distance"])),
    )
    if choice:
        selected = next((obj for obj in objects if obj["local_id"] == choice["local_id"]), None)
        if selected:
            ctx.selected_npc_reason = choice.get("reason", "")
            return selected

    return min(
        objects,
        key=lambda obj: abs(obj["x"] - world.player_x) + abs(obj["y"] - world.player_y),
    )


def _visible_interactable_summary(world: WorldSnapshot) -> str:
    parts = []
    for obj in (world.objects or [])[:8]:
        parts.append(f"{obj.get('label', 'NPC')}@({obj['x']},{obj['y']})")
    for event in (world.bg_events or [])[:8]:
        parts.append(f"{event.get('label', event.get('type', 'event'))}@({event['x']},{event['y']})")
    return ", ".join(parts)


def _needs_healing(world: WorldSnapshot) -> bool:
    if not world.party:
        return False
    lead = world.party[0]
    return lead.get("hp", 0) < lead.get("max_hp", 0)


def _has_non_npc_interactables(world: WorldSnapshot) -> bool:
    for obj in world.objects or []:
        if obj.get("label", "NPC") != "NPC":
            return True
    return bool(world.bg_events)


def _has_trigger_or_item_interactable(world: WorldSnapshot) -> bool:
    for obj in world.objects or []:
        label = obj.get("label", "").lower()
        if obj.get("label") != "NPC" and any(keyword in label for keyword in ("trigger", "pokeball", "item")):
            return True
    for event in world.bg_events or []:
        label = event.get("label", "").lower()
        if event.get("type") == "coord" or any(keyword in label for keyword in ("trigger", "item", "pokeball")):
            return True
    return False


def _talk_to_npc_action(ctx: NavContext, world: WorldSnapshot) -> dict | None:
    target = None
    if ctx.target_object_id is not None:
        target = next((obj for obj in (world.objects or []) if obj["local_id"] == ctx.target_object_id), None)
        if target and target.get("interaction_count", 0) >= 3:
            alternatives = [
                obj for obj in (world.objects or [])
                if obj["local_id"] != ctx.target_object_id and obj.get("interaction_count", 0) < target.get("interaction_count", 0)
            ]
            if alternatives:
                target = None
                ctx.target_object_id = None
    if target is None:
        target = _choose_object_target(ctx, world)
        ctx.target_object_id = target["local_id"] if target else None
    if not target:
        return None
    if ctx.target_object_id != target.get("local_id"):
        ctx.target_object_last_pos = None
    current_target_pos = (target["x"], target["y"])
    if (
        ctx.target_object_id == target.get("local_id")
        and ctx.target_object_last_pos is not None
        and ctx.target_object_last_pos != current_target_pos
        and not world.in_dialogue
    ):
        ctx.moving_npc_cooldowns[target["local_id"]] = 3
        ctx.target_object_id = None
        ctx.target_object_last_pos = None
        ctx.selected_npc_reason = "Previous NPC target was moving; trying a different NPC."
        return decide_nav_action(ctx, world)
    ctx.target_object_last_pos = current_target_pos

    dist = abs(target["x"] - world.player_x) + abs(target["y"] - world.player_y)
    across_counter = (
        dist == 2
        and world.collision is not None
        and is_counter_talk_position(world.collision, world.player_x, world.player_y, target["x"], target["y"])
    )
    if dist == 1 or across_counter:
        if target.get("interaction_count", 0) >= 4 and not world.in_dialogue:
            alternatives = [
                obj for obj in (world.objects or [])
                if obj.get("local_id") != target.get("local_id")
                and obj.get("label") == "NPC"
                and obj.get("interaction_count", 0) < target.get("interaction_count", 0)
            ]
            if alternatives:
                ctx.target_object_id = None
                ctx.selected_npc_reason = ""
                ctx.intent = None
                ctx.state = NavState.FREE_EXPLORE
                ctx.llm_intent = None
                ctx.llm_intent_reason = ""
                ctx.llm_intent_key = None
                return decide_nav_action(ctx, world)
        ctx.state = NavState.BUILDING_TALK_TO_NPC
        # For counter-talk, face toward the counter (midpoint) since facing_direction only handles dist=1
        if across_counter:
            mid = (world.player_x + (target["x"] - world.player_x) // 2,
                   world.player_y + (target["y"] - world.player_y) // 2)
            face_btn = facing_direction((world.player_x, world.player_y), mid)
        else:
            face_btn = facing_direction((world.player_x, world.player_y), (target["x"], target["y"]))
        facing_map = {"Down": 1, "Up": 2, "Left": 3, "Right": 4}
        if face_btn and world.player_facing != facing_map.get(face_btn):
            return {
                "action": face_btn,
                "reason": (
                    f"Structured nav: face the chosen NPC, then interact ({ctx.selected_npc_reason})"
                    if ctx.selected_npc_reason else
                    "Structured nav: face the NPC, then interact"
                ),
                "display": "Talking to the nearby NPC.",
            }
        return {
            "action": "A",
            "reason": (
                f"Structured nav: interact with the chosen NPC ({ctx.selected_npc_reason})"
                if ctx.selected_npc_reason else
                "Structured nav: interact with the nearby NPC"
            ),
            "display": "Talking to the nearby NPC.",
        }

    ctx.state = NavState.BUILDING_PATH_TO_NPC
    path, target_obj, target_pos = path_to_adjacent_object(
        world.collision,
        world.player_x,
        world.player_y,
        world.objects or [],
        lambda obj: obj["local_id"] == ctx.target_object_id,
    )
    if target_obj:
        ctx.target_object_id = target_obj["local_id"]
        ctx.target_pos = target_pos
    step = _follow_path(path)
    if step:
        return {
            "action": step,
            "reason": (
                f"Structured nav: pathing next to the chosen NPC ({ctx.selected_npc_reason})"
                if ctx.selected_npc_reason else
                "Structured nav: pathing next to the target NPC"
            ),
            "display": "Walking over to an NPC.",
        }
    return None


def _choose_interactable_target(ctx: NavContext, world: WorldSnapshot) -> dict | None:
    candidates = []
    for obj in world.objects or []:
        label = obj.get("label", "object")
        if label == "NPC":
            continue
        candidates.append({
            "id": f"obj:{obj['local_id']}",
            "local_id": obj["local_id"],
            "label": label,
            "x": obj["x"],
            "y": obj["y"],
            "distance": abs(obj["x"] - world.player_x) + abs(obj["y"] - world.player_y),
            "interaction_count": obj.get("interaction_count", 0),
            "kind": "walk_on" if "trigger" in label.lower() else "press_a",
        })

    for event in world.bg_events or []:
        label = event.get("label", event.get("type", "event"))
        event_type = event.get("type", "bg")
        kind = "walk_on" if event_type == "coord" or "trigger" in label.lower() else "press_a"
        candidates.append({
            "id": f"bg:{event_type}:{event['x']}:{event['y']}:{label}",
            "local_id": None,
            "label": label,
            "x": event["x"],
            "y": event["y"],
            "distance": abs(event["x"] - world.player_x) + abs(event["y"] - world.player_y),
            "interaction_count": 0,
            "kind": kind,
            "source": "bg_event",
        })

    if not candidates:
        return None

    # Generic early-game preference: if the player has no party and a visible item-like
    # interactable exists, prioritize the direct object over nearby trigger tiles.
    if not world.party:
        item_like = [
            cand for cand in candidates
            if any(keyword in cand["label"].lower() for keyword in ("pokeball", "item"))
        ]
        if item_like:
            nearest_distance = min(cand["distance"] for cand in item_like)
            nearest_items = [cand for cand in item_like if cand["distance"] == nearest_distance]
            selected = random.choice(nearest_items)
            ctx.selected_object_reason = "Randomly chosen starter/item interactable among visible nearby options."
            return selected

    choice = select_interactable_target(
        map_name=world.map_name,
        current_objective=world.current_objective,
        strategy_objective=world.strategy_objective,
        party_count=len(world.party),
        player_pos=(world.player_x, world.player_y),
        candidates=sorted(candidates, key=lambda cand: (cand["interaction_count"], cand["distance"])),
    )
    if choice:
        selected = next((cand for cand in candidates if cand["id"] == choice["id"]), None)
        if selected:
            ctx.selected_object_reason = choice.get("reason", "")
            return selected

    if not world.party:
        selected = min(
            candidates,
            key=lambda cand: (
                0 if any(keyword in cand["label"].lower() for keyword in ("pokeball", "item")) else 1,
                0 if cand.get("kind") != "walk_on" else 1,
                cand["interaction_count"],
                cand["distance"],
            ),
        )
    else:
        selected = min(candidates, key=lambda cand: (cand["interaction_count"], cand["distance"]))
    ctx.selected_object_reason = ""
    return selected


def _follow_path(path: list[str] | None) -> str | None:
    if path:
        return path[0]
    return None


def _training_step(world: WorldSnapshot) -> str | None:
    """Take a small deterministic step while remaining inside grass if possible."""
    preferred = ["Right", "Left", "Up", "Down"]
    adjacent = {direction: tile for direction, _tx, _ty, tile in _adjacent_tiles(world)}
    for direction in preferred:
        if adjacent.get(direction) == "G":
            return direction
    for direction in preferred:
        if adjacent.get(direction) and adjacent.get(direction) != "1":
            return direction
    return None


def _step_away_from_object(world: WorldSnapshot, target: dict) -> str | None:
    """Take a small reset step away from an overused adjacent target."""
    tx, ty = target["x"], target["y"]
    preferred = []
    if tx > world.player_x:
        preferred.append("Left")
    elif tx < world.player_x:
        preferred.append("Right")
    if ty > world.player_y:
        preferred.append("Up")
    elif ty < world.player_y:
        preferred.append("Down")
    for fallback in ("Up", "Down", "Left", "Right"):
        if fallback not in preferred:
            preferred.append(fallback)

    adjacent = {direction: tile for direction, _x, _y, tile in _adjacent_tiles(world)}
    for direction in preferred:
        tile = adjacent.get(direction)
        if tile and is_passable(tile) and tile != "D":
            return direction
    return None


def _goal_type_from_objective(world: WorldSnapshot) -> str:
    text = _combined_objective_text(world)
    if any(keyword in text for keyword in ("heal", "pokecenter", "pokemon center", "nurse", "counter")):
        return "heal"
    if any(keyword in text for keyword in ("mart", "shop", "buy", "sell", "clerk")):
        return "shop"
    if any(keyword in text for keyword in ("professor oak", "oak", "pokedex", "parcel", "lab", "laboratory")):
        return "lab"
    if any(keyword in text for keyword in ("gym", "leader", "badge")):
        return "gym"
    if any(keyword in text for keyword in ("house", "home", "room", "cottage")):
        return "house"
    if any(keyword in text for keyword in ("museum", "school", "office", "fan club", "store")):
        return "building"
    return "generic"


def _preferred_route_side(world: WorldSnapshot) -> str | None:
    text = _primary_objective_text(world)
    if not text:
        return None
    if "walk south" in text or "head south" in text or "go south" in text:
        return "south"
    if "walk north" in text or "head north" in text or "go north" in text:
        return "north"
    if "walk west" in text or "head west" in text or "go west" in text:
        return "west"
    if "walk east" in text or "head east" in text or "go east" in text:
        return "east"
    return None


def _goal_matches_map(goal_type: str, map_name: str) -> bool:
    if goal_type == "heal":
        return "POKECENTER" in map_name
    if goal_type == "shop":
        return "MART" in map_name
    if goal_type == "lab":
        return "LAB" in map_name
    if goal_type == "gym":
        return "GYM" in map_name
    if goal_type == "house":
        return any(tag in map_name for tag in ("HOUSE", "ROOM", "COTTAGE"))
    if goal_type == "building":
        return is_building_map(map_name)
    if goal_type == "generic":
        return False
    return is_probably_building(map_name)


def _door_key(map_id: int, door: tuple[int, int]) -> str:
    return f"{map_id}:{door[0]},{door[1]}"


def _goal_key(map_id: int, goal_type: str) -> str:
    return f"{map_id}:{goal_type}"


def _parse_door_key(door_key: str) -> tuple[int, int]:
    coords = door_key.split(":", 1)[1]
    x_str, y_str = coords.split(",", 1)
    return int(x_str), int(y_str)


def _discovered_buildings_for_map(ctx: NavContext, map_id: int) -> list[dict]:
    discoveries = []
    for door_key, indoor_map in ctx.known_doors.items():
        if not door_key.startswith(f"{map_id}:"):
            continue
        x, y = _parse_door_key(door_key)
        discoveries.append({"x": x, "y": y, "map_name": indoor_map})
    discoveries.sort(key=lambda item: (item["y"], item["x"]))
    return discoveries


def _reachable_route_exits(world: WorldSnapshot, max_candidates: int = 6) -> list[dict]:
    if not world.collision:
        return []

    grid_w, grid_h, rows = world.collision
    allowed_sides = {
        conn.get("direction")
        for conn in (world.connections or [])
        if conn.get("direction") in {"north", "south", "east", "west"}
    }
    segments: list[dict] = []

    def collect(side: str, points: list[tuple[int, int]]) -> None:
        current: list[tuple[int, int]] = []
        last_coord = None
        for x, y in points:
            tile = rows[y][x]
            variable = x if side in ("north", "south") else y
            if is_passable(tile):
                if current and last_coord is not None and variable != last_coord + 1:
                    segments.append({"side": side, "points": current[:]})
                    current = []
                current.append((x, y))
                last_coord = variable
            elif current:
                segments.append({"side": side, "points": current[:]})
                current = []
                last_coord = None
            else:
                last_coord = None
        if current:
            segments.append({"side": side, "points": current[:]})

    collect("north", [(x, 0) for x in range(grid_w)])
    collect("south", [(x, grid_h - 1) for x in range(grid_w)])
    collect("west", [(0, y) for y in range(grid_h)])
    collect("east", [(grid_w - 1, y) for y in range(grid_h)])

    candidates = []
    seen = set()
    for segment in segments:
        if allowed_sides and segment["side"] not in allowed_sides:
            continue
        points = segment["points"]
        if not points:
            continue
        center = points[len(points) // 2]
        if center in seen:
            continue
        seen.add(center)
        path = path_to_target_tile(world.collision, world.player_x, world.player_y, center[0], center[1])
        if path is None:
            continue
        candidates.append({
            "side": segment["side"],
            "x": center[0],
            "y": center[1],
            "distance": len(path),
            "span": len(points),
            "path": path,
        })

    candidates.sort(key=lambda cand: cand["distance"])
    return candidates[:max_candidates]


def _select_route_target(ctx: NavContext, world: WorldSnapshot) -> tuple[tuple[int, int] | None, str]:
    candidates = _reachable_route_exits(world)
    if not candidates:
        return None, ""
    rejected = ctx.rejected_route_exits.get(world.map_id, set())
    candidates = [cand for cand in candidates if (cand["x"], cand["y"]) not in rejected]
    if not candidates:
        return None, "all candidate route exits rejected"
    preferred_side = _preferred_route_side(world)
    if preferred_side:
        preferred = [cand for cand in candidates if cand["side"] == preferred_side]
        if preferred:
            candidates = preferred
    if len(candidates) == 1:
        only = candidates[0]
        return (only["x"], only["y"]), "only reachable route exit"

    choice = select_route_exit(
        map_name=world.map_name,
        objective=world.current_objective or world.strategy_objective,
        player_pos=(world.player_x, world.player_y),
        candidates=candidates,
    )
    if choice:
        selected = next((cand for cand in candidates if cand["x"] == choice["x"] and cand["y"] == choice["y"]), None)
        if selected:
            return (selected["x"], selected["y"]), choice.get("reason", "LLM-selected route exit")

    best = candidates[0]
    return (best["x"], best["y"]), "nearest reachable route exit fallback"


def _reachable_doors(world: WorldSnapshot, max_candidates: int = 8) -> list[dict]:
    if not world.collision:
        return []
    grid_w, grid_h, rows = world.collision
    candidates = []
    for y in range(grid_h):
        for x in range(grid_w):
            if rows[y][x] != "D":
                continue
            path = path_to_target_tile(world.collision, world.player_x, world.player_y, x, y)
            if path is None:
                continue
            candidates.append({
                "x": x,
                "y": y,
                "path": path,
                "distance": len(path),
            })
    candidates.sort(key=lambda cand: cand["distance"])
    return candidates[:max_candidates]


def _select_building_target(ctx: NavContext, world: WorldSnapshot) -> tuple[tuple[int, int] | None, str]:
    goal_type = _goal_type_from_objective(world)
    ctx.selected_goal_type = goal_type
    goal_key = _goal_key(world.map_id, goal_type)
    discoveries = _discovered_buildings_for_map(ctx, world.map_id)

    known_matches = []
    if goal_type != "generic":
        for door_key, indoor_map in ctx.known_doors.items():
            if not door_key.startswith(f"{world.map_id}:"):
                continue
            if _goal_matches_map(goal_type, indoor_map):
                known_matches.append(_parse_door_key(door_key))
    if known_matches:
        return known_matches[0], "discovered matching entrance"

    candidates = _reachable_doors(world)
    rejected = ctx.rejected_doors.get(goal_key, set())
    candidates = [cand for cand in candidates if _door_key(world.map_id, (cand["x"], cand["y"])) not in rejected]
    discovery_lookup = {(item["x"], item["y"]): item["map_name"] for item in discoveries}
    for cand in candidates:
        cand["label"] = discovery_lookup.get((cand["x"], cand["y"]), "unknown")
    if not candidates:
        return None, "all reachable doors already rejected for this goal"
    if len(candidates) == 1:
        return (candidates[0]["x"], candidates[0]["y"]), "only reachable entrance"

    choice = select_building_door(
        map_name=world.map_name,
        objective=world.current_objective or world.strategy_objective,
        player_pos=(world.player_x, world.player_y),
        candidates=candidates,
        rejected=[_parse_door_key(key) for key in rejected],
        discoveries=discoveries,
    )
    if choice:
        selected = next((cand for cand in candidates if cand["x"] == choice["x"] and cand["y"] == choice["y"]), None)
        if selected:
            return (selected["x"], selected["y"]), choice.get("reason", "LLM-selected entrance")

    best = candidates[0]
    return (best["x"], best["y"]), "nearest reachable entrance fallback"


def _verify_selected_building(ctx: NavContext, world: WorldSnapshot) -> str | None:
    if ctx.selected_door is None or ctx.selected_door_origin_map is None or ctx.selected_goal_type is None:
        return None
    if world.map_id == ctx.selected_door_origin_map:
        return None

    key = _door_key(ctx.selected_door_origin_map, ctx.selected_door)
    goal_key = _goal_key(ctx.selected_door_origin_map, ctx.selected_goal_type)
    ctx.known_doors[key] = world.map_name
    if _goal_matches_map(ctx.selected_goal_type, world.map_name):
        ctx.last_building_result = f"entered {world.map_name}"
        ctx.selected_door = None
        ctx.selected_door_origin_map = None
        ctx.selected_goal_type = None
        ctx.selected_door_reason = ""
        return None

    ctx.rejected_doors.setdefault(goal_key, set()).add(key)
    ctx.force_leave_building = True
    ctx.last_building_result = f"wrong building: {world.map_name}"
    ctx.selected_door = None
    ctx.selected_door_origin_map = None
    ctx.selected_goal_type = None
    ctx.selected_door_reason = ""
    return ctx.last_building_result


def decide_nav_action(ctx: NavContext, world: WorldSnapshot) -> dict | None:
    """Return a deterministic action when a structured overworld state applies."""
    ctx.note_progress(world)
    verification_warning = _verify_selected_building(ctx, world)
    outside_now = is_probably_outdoor(world.map_name, world.collision)
    if ctx.force_leave_building and outside_now:
        ctx.force_leave_building = False
        ctx.pending_exit_tile = None
        if ctx.state == NavState.BUILDING_LEAVE:
            ctx.state = NavState.FREE_EXPLORE
            ctx.target_pos = None
    if ctx.pending_exit_tile == (world.player_x, world.player_y):
        ctx.state = NavState.BUILDING_LEAVE
        return {
            "action": "Down",
            "reason": "Structured nav: second step through the exit to leave the building",
            "display": "Stepping out of the building.",
        }
    if (
        world.game_state == "overworld"
        and not world.in_battle
        and not world.in_dialogue
        and world.objects
        and ("POKECENTER" in world.map_name or "MART" in world.map_name)
        and (ctx.selected_goal_type in ("heal", "shop") or _needs_healing(world))
    ):
        intent = "talk_to_npc"
        ctx.intent = intent
    else:
        intent = infer_nav_intent(ctx, world)
    if ctx.force_leave_building:
        intent = "leave_building"
    if (
        len(world.party) == 0
        and _has_trigger_or_item_interactable(world)
        and intent in {None, "go_to_route_exit", "leave_building", "talk_to_npc"}
    ):
        intent = "interact_with_object"
    if world.game_state == "overworld" and not world.in_battle and not world.in_dialogue:
        if ctx.state in (NavState.PATH_TO_BUILDING, NavState.ENTER_BUILDING) and not is_probably_building(world.map_name, world.collision):
            intent = "go_to_building"
        elif (
            ctx.state in (NavState.PATH_TO_ROUTE_EXIT, NavState.EXIT_ROUTE)
            and not is_probably_building(world.map_name, world.collision)
            and _should_stay_in_route_travel(world)
        ):
            intent = "go_to_route_exit"
        elif ctx.state in (NavState.TRAIN_FIND_GRASS, NavState.TRAIN_IN_GRASS):
            intent = "train"
        elif (
            ctx.state in (NavState.BUILDING_PATH_TO_NPC, NavState.BUILDING_TALK_TO_NPC)
            and is_probably_building(world.map_name, world.collision)
            and _should_keep_talking(ctx, world)
        ):
            intent = "talk_to_npc"
        elif ctx.state == NavState.BUILDING_LEAVE and not outside_now:
            intent = "leave_building"
    ctx.intent = intent

    if intent != "go_to_route_exit":
        ctx.selected_route_exit = None
        ctx.selected_route_reason = ""
    if intent != "talk_to_npc":
        ctx.target_object_id = None
        ctx.selected_npc_reason = ""
        ctx.target_object_last_pos = None
    if intent != "interact_with_object":
        ctx.selected_object_reason = ""
        ctx.target_object_kind = None

    if intent is None:
        ctx.state = NavState.FREE_EXPLORE
        ctx.target_pos = None
        ctx.target_object_id = None
        return None

    current_tile = _tile_at(world, world.player_x, world.player_y)

    if intent == "go_to_building":
        if is_probably_building(world.map_name, world.collision):
            intent = "talk_to_npc"
        else:
            if ctx.selected_door_origin_map != world.map_id:
                ctx.selected_door = None
                ctx.selected_door_origin_map = None
                ctx.selected_goal_type = None
                ctx.selected_door_reason = ""
            if ctx.selected_door is None:
                ctx.state = NavState.SELECT_BUILDING_TARGET
                selected, reason = _select_building_target(ctx, world)
                if not selected:
                    ctx.state = NavState.FREE_EXPLORE
                    ctx.target_pos = None
                    ctx.selected_door = None
                    ctx.selected_door_origin_map = None
                    ctx.selected_door_reason = ""
                    return None
                ctx.selected_door = selected
                ctx.selected_door_origin_map = world.map_id
                ctx.selected_door_reason = reason
            ctx.state = NavState.PATH_TO_BUILDING
            ctx.target_pos = ctx.selected_door
            if current_tile == "D" and ctx.selected_door == (world.player_x, world.player_y):
                ctx.state = NavState.ENTER_BUILDING
                return {
                    "action": "Up",
                    "reason": f"Structured nav: enter selected building ({ctx.selected_door_reason or 'chosen entrance'})",
                    "display": "Entering the selected building.",
                }
            toward_selected = None
            for direction, tx, ty, tile in _adjacent_tiles(world):
                if tile == "D" and (tx, ty) == ctx.selected_door:
                    toward_selected = direction
                    break
            if toward_selected:
                return {
                    "action": toward_selected,
                    "reason": f"Structured nav: move onto the selected entrance ({ctx.selected_door_reason or 'chosen door'})",
                    "display": "Moving to the chosen building door.",
                }
            path = path_to_target_tile(
                world.collision,
                world.player_x,
                world.player_y,
                ctx.selected_door[0],
                ctx.selected_door[1],
            )
            step = _follow_path(path)
            if step:
                return {
                    "action": step,
                    "reason": f"Structured nav: pathing to selected entrance ({ctx.selected_door_reason or 'chosen door'})",
                    "display": "Pathing to the chosen building.",
                }
            return None

    if intent == "go_to_route_exit":
        if is_probably_building(world.map_name, world.collision):
            intent = "leave_building"
        else:
            if ctx.selected_route_exit is None:
                selected, reason = _select_route_target(ctx, world)
                if selected:
                    ctx.selected_route_exit = selected
                    ctx.selected_route_reason = reason
            target = ctx.selected_route_exit
            if target is None:
                path, target = path_to_map_edge(world.collision, world.player_x, world.player_y)
                ctx.target_pos = target
                step = _follow_path(path)
                if step:
                    ctx.state = NavState.PATH_TO_ROUTE_EXIT
                    return {
                        "action": step,
                        "reason": "Structured nav: fallback pathing to the nearest route exit",
                        "display": "Heading for a route exit.",
                    }
                return None

            ctx.target_pos = target
            path = path_to_target_tile(
                world.collision,
                world.player_x,
                world.player_y,
                target[0],
                target[1],
            )
            if target == (world.player_x, world.player_y):
                ctx.state = NavState.EXIT_ROUTE
                action = edge_exit_action(target, world.collision)
                if ctx.stalled_ticks >= 1:
                    ctx.rejected_route_exits.setdefault(world.map_id, set()).add(target)
                    ctx.selected_route_exit = None
                    ctx.selected_route_reason = ""
                    selected, reason = _select_route_target(ctx, world)
                    if selected:
                        ctx.selected_route_exit = selected
                        ctx.selected_route_reason = reason
                        path = path_to_target_tile(
                            world.collision,
                            world.player_x,
                            world.player_y,
                            selected[0],
                            selected[1],
                        )
                        step = _follow_path(path)
                        if step:
                            ctx.state = NavState.PATH_TO_ROUTE_EXIT
                            return {
                                "action": step,
                                "reason": "Structured nav: previous route exit was blocked, replanning to a different exit",
                                "display": "Changing course to a different exit.",
                            }
                    return None
                if action:
                    return {
                        "action": action,
                        "reason": f"Structured nav: walk off the selected route exit ({ctx.selected_route_reason or 'chosen exit'})",
                        "display": "Leaving the area.",
                    }
            ctx.state = NavState.PATH_TO_ROUTE_EXIT
            if path and len(path) == 1:
                ctx.selected_route_exit = target
            step = _follow_path(path)
            if step:
                return {
                    "action": step,
                    "reason": f"Structured nav: pathing to selected route exit ({ctx.selected_route_reason or 'chosen exit'})",
                    "display": "Heading for a route exit.",
                }
            return None

    if intent == "train":
        if is_probably_building(world.map_name, world.collision):
            intent = "leave_building"
        elif current_tile == "G":
            ctx.state = NavState.TRAIN_IN_GRASS
            step = _training_step(world)
            if step:
                return {
                    "action": step,
                    "reason": "Structured nav: moving through grass to find battles",
                    "display": "Training in tall grass.",
                }
            return None
        else:
            ctx.state = NavState.TRAIN_FIND_GRASS
            path, target = path_to_nearest_grass(world.collision, world.player_x, world.player_y)
            if not path and not target:
                path, target = path_to_map_edge(world.collision, world.player_x, world.player_y)
                ctx.state = NavState.PATH_TO_ROUTE_EXIT
                ctx.target_pos = target
                step = _follow_path(path)
                if step:
                    return {
                        "action": step,
                        "reason": "Structured nav: no grass here, heading for a route exit to keep searching",
                        "display": "No grass here, leaving to find a better route.",
                    }
                return None
            else:
                ctx.target_pos = target
                step = _follow_path(path)
                if step:
                    return {
                        "action": step,
                        "reason": "Structured nav: pathing to the nearest tall grass",
                        "display": "Looking for grass to train in.",
                    }
                return None

    if intent == "talk_to_npc":
        return _talk_to_npc_action(ctx, world)

    if intent == "interact_with_object":
        target = _choose_interactable_target(ctx, world)
        if not target:
            return None

        ctx.target_object_id = target["local_id"]
        ctx.target_object_kind = target["kind"]

        if target["kind"] == "walk_on":
            ctx.state = NavState.PATH_TO_OBJECT
            ctx.target_pos = (target["x"], target["y"])
            if (world.player_x, world.player_y) == ctx.target_pos:
                ctx.state = NavState.INTERACT_OBJECT
                return None
            path = path_to_target_tile(
                world.collision,
                world.player_x,
                world.player_y,
                target["x"],
                target["y"],
            )
            step = _follow_path(path)
            if step:
                return {
                    "action": step,
                    "reason": (
                        f"Structured nav: pathing onto the chosen interactable ({ctx.selected_object_reason})"
                        if ctx.selected_object_reason else
                        "Structured nav: pathing onto the chosen interactable"
                    ),
                    "display": "Moving to an important trigger or object.",
                }
            return None

        dist = abs(target["x"] - world.player_x) + abs(target["y"] - world.player_y)
        if dist == 1:
            if target.get("interaction_count", 0) >= 3 and not world.in_dialogue:
                ctx.target_object_id = None
                ctx.target_object_kind = None
                ctx.selected_object_reason = ""
                ctx.intent = None
                ctx.state = NavState.FREE_EXPLORE
                ctx.llm_intent = None
                ctx.llm_intent_reason = ""
                ctx.llm_intent_key = None
                return decide_nav_action(ctx, world)
            ctx.state = NavState.INTERACT_OBJECT
            face_btn = facing_direction((world.player_x, world.player_y), (target["x"], target["y"]))
            facing_map = {"Down": 1, "Up": 2, "Left": 3, "Right": 4}
            if face_btn and world.player_facing != facing_map.get(face_btn):
                return {
                    "action": face_btn,
                    "reason": (
                        f"Structured nav: face the chosen object, then interact ({ctx.selected_object_reason})"
                        if ctx.selected_object_reason else
                        "Structured nav: face the chosen object, then interact"
                    ),
                    "display": "Inspecting an object.",
                }
            return {
                "action": "A",
                "reason": (
                    f"Structured nav: interact with the chosen object ({ctx.selected_object_reason})"
                    if ctx.selected_object_reason else
                    "Structured nav: interact with the chosen object"
                ),
                "display": "Interacting with an object.",
            }

        ctx.state = NavState.PATH_TO_OBJECT
        if target.get("source") == "bg_event":
            path = path_to_target_tile(
                world.collision,
                world.player_x,
                world.player_y,
                target["x"],
                target["y"],
            )
            step = _follow_path(path)
            if step:
                return {
                    "action": step,
                    "reason": (
                        f"Structured nav: pathing toward the chosen event tile ({ctx.selected_object_reason})"
                        if ctx.selected_object_reason else
                        "Structured nav: pathing toward the chosen event tile"
                    ),
                    "display": "Walking to an event tile.",
                }
        else:
            path, _target_obj, target_pos = path_to_adjacent_object(
                world.collision,
                world.player_x,
                world.player_y,
                world.objects or [],
                lambda obj: obj["local_id"] == target["local_id"],
            )
            ctx.target_pos = target_pos
            step = _follow_path(path)
            if step:
                return {
                    "action": step,
                    "reason": (
                        f"Structured nav: pathing next to the chosen object ({ctx.selected_object_reason})"
                        if ctx.selected_object_reason else
                        "Structured nav: pathing next to the chosen object"
                    ),
                    "display": "Walking over to an object.",
                }
        return None

    if intent == "leave_building":
        ctx.state = NavState.BUILDING_LEAVE
        if current_tile == "D":
            ctx.pending_exit_tile = None
            return {
                "action": "Down",
                "reason": (
                    f"Structured nav: leave {ctx.last_building_result} and continue search"
                    if verification_warning or ctx.force_leave_building
                    else "Structured nav: step through the building exit"
                ),
                "display": "Leaving the building.",
            }
        toward_door = _find_adjacent_tile(world, "D")
        if toward_door:
            for _direction, tx, ty, tile in _adjacent_tiles(world):
                if _direction == toward_door and tile == "D":
                    ctx.pending_exit_tile = (tx, ty)
                    break
            return {
                "action": toward_door,
                "reason": "Structured nav: move onto the exit tile",
                "display": "Heading for the exit.",
            }
        path, target = path_to_nearest_door(world.collision, world.player_x, world.player_y)
        ctx.target_pos = target
        if path and len(path) == 1 and target:
            ctx.pending_exit_tile = target
        step = _follow_path(path)
        if step:
            return {
                "action": step,
                "reason": "Structured nav: pathing to the building exit",
                "display": "Walking to the exit.",
            }
        return None

    return None

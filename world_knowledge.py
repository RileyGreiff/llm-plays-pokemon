"""Persistent world knowledge — doors, NPCs, and map edges discovered during play."""

import json
import os

SAVE_PATH = "logs/world_knowledge.json"


class WorldKnowledge:
    def __init__(self):
        self.doors: dict[str, dict] = {}      # "map_id:x,y" -> {label, destination_map}
        self.npcs: dict[str, dict] = {}        # "map_id:local_id" -> {label, map_name, last_dialogue}
        self.map_edges: dict[str, dict] = {}   # "map_id:direction" -> {destination_map}
        self._load()

    # --- Persistence ---

    def _load(self):
        if not os.path.exists(SAVE_PATH):
            return
        try:
            with open(SAVE_PATH, "r") as f:
                raw = json.load(f)
            self.doors = raw.get("doors", {})
            self.npcs = raw.get("npcs", {})
            self.map_edges = raw.get("map_edges", {})
        except (json.JSONDecodeError, ValueError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as f:
            json.dump({
                "doors": self.doors,
                "npcs": self.npcs,
                "map_edges": self.map_edges,
            }, f, indent=2)

    # --- Doors ---

    def _door_key(self, map_id: int, x: int, y: int) -> str:
        return f"{map_id}:{x},{y}"

    def ensure_door(self, map_id: int, x: int, y: int):
        """Register a door tile as known but unexplored."""
        key = self._door_key(map_id, x, y)
        if key not in self.doors:
            self.doors[key] = {"label": "unknown", "destination_map": None}

    def learn_door(self, from_map_id: int, from_x: int, from_y: int,
                   destination_map_name: str):
        """Label a door after walking through it."""
        key = self._door_key(from_map_id, from_x, from_y)
        # Make a human-friendly label from the map name
        label = destination_map_name.replace("_", " ").title()
        self.doors[key] = {
            "label": label,
            "destination_map": destination_map_name,
        }

    def get_door_label(self, map_id: int, x: int, y: int) -> str:
        key = self._door_key(map_id, x, y)
        entry = self.doors.get(key)
        if entry and entry.get("destination_map"):
            return entry["label"]
        return "unknown"

    def get_doors_on_map(self, map_id: int) -> list[dict]:
        """Get all known doors on a given map with their labels."""
        results = []
        prefix = f"{map_id}:"
        for key, entry in self.doors.items():
            if key.startswith(prefix):
                coords = key[len(prefix):]
                x, y = coords.split(",")
                results.append({
                    "x": int(x),
                    "y": int(y),
                    "label": entry.get("label", "unknown"),
                    "destination_map": entry.get("destination_map"),
                })
        return results

    # --- NPCs ---

    def _npc_key(self, map_id: int, local_id: int) -> str:
        return f"{map_id}:{local_id}"

    def learn_npc(self, map_id: int, local_id: int, map_name: str,
                  dialogue_snippet: str = ""):
        """Label an NPC after talking to them."""
        key = self._npc_key(map_id, local_id)
        existing = self.npcs.get(key, {})
        self.npcs[key] = {
            "label": existing.get("label", "NPC"),
            "map_name": map_name,
            "last_dialogue": dialogue_snippet[:120] if dialogue_snippet else existing.get("last_dialogue", ""),
        }

    def get_npc_label(self, map_id: int, local_id: int) -> str | None:
        """Get NPC info. Returns None if never talked to."""
        key = self._npc_key(map_id, local_id)
        entry = self.npcs.get(key)
        if not entry:
            return None
        dialogue = entry.get("last_dialogue", "")
        if dialogue:
            return f'said: "{dialogue}"'
        return "talked before"

    # --- Map Edges ---

    def _edge_key(self, map_id: int, direction: str) -> str:
        return f"{map_id}:{direction}"

    def learn_map_edges(self, map_id: int, connections: list[dict]):
        """Store map edge connections (from emulator get_map_connections)."""
        for conn in connections:
            direction = conn.get("direction", "")
            dest_name = conn.get("map_name", "")
            if direction and dest_name:
                key = self._edge_key(map_id, direction)
                self.map_edges[key] = {"destination_map": dest_name}

    def get_map_edges(self, map_id: int) -> list[dict]:
        """Get known edge connections for a map."""
        results = []
        prefix = f"{map_id}:"
        for key, entry in self.map_edges.items():
            if key.startswith(prefix):
                direction = key[len(prefix):]
                dest = entry.get("destination_map", "unknown")
                label = dest.replace("_", " ").title()
                results.append({
                    "direction": direction,
                    "destination_map": dest,
                    "label": label,
                })
        return results

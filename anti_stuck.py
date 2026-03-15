"""Loop detection, position watchdog, and recovery logic."""

import time


class AntiStuck:
    def __init__(self, repeat_threshold: int = 10, stuck_timeout: int = 120):
        self.repeat_threshold = repeat_threshold  # same action N times → warning
        self.stuck_timeout = stuck_timeout  # seconds with no position change → stuck

        self.last_position = None
        self.position_changed_at = time.time()
        self.force_sonnet_until = 0  # action count until which Sonnet is forced

    def check(self, recent_actions: list[dict], game_state: dict,
              action_count: int) -> tuple[str | None, bool]:
        """Check for stuck conditions.

        Returns:
            (warning_message or None, force_sonnet)
        """
        warning = None
        force_sonnet = action_count < self.force_sonnet_until

        # Skip repetition/oscillation checks during dialogue or battle
        in_dialogue = game_state.get("in_dialogue", False)
        in_battle = game_state.get("in_battle", False)
        if in_dialogue or in_battle:
            return warning, force_sonnet

        # --- Repetition detection ---
        rep_warning = self._check_repetition(recent_actions)
        if rep_warning:
            warning = rep_warning
            self.force_sonnet_until = action_count + 5

        # --- Position watchdog ---
        pos_warning = self._check_position(game_state)
        if pos_warning:
            warning = pos_warning if not warning else f"{warning} {pos_warning}"
            self.force_sonnet_until = action_count + 5

        force_sonnet = action_count < self.force_sonnet_until
        return warning, force_sonnet

    def _check_repetition(self, recent_actions: list[dict]) -> str | None:
        """Detect if the same action has been repeated too many times,
        or if a short pattern is repeating (oscillation)."""
        if len(recent_actions) < self.repeat_threshold:
            return None

        last_n = recent_actions[-self.repeat_threshold:]
        actions = [a["action"] for a in last_n]

        # Single button repeat
        if len(set(actions)) == 1:
            return (f"You have pressed {actions[0]} {self.repeat_threshold} times "
                    f"in a row. You MUST try a different button.")

        # Oscillation detection: check if a 2-4 button pattern is looping
        last_20 = [a["action"] for a in recent_actions[-20:]]
        if len(last_20) >= 12:
            for pattern_len in (2, 3, 4):
                pattern = last_20[-pattern_len:]
                repeats = 0
                for i in range(len(last_20) - pattern_len, -1, -pattern_len):
                    chunk = last_20[i:i + pattern_len]
                    if chunk == pattern:
                        repeats += 1
                    else:
                        break
                if repeats >= 3:
                    cycle = " -> ".join(pattern)
                    return (f"You are stuck in a repeating cycle: {cycle} "
                            f"(repeated {repeats}x). This pattern is NOT making progress. "
                            f"STOP and try something completely different. "
                            f"If dialogue keeps blocking you, it's a HINT — you need to do "
                            f"something else in this area first (interact with an object, "
                            f"talk to someone, pick up an item). "
                            f"Look at the minimap for unvisited '.' tiles and go explore them.")

        return None

    def _check_position(self, game_state: dict) -> str | None:
        """Detect if player position hasn't changed."""
        current_pos = (game_state.get("player_x"), game_state.get("player_y"),
                       game_state.get("map_id"))

        if current_pos != self.last_position:
            self.last_position = current_pos
            self.position_changed_at = time.time()
            return None

        elapsed = time.time() - self.position_changed_at
        if elapsed > self.stuck_timeout:
            self.position_changed_at = time.time()  # reset to avoid spamming
            return (f"Your position has not changed for {int(elapsed)} seconds. "
                    f"You appear to be physically stuck. Try a different direction or interact with something.")

        return None

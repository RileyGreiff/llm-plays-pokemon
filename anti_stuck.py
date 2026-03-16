"""Stuck detection and short random-movement recovery logic."""


class AntiStuck:
    def __init__(self, no_move_threshold: int = 5, recovery_turns: int = 10):
        self.no_move_threshold = no_move_threshold
        self.recovery_turns_default = recovery_turns
        self.last_position = None
        self.no_move_turns = 0
        self.random_recovery_remaining = 0

    def check(self, recent_actions: list[dict], game_state: dict,
              action_count: int) -> tuple[str | None, bool]:
        """Check for stuck conditions.

        Returns:
            (warning_message or None, force_sonnet)
        """
        del recent_actions, action_count
        warning = None
        force_sonnet = False

        # Skip movement-based stuck checks while the player is not really in
        # free overworld control.
        if game_state.get("in_dialogue", False) or game_state.get("in_battle", False):
            return warning, force_sonnet

        warning = self._check_no_movement(game_state)
        return warning, force_sonnet

    def _check_no_movement(self, game_state: dict) -> str | None:
        current_pos = (
            game_state.get("player_x"),
            game_state.get("player_y"),
            game_state.get("map_id"),
        )

        if self.last_position is None or current_pos != self.last_position:
            self.last_position = current_pos
            self.no_move_turns = 0
            return None

        self.no_move_turns += 1
        if self.random_recovery_remaining > 0:
            return (
                f"Random recovery active: {self.random_recovery_remaining} "
                f"action(s) remaining."
            )

        if self.no_move_turns >= self.no_move_threshold:
            self.random_recovery_remaining = self.recovery_turns_default
            return (
                f"You have not moved for {self.no_move_turns} turns. "
                f"Entering random recovery for {self.recovery_turns_default} actions."
            )

        return None

    def in_random_recovery(self) -> bool:
        return self.random_recovery_remaining > 0

    def consume_random_recovery_turn(self) -> bool:
        if self.random_recovery_remaining <= 0:
            return False
        self.random_recovery_remaining -= 1
        if self.random_recovery_remaining == 0:
            self.no_move_turns = 0
        return True

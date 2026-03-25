"""Curriculum scheduler — controls which flag to train and when to advance.

Flags unlock sequentially. Mastery = success rate > 80% AND avg steps < 2x optimal
over the last 10 episodes. The model trains exclusively on in-progress flags.
"""

import json
import os
from dataclasses import dataclass, field
from story_flags import STORY_FLAGS, FLAG_BY_ID, FLAG_ORDER


@dataclass
class FlagStats:
    """Per-flag training statistics."""
    episodes: list = field(default_factory=list)  # list of {success, steps}
    mastered: bool = False
    optimal_steps: int | None = None  # set from first success

    def add_episode(self, success: bool, steps: int):
        self.episodes.append({"success": success, "steps": steps})
        # Keep only last 20
        if len(self.episodes) > 20:
            self.episodes = self.episodes[-20:]
        # Set optimal from first success
        if success and self.optimal_steps is None:
            self.optimal_steps = steps

    def check_mastery(self) -> bool:
        """Check if flag is mastered: last 10 episodes, >80% success, avg steps < 2x optimal."""
        recent = self.episodes[-10:]
        if len(recent) < 10:
            return False

        successes = sum(1 for e in recent if e["success"])
        success_rate = successes / len(recent)
        if success_rate < 0.8:
            return False

        if self.optimal_steps is None:
            return False

        successful_steps = [e["steps"] for e in recent if e["success"]]
        if not successful_steps:
            return False

        avg_steps = sum(successful_steps) / len(successful_steps)
        return avg_steps < 2.0 * self.optimal_steps

    @property
    def success_rate(self) -> float:
        recent = self.episodes[-10:]
        if not recent:
            return 0.0
        return sum(1 for e in recent if e["success"]) / len(recent)

    @property
    def avg_steps(self) -> float:
        recent = [e["steps"] for e in self.episodes[-10:] if e["success"]]
        if not recent:
            return float("inf")
        return sum(recent) / len(recent)

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)


class CurriculumScheduler:
    """Manages flag progression and selects which flag to train."""

    def __init__(self, save_path: str = "curriculum_state.json"):
        self.save_path = save_path
        self.stats: dict[str, FlagStats] = {}
        self.unlocked: set[str] = set()

        # Initialize stats for all flags
        for flag in STORY_FLAGS:
            self.stats[flag.id] = FlagStats()

        # First flag is always unlocked
        if FLAG_ORDER:
            self.unlocked.add(FLAG_ORDER[0])

    def get_active_flag_id(self) -> str | None:
        """Get the current flag to train.

        Selects the earliest unlocked, non-mastered flag.
        Returns None if all flags are mastered.
        """
        for flag_id in FLAG_ORDER:
            if flag_id in self.unlocked and not self.stats[flag_id].mastered:
                return flag_id
        return None

    def report_episode(self, flag_id: str, success: bool, steps: int):
        """Report episode results and check for mastery/unlock."""
        stats = self.stats[flag_id]
        stats.add_episode(success, steps)

        # Check mastery
        if not stats.mastered and stats.check_mastery():
            stats.mastered = True
            print(f"  [curriculum] FLAG MASTERED: {flag_id} "
                  f"(optimal={stats.optimal_steps}, avg={stats.avg_steps:.0f})")

            # Unlock next flag
            flag = FLAG_BY_ID[flag_id]
            idx = FLAG_ORDER.index(flag_id)
            if idx + 1 < len(FLAG_ORDER):
                next_id = FLAG_ORDER[idx + 1]
                self.unlocked.add(next_id)
                print(f"  [curriculum] UNLOCKED: {next_id}")

    def overall_mastery(self) -> float:
        """Fraction of flags mastered (0.0 to 1.0)."""
        if not FLAG_ORDER:
            return 0.0
        mastered = sum(1 for fid in FLAG_ORDER if self.stats[fid].mastered)
        return mastered / len(FLAG_ORDER)

    def get_flag_mastery(self, flag_id: str) -> float:
        """Get mastery score for a specific flag (0.0 to 1.0).

        Based on success rate of recent episodes. Used for exploration decay.
        """
        stats = self.stats[flag_id]
        if stats.mastered:
            return 1.0
        return min(stats.success_rate, 1.0)

    def status_summary(self) -> str:
        """Human-readable status."""
        lines = []
        for flag_id in FLAG_ORDER:
            stats = self.stats[flag_id]
            flag = FLAG_BY_ID[flag_id]
            status = "MASTERED" if stats.mastered else (
                "TRAINING" if flag_id in self.unlocked else "LOCKED"
            )
            episodes = stats.total_episodes
            sr = stats.success_rate
            avg = stats.avg_steps if stats.avg_steps != float("inf") else "N/A"
            lines.append(f"  {status:8s} | {flag.name:30s} | ep={episodes:4d} | sr={sr:.0%} | avg={avg}")
        return "\n".join(lines)

    def save(self):
        """Save curriculum state to disk."""
        data = {
            "unlocked": list(self.unlocked),
            "stats": {},
        }
        for flag_id, stats in self.stats.items():
            data["stats"][flag_id] = {
                "episodes": stats.episodes,
                "mastered": stats.mastered,
                "optimal_steps": stats.optimal_steps,
            }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load curriculum state from disk."""
        if not os.path.exists(self.save_path):
            return
        with open(self.save_path) as f:
            data = json.load(f)
        self.unlocked = set(data.get("unlocked", []))
        for flag_id, sdata in data.get("stats", {}).items():
            if flag_id in self.stats:
                self.stats[flag_id].episodes = sdata.get("episodes", [])
                self.stats[flag_id].mastered = sdata.get("mastered", False)
                self.stats[flag_id].optimal_steps = sdata.get("optimal_steps")
        # Ensure first flag is unlocked
        if FLAG_ORDER:
            self.unlocked.add(FLAG_ORDER[0])

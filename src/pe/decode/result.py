"""Generation result with the metrics the benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenResult:
    output_ids: list[int]
    steps: int  # verification iterations
    target_calls: int  # target forward passes
    drafter_calls: int  # drafter forward passes
    accepted_tokens: int  # drafted tokens accepted (excludes bonus tokens)
    num_generated: int
    seconds: float = 0.0

    @property
    def acceptance_length(self) -> float:
        """Mean tokens committed per verification iteration."""
        return self.num_generated / max(1, self.steps)

    @property
    def target_calls_per_token(self) -> float:
        return self.target_calls / max(1, self.num_generated)

    @property
    def tokens_per_second(self) -> float:
        return self.num_generated / self.seconds if self.seconds > 0 else 0.0

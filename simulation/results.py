"""Simulation result records (Design Document §11).

A simulation run is one possible future account history (§11). The per-run
records here are deliberately auditable: `BannerResult.copy_wishes` and
`BannerResult.account_after` make a history explainable after the fact -
the same information the legacy Monte Carlo prototype recorded per run
(wishes at each featured copy, resulting pity/guarantee) - instead of
reducing a run to "success = True".
"""

from dataclasses import dataclass

from domain import Account, Banner, Goal
from simulation.strategy import SpendPlan


@dataclass(frozen=True)
class GoalOutcome:
    """One roadmap goal's standing at the end of one history (§11).

    Attributes:
        goal: the roadmap objective (§5).
        satisfied: True when the simulated account meets the goal's
            constellation at the end of the history.
        satisfied_after: the first banner after which the goal was satisfied;
            None when the goal was already satisfied before the simulation
            began (e.g. already owned) or when it was never satisfied
            (disambiguated by `satisfied`).
    """

    goal: Goal
    satisfied: bool
    satisfied_after: Banner | None


@dataclass(frozen=True)
class BannerResult:
    """One banner's slice of one simulated history (§11).

    Attributes:
        banner: the processed roadmap banner.
        target_constellation: the plan's desired resulting constellation,
            or None when the plan skips this banner. Never a copy count
            (§4.2, §12).
        budget: the plan's spending cap for this banner (0 when skipped).
        copies_needed: copies between the simulated ownership and the target
            at banner start (§10.4); 0 when skipped or when the account is
            already at or above the target.
        income_credited: version-level income credited on reaching this
            banner (0 for the current banner; see simulation.engine for the
            timing rule, §16).
        wishes_spent: min(budget, available wishes, wishes the target still
            needed) (§12).
        copies_obtained: featured copies pulled on this banner.
        copy_wishes: 1-based within-banner wish index of each featured copy.
        target_met: True when a target was pursued and met - including the
            account already being at or above it (copies_needed 0). False
            when the target was missed, and False when no target was pursued:
            a skipped banner did not "fail" (see target_constellation).
        account_after: the simulated account state leaving this banner.
    """

    banner: Banner
    target_constellation: int | None
    budget: int
    copies_needed: int
    income_credited: int
    wishes_spent: int
    copies_obtained: int
    copy_wishes: tuple[int, ...]
    target_met: bool
    account_after: Account


@dataclass(frozen=True)
class RunResult:
    """One possible future account history (§11).

    Attributes:
        banner_results: one entry per processed roadmap banner, in
            chronological order (§7).
        account_after: the account state at the end of the history (equals
            the last banner's account_after).
        goal_outcomes: every roadmap goal, in priority order (§5), with its
            standing at the end of the history.

    Roadmap-wide satisfaction is deliberately not stored here:
    `all(outcome.satisfied for outcome in goal_outcomes)` derives it without
    duplicate state; SimulationResult.all_goals_probability is where the
    derived aggregate belongs.
    """

    banner_results: tuple[BannerResult, ...]
    account_after: Account
    goal_outcomes: tuple[GoalOutcome, ...]


@dataclass(frozen=True)
class GoalProbability:
    """How often one goal was satisfied across simulated histories (§11)."""

    goal: Goal
    probability: float


@dataclass(frozen=True)
class BannerAggregate:
    """One banner's behavior across simulated histories (§11)."""

    banner: Banner
    target_constellation: int | None
    planned_budget: int
    mean_income_credited: float
    mean_wishes_spent: float
    target_met_probability: float
    mean_copies_obtained: float


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated outcome of many simulated histories (§11).

    Attributes:
        runs: how many histories were simulated.
        seed: the rng seed used (an identical seed reproduces the result).
        plan: the executed spending plan (§12).
        goals: one entry per roadmap goal, in priority order (§5).
        banners: one entry per processed banner, in chronological order (§7).
        all_goals_probability: fraction of histories in which every roadmap
            goal was satisfied.
        final_wishes_mean / final_wishes_min / final_wishes_max: the wish
            pool at the end of each history.
    """

    runs: int
    seed: int | None
    plan: SpendPlan
    goals: tuple[GoalProbability, ...]
    banners: tuple[BannerAggregate, ...]
    all_goals_probability: float
    final_wishes_mean: float
    final_wishes_min: int
    final_wishes_max: int

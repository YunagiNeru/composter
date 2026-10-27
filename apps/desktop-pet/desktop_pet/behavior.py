from __future__ import annotations

from dataclasses import dataclass
from typing import Any


QUESTIONS: dict[str, dict[str, Any]] = {
    "gait_style": {
        "type": "choice",
        "instructions": "Choose the single best fixed gait action for this structured pet state.",
        "criteria": {
            "RESTING": "Remain still and rest.",
            "SLOW_WALK": "Walk slowly and calmly.",
            "DROWSY_SHUFFLE": "Shuffle while sleepy, with an unsteady gait.",
            "BRISK_WALK": "Walk briskly and attentively.",
            "RUN": "Run energetically for a short movement burst.",
        },
    },
    "expression": {
        "type": "choice",
        "instructions": "Choose the single best fixed expression action for this structured pet state.",
        "criteria": {
            "SLEEPY": "Heavy eyelids and a relaxed sleepy face.",
            "CALM": "Relaxed neutral eyes and a calm face.",
            "ALERT": "Open attentive eyes and an alert face.",
            "CHEERFUL": "Bright eyes and a cheerful face.",
        },
    },
    "meow_style": {
        "type": "choice",
        "instructions": "Choose the single best fixed meow action for this structured pet state.",
        "criteria": {
            "NONE": "Do not meow.",
            "SLEEPY_MEOW": "Use a quiet sleepy meow.",
            "SHORT_MEOW": "Use a brief bright meow.",
            "PLAYFUL_MEOW": "Use a playful varied meow.",
        },
    },
    "social_response": {
        "type": "choice",
        "instructions": "Choose the single best fixed social response action for this structured pet state.",
        "criteria": {
            "LEAN_IN": "Lean in toward a trusted person.",
            "SLOW_BLINK": "Offer a relaxed slow blink.",
            "TOLERATE": "Stay calm without seeking contact.",
            "GENTLE_WITHDRAW": "Move away gently when cautious.",
        },
    },
}


@dataclass(frozen=True)
class PetMetrics:
    captured_at: str
    cpu_percent: float | None
    ram_percent: float | None
    gpu_percent: float | None
    network_rx_level: str | None
    network_tx_level: str | None
    network_rx_rate: float | None
    network_tx_rate: float | None
    disk_used_percent: float | None
    disk_busy_percent: float | None
    unavailable: tuple[str, ...] = ()

    def as_model_state(self, pet: dict[str, Any], user_event: str = "NONE") -> str:
        def value(number: float | None) -> str:
            return "UNAVAILABLE" if number is None else f"{number:.1f}"

        fields = (
            ("cpu_usage_percent", value(self.cpu_percent)),
            ("ram_usage_percent", value(self.ram_percent)),
            ("gpu_usage_percent", value(self.gpu_percent)),
            ("network_rx_level", self.network_rx_level or "UNAVAILABLE"),
            ("network_tx_level", self.network_tx_level or "UNAVAILABLE"),
            ("disk_usage_percent", value(self.disk_used_percent)),
            ("disk_activity_percent", value(self.disk_busy_percent)),
            ("wakefulness", "SLEEPY" if self.ram_percent is not None and self.ram_percent < 30
             else "ALERT" if self.ram_percent is not None and self.ram_percent >= 70 else "CALM"),
            ("personality", str(pet.get("personality", "AFFECTIONATE"))),
            ("pet_mood", str(pet.get("mood", "CALM"))),
            ("bond_level", str(pet.get("bond", 0))),
            ("satisfaction", str(pet.get("satisfaction", 0))),
            ("user_event", user_event),
        )
        return "\n".join(f"{key}: {item}" for key, item in fields)


@dataclass(frozen=True)
class AppliedBehavior:
    animation: str
    gait_style: str
    expression: str
    meow_style: str
    social_response: str
    movement_rate: float
    ear_rate: float
    body_fullness: float
    display_area_percent: float
    requested_actions: dict[str, str]
    applied_actions: dict[str, str]
    source: str
    warnings: tuple[str, ...] = ()


def compose_behavior(metrics: PetMetrics, pet: dict[str, Any], settings: dict[str, Any],
                     requested: dict[str, str] | None, *, rule_only: bool = False) -> AppliedBehavior:
    requested = dict(requested or {})
    warnings: list[str] = []
    ram = metrics.ram_percent
    cpu = metrics.cpu_percent
    gpu = metrics.gpu_percent
    awake = 0.5 if ram is None else min(1.0, max(0.0, ram / 100.0))
    movement_rate = 0.05 if cpu is None else 0.08 + 1.75 * min(1.0, max(0.0, cpu / 100.0))
    rx = _level_value(metrics.network_rx_level)
    tx = _level_value(metrics.network_tx_level)
    ear_rate = 0.2 + 3.0 * rx
    fullness = 0.5 if metrics.disk_used_percent is None else min(
        1.0, max(0.0, metrics.disk_used_percent / 100.0)
    )
    max_area = float(settings.get("window", {}).get("max_area_percent", 3.0))
    gpu_level = 0.5 if gpu is None else min(1.0, max(0.0, gpu / 100.0))
    area_percent = max_area * (0.25 + 0.75 * gpu_level)

    safe_actions: dict[str, str] = {}
    safe_actions["gait_style"] = _candidate(requested, "gait_style", {
        "RESTING", "SLOW_WALK", "DROWSY_SHUFFLE", "BRISK_WALK", "RUN"
    }, "SLOW_WALK" if cpu is None or cpu < 20 else "BRISK_WALK")
    safe_actions["expression"] = _candidate(requested, "expression", {
        "SLEEPY", "CALM", "ALERT", "CHEERFUL"
    }, "SLEEPY" if awake < 0.3 else "ALERT" if awake >= 0.7 else "CALM")
    safe_actions["meow_style"] = _candidate(requested, "meow_style", {
        "NONE", "SLEEPY_MEOW", "SHORT_MEOW", "PLAYFUL_MEOW"
    }, "SLEEPY_MEOW" if awake < 0.3 else "SHORT_MEOW")
    safe_actions["social_response"] = _candidate(requested, "social_response", {
        "LEAN_IN", "SLOW_BLINK", "TOLERATE", "GENTLE_WITHDRAW"
    }, "SLOW_BLINK" if pet.get("personality") == "AFFECTIONATE" else "TOLERATE")
    if not rule_only:
        for key in QUESTIONS:
            if key not in requested:
                warnings.append(f"AI_ACTION_UNAVAILABLE:{key}")
    if cpu is not None and cpu >= 75 and awake < 0.3:
        safe_actions["gait_style"] = "DROWSY_SHUFFLE"
    if awake < 0.3:
        safe_actions["expression"] = "SLEEPY"
        safe_actions["meow_style"] = "SLEEPY_MEOW"
    elif awake >= 0.7 and safe_actions["expression"] == "SLEEPY":
        safe_actions["expression"] = "ALERT"
    if tx >= 0.66:
        required_meow = "SLEEPY_MEOW" if awake < 0.3 else "SHORT_MEOW" if awake >= 0.7 else None
        if required_meow is not None:
            safe_actions["meow_style"] = required_meow
        elif safe_actions["meow_style"] == "NONE":
            safe_actions["meow_style"] = "SHORT_MEOW"

    animation = _animation_for(safe_actions, cpu, metrics, pet)
    if not rule_only and requested and safe_actions != requested:
        warnings.append("SAFETY_CONSTRAINT_APPLIED")
    source = "RULE_BASELINE" if rule_only else "SOM_NPU"
    return AppliedBehavior(
        animation=animation, gait_style=safe_actions["gait_style"],
        expression=safe_actions["expression"], meow_style=safe_actions["meow_style"],
        social_response=safe_actions["social_response"], movement_rate=movement_rate,
        ear_rate=ear_rate, body_fullness=fullness, display_area_percent=area_percent,
        requested_actions=requested, applied_actions=safe_actions, source=source,
        warnings=tuple(warnings),
    )


def _candidate(requested: dict[str, str], key: str, allowed: set[str], fallback: str) -> str:
    value = requested.get(key)
    return value if isinstance(value, str) and value in allowed else fallback


def _level_value(level: str | None) -> float:
    return {None: 0.0, "LOW": 0.15, "MEDIUM": 0.55, "HIGH": 1.0}.get(level, 0.0)


def _animation_for(actions: dict[str, str], cpu: float | None, metrics: PetMetrics,
                   pet: dict[str, Any]) -> str:
    if metrics.disk_busy_percent is not None and metrics.disk_busy_percent >= 60:
        return "dig"
    if actions["social_response"] == "GENTLE_WITHDRAW":
        return "pet_dislike"
    if actions["social_response"] in {"LEAN_IN", "SLOW_BLINK"} and pet.get("mood") == "HAPPY":
        return "pet_happy"
    if actions["meow_style"] in {"SLEEPY_MEOW", "SHORT_MEOW", "PLAYFUL_MEOW"}:
        return "sleepy_meow" if actions["meow_style"] == "SLEEPY_MEOW" else "meow"
    if actions["expression"] == "SLEEPY":
        return "sleepy_idle"
    if cpu is not None and cpu >= 80:
        return "drowsy_walk" if actions["gait_style"] == "DROWSY_SHUFFLE" else "run"
    if cpu is not None and cpu >= 30:
        return "walk"
    return "normal_idle"

from dataclasses import dataclass, field

from backend.engine.strategy_interpreter.models import StrategyDefinition

CanonicalStrategy = StrategyDefinition


@dataclass
class StrategyInterpretation:
    strategy: CanonicalStrategy
    missing_information: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

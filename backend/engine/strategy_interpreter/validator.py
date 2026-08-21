from .models import StrategyDefinition


class StrategyValidator:

    def validate(self, strategy: StrategyDefinition):

        errors = []

        if not strategy.name:
            errors.append("Strategy name is required.")

        if not strategy.symbol:
            errors.append("Symbol is required.")

        if not strategy.timeframe:
            errors.append("Timeframe is required.")

        if strategy.direction not in {"long", "short", "both"}:
            errors.append("Invalid strategy direction.")

        if not strategy.entry_conditions:
            errors.append("At least one entry condition is required.")

        if strategy.risk_percent <= 0:
            errors.append("Risk percent must be greater than zero.")

        if strategy.risk_reward <= 0:
            errors.append("Risk/reward must be greater than zero.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
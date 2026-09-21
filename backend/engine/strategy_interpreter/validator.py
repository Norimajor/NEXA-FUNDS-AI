from .models import StrategyDefinition, SUPPORTED_TIMEFRAMES


SUPPORTED_OPERATORS = {
    ">",
    "<",
    ">=",
    "<=",
    "=",
    "cross_above",
    "cross_below",
}


def _validate_exit(errors, label, exit_definition):
    if exit_definition is None:
        return
    if not isinstance(exit_definition, dict):
        errors.append(f"{label} must be an object or null.")
        return
    exit_type = exit_definition.get("type")
    required_values = {
        "percentage": "value",
        "atr": "multiplier",
        "risk_multiple": "multiple",
    }
    if exit_type not in required_values:
        errors.append(f"Invalid {label} type.")
        return
    required_key = required_values[exit_type]
    value = exit_definition.get(required_key)
    try:
        if float(value) <= 0:
            errors.append(f"{label} value must be greater than zero.")
    except (TypeError, ValueError):
        errors.append(f"{label} value must be numeric.")


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

        if not isinstance(strategy.timeframe, str) or strategy.timeframe.upper() not in SUPPORTED_TIMEFRAMES:
            errors.append("Unsupported strategy timeframe.")

        for condition_group in (strategy.entry_conditions, strategy.exit_conditions, strategy.filters):
            for condition in condition_group:
                if condition.operator not in SUPPORTED_OPERATORS:
                    errors.append(f"Invalid operator '{condition.operator}'.")

        try:
            risk_percent = float(strategy.risk_percent)
        except (TypeError, ValueError):
            risk_percent = None
        if risk_percent is None or risk_percent <= 0:
            errors.append("Risk percent must be greater than zero.")

        try:
            risk_reward = float(strategy.risk_reward)
        except (TypeError, ValueError):
            risk_reward = None
        if risk_reward is None or risk_reward <= 0:
            errors.append("Risk/reward must be greater than zero.")

        _validate_exit(errors, "Stop loss", strategy.stop_loss)
        _validate_exit(errors, "Take profit", strategy.take_profit)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
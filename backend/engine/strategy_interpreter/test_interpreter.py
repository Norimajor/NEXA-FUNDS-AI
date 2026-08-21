from .parser import StrategyParser
from .validator import StrategyValidator


print("=" * 40)
print("       NEXA FUNDS AI")
print("   STRATEGY INTERPRETER TEST")
print("=" * 40)

text = """
Buy when EMA 20 crosses above EMA 100,
RSI 14 above 55,
and ADX 14 above 25.
"""

parser = StrategyParser()
validator = StrategyValidator()

strategy = parser.parse(text)
validation = validator.validate(strategy)

print()
print("INPUT STRATEGY:")
print(text)

print("VALIDATION:")
print(validation)

print()
print("PARSED STRATEGY:")
print(strategy)

print()
print("ENTRY CONDITIONS:")

for condition in strategy.entry_conditions:
    print(condition)

if validation["valid"]:
    print()
    print("NEXA FUNDS AI STRATEGY INTERPRETER OK")
else:
    print()
    print("NEXA FUNDS AI STRATEGY INTERPRETER FAILED")
from .registry import ConditionRegistry


print("=" * 40)
print("       NEXA FUNDS AI")
print("   CONDITION REGISTRY TEST")
print("=" * 40)

registry = ConditionRegistry()

print()
print("Total conditions:")
print(len(registry.all()))

print()
print("Categories:")

for category in registry.categories():
    conditions = registry.by_category(category)

    print(f"{category}: {len(conditions)}")

print()
print("Testing RSI:")

rsi = registry.get("RSI")
print(rsi)

print()
print("Testing alias 'fair value gap':")

fvg = registry.find_by_alias("fair value gap")
print(fvg)

print()
print("Testing alias 'liquidity sweep':")

liquidity = registry.find_by_alias("liquidity sweep")
print(liquidity)

print()
print("NEXA FUNDS AI CONDITION REGISTRY OK")
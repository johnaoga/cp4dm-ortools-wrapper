"""
Example: Frequent Itemset Mining using Oscar ML globals with CPMpy.

Demonstrates:
1. Loading data with PatternDataset.
2. Creating CPMpy BoolVars for items.
3. Posting FrequentItemset as a native Oscar ML global.
4. Mixing with a CPMpy cardinality constraint (sum <= k).
5. Solving with CPM_oscar_ml.
6. Enumerating all solutions.
"""

from pathlib import Path

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import FrequentItemset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

# Load dataset
data_path = Path(__file__).parent.parent / "tests" / "fixtures" / "fim" / "test.txt"
data = PatternDataset.from_file(data_path, format=TdbFormat())

print(f"Dataset: {data.name}")
print(f"  Transactions: {data.n_transactions}")
print(f"  Items: {data.n_items - 1} (real items 1..{data.n_items - 1})")
print(f"  Density: {data.density():.2%}")
print()

# Create CPMpy model
items = cp.boolvar(shape=data.n_items, name="I")
model = cp.Model()

# Post Oscar ML FIM global: items must form a frequent itemset with support >= 3
minsup = 3
model += FrequentItemset(items, minsup=minsup, data=data)

# Additional CPMpy constraint: at most 2 items selected
model += cp.sum(items) <= 2

# Solve with CPM_oscar_ml (the only solver that handles Oscar globals natively)
solver = CPM_oscar_ml(model)

print(f"Finding all frequent itemsets with support >= {minsup} and size <= 2:")
print("-" * 50)

count = 0


def display_solution():
    global count
    count += 1
    selected = [i for i in range(data.n_items) if items[i].value() == 1]
    # Compute support for display
    vertical = data.as_vertical()
    coverage = set(range(data.n_transactions))
    for item in selected:
        coverage &= vertical[item]
    print(f"  Pattern {count}: items={selected}, support={len(coverage)}")


solver.solveAll(display=display_solution)

print("-" * 50)
print(f"Total frequent itemsets found: {count}")
print(f"Search stats: {solver.stats.nodes} nodes, {solver.stats.failures} failures")

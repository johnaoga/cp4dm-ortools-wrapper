# cp4dm-cpmpy-oscar-ml

Oscar ML constraint propagation for pattern mining and classification trees,
integrated with [CPMpy](https://github.com/CPMpy/cpmpy).

## Overview

This package migrates the Scala Oscar ML module into Python, built around CPMpy
as the modeling layer. Oscar ML global constraints are implemented as **native
propagators** — they are NOT decomposed into large declarative encodings.

### Supported constraint domains

- **Frequent Itemset Mining (FIM)**: `FrequentItemset`, `ClosedFrequentItemset`, `CoverSize`, `CoverClosure`, `ZDC`
- **Sequential Pattern Mining (SPM)**: `SequentialPattern`, `TimedSequentialPattern` (PPIC, PPICt, PPDC, PPmixed)
- **Frequent Episode Mining (FEM)**: `FrequentEpisode`, `TimedFrequentEpisode`
- **Classification Trees**: `TreeCoverSizeSR`, `SplitPossible`, `SplitUseful`, `DummyNode` + AND/OR search

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### High-level API (planned)

```python
from cp4dm_cpmpy_oscar_ml import PatternDataset
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

data = PatternDataset.from_file("retail.txt", format=TdbFormat())
```

### CPMpy-native API

```python
import cpmpy as cp
from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import FrequentItemset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

data = PatternDataset.from_file("test.txt", format=TdbFormat())
items = cp.boolvar(shape=data.n_items, name="item")
model = cp.Model()

model += FrequentItemset(items, minsup=3, data=data)
model += cp.sum(items) <= 3

solver = CPM_oscar_ml(model)
solver.solveAll(display=lambda: print([i for i in range(data.n_items) if items[i].value()]))
```

## Architecture

Oscar ML globals require the dedicated `CPM_oscar_ml` solver. Using them with
other CPMpy backends without explicit permission will raise `NotSupportedError`.

## License

LGPL-2.1-or-later (following Oscar's license).

## Attribution

Based on Oscar (https://bitbucket.org/oscarlib/oscar/wiki/Home).
Original authors: John Aoga, Pierre Schaus.

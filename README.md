# DequeDict

Ordered dictionary with O(1) deque operations at both ends.

## Features

- **O(1) key lookup** — like `dict`
- **O(1) pop/peek from both ends** — like `deque`
- **O(1) appendleft, move_to_end, delete by key**
- **Full typing support** with `.pyi` stubs

## Installation

```bash
pip install dequedict
```

## Usage

```python
from dequedict import DequeDict, DefaultDequeDict

dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

dd["d"] = 4              # Set
value = dd["a"]          # Get
del dd["b"]              # Delete

dd.peekleft()            # First value
dd.popleft()             # Remove first
dd.appendleft("z", 0)    # Insert at front
dd.move_to_end("a")      # Move to back

# DefaultDequeDict: auto-create values like defaultdict
groups = DefaultDequeDict(list)
groups["a"].append(1)
groups["a"].append(2)
```

## API

| Method | Description |
|--------|-------------|
| `peekleft()` / `peek()` | First/last value |
| `peekleftitem()` / `peekitem()` | First/last (key, value) |
| `popleft()` / `pop()` | Remove and return first/last |
| `popleftitem()` / `popitem()` | Remove and return first/last pair |
| `appendleft(key, value)` | Insert at front |
| `move_to_end(key, last=True)` | Move to front or back |
| `get`, `keys`, `values`, `items`, `clear`, `copy`, `update`, `setdefault` | Standard dict ops |

## Performance

Mac M1, Python 3.11, C extension:

| Operation | DequeDict | dict | deque | OrderedDict |
|-----------|-----------|------|-------|-------------|
| Key lookup | 32 ns | **23 ns** | O(n) | 24 ns |
| Peek left | **22 ns** | N/A | 23 ns | 59 ns |
| Peek right | **22 ns** | N/A | 23 ns | 70 ns |
| Pop left ×100 | 3.7 µs | N/A | **2.1 µs** | 7.7 µs |
| Delete by key ×100 | 4.3 µs | **2.8 µs** | O(n) | 6.0 µs |
| Insert ×100 | **6.6 µs** | 9.5 µs | N/A | 6.8 µs |

Pop is ~1.8x slower than deque due to dict maintenance, but provides O(1) key lookup.

## Benchmarks

```bash
python benchmarks/benchmark.py
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

MIT

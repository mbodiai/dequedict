"""Typing stubs for dequedict."""
from typing import TypeVar, Iterator, ItemsView, KeysView, ValuesView, Callable, overload
from collections.abc import Iterable, Mapping

K = TypeVar("K")
V = TypeVar("V")

class DequeDict(Mapping[K, V]):
    """Ordered dictionary with O(1) deque operations at both ends.

    Combines dict interface with deque-like operations:
    - O(1) popleft/popleftitem/peekleft/peekleftitem
    - O(1) pop/popitem/peek/peekitem (right side)
    - O(1) appendleft (insert at front)
    - O(1) lookup by key
    - Maintains insertion order
    """

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: K) -> V: ...
    def __setitem__(self, key: K, value: V) -> None: ...
    def __delitem__(self, key: K) -> None: ...
    def __iter__(self) -> Iterator[K]: ...
    def __reversed__(self) -> Iterator[K]: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

    # Deque-like operations - O(1)
    def peekleft(self) -> V:
        """Return first value without removing - O(1)."""
        ...

    def peekleftitem(self) -> tuple[K, V]:
        """Return first (key, value) without removing - O(1)."""
        ...

    def peekleftkey(self) -> K:
        """Return first key without removing - O(1)."""
        ...

    def peek(self) -> V:
        """Return last value without removing - O(1)."""
        ...

    def peekitem(self) -> tuple[K, V]:
        """Return last (key, value) without removing - O(1)."""
        ...

    def popleft(self) -> V:
        """Remove and return first value - O(1)."""
        ...

    def popleftitem(self) -> tuple[K, V]:
        """Remove and return first (key, value) - O(1)."""
        ...

    @overload
    def pop(self) -> V: ...
    @overload
    def pop(self, key: K) -> V: ...
    @overload
    def pop(self, key: K, default: V) -> V: ...

    def popitem(self) -> tuple[K, V]:
        """Remove and return last (key, value) - O(1)."""
        ...

    def appendleft(self, key: K, value: V) -> None:
        """Insert (key, value) at front - O(1)."""
        ...

    def move_to_end(self, key: K, last: bool = True) -> None:
        """Move key to front (last=False) or back (last=True) - O(1)."""
        ...

    # Dict-like operations
    @overload
    def get(self, key: K) -> V | None: ...
    @overload
    def get(self, key: K, default: V) -> V: ...

    def keys(self) -> KeysView[K]:
        """D.keys() -> view of keys in order."""
        ...

    def values(self) -> ValuesView[V]:
        """D.values() -> view of values in order."""
        ...

    def items(self) -> ItemsView[K, V]:
        """D.items() -> view of (key, value) in order."""
        ...

    def clear(self) -> None:
        """D.clear() -- remove all items."""
        ...

    def copy(self) -> DequeDict[K, V]:
        """D.copy() -> a shallow copy."""
        ...

    def update(self, other: Mapping[K, V] | Iterable[tuple[K, V]] | None = None, **kwargs: V) -> None:
        """D.update([E, ]**F)."""
        ...

    @overload
    def setdefault(self, key: K) -> V | None: ...
    @overload
    def setdefault(self, key: K, default: V) -> V: ...


class DefaultDequeDict(DequeDict[K, V]):
    """DequeDict with default_factory for missing keys."""

    default_factory: Callable[[], V] | None

    def __init__(
        self,
        default_factory: Callable[[], V] | None = None,
        items: Mapping[K, V] | Iterable[tuple[K, V]] | None = None,
    ) -> None: ...

    def __missing__(self, key: K) -> V: ...
    def copy(self) -> DefaultDequeDict[K, V]: ...


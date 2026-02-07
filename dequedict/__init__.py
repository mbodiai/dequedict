"""DequeDict - Ordered dictionary with O(1) deque operations at both ends."""
from __future__ import annotations

import os
import types
from collections.abc import Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Generic, ItemsView, Iterator, KeysView, TypeVar, ValuesView, overload

from typing_extensions import TypeIs

__all__ = ["DequeDict", "DefaultDequeDict"]

K = TypeVar("K")
V = TypeVar("V")


def _is_iterable_of_pairs(items: object) -> TypeIs[Iterable[tuple[K, V]]]:
    return not isinstance(items, Mapping) and getattr(items, "__iter__", None) is not None


class DequeDict(Mapping[K, V]):
    """Ordered dictionary with O(1) deque operations at both ends.

    Combines dict interface with deque-like operations:
    - O(1) key lookup, contains, delete
    - O(1) popleft/popleftitem/peekleft/peekleftitem
    - O(1) pop/popitem/peek/peekitem
    - O(1) appendleft
    - O(1) move_to_end
    - O(1) at(index) — amortized via incremental cache
    """

    __slots__ = ("_dict", "_head", "_tail", "_cache", "_cache_offset")
    __hash__ = None  # type: ignore[assignment]

    def __class_getitem__(cls, params: object) -> types.GenericAlias:
        return types.GenericAlias(cls, params)

    class _Node:
        __slots__ = ("key", "value", "prev", "next", "cache_idx")

        def __init__(self, key: K, value: V) -> None:
            self.key = key
            self.value = value
            self.prev: DequeDict._Node | None = None
            self.next: DequeDict._Node | None = None
            self.cache_idx: int = -1

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] | None = None) -> None:
        self._dict: dict[K, DequeDict._Node] = {}
        self._head: DequeDict._Node | None = None
        self._tail: DequeDict._Node | None = None
        self._cache: list[V] | None = None
        self._cache_offset: int = 0
        if items is not None:
            if _is_iterable_of_pairs(items):
                for k, v in items:
                    self[k] = v
            else:
                for k, v in items.items():
                    self[k] = v

    def _invalidate_cache(self) -> None:
        self._cache = None
        self._cache_offset = 0

    def _build_cache(self) -> list[_Node]:
        cache: list[DequeDict._Node] = []
        node = self._head
        i = 0
        while node:
            node.cache_idx = i
            cache.append(node)
            node = node.next
            i += 1
        self._cache = cache
        self._cache_offset = 0
        return cache

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __getitem__(self, key: K) -> V:
        if key not in self._dict:
            raise KeyError(key)
        return self._dict[key].value

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._dict:
            node = self._dict[key]
            node.value = value
            return
        node = self._Node(key, value)
        self._dict[key] = node
        if self._cache is not None:
            node.cache_idx = len(self._cache)
            self._cache.append(node)
        if self._tail is None:
            self._head = self._tail = node
        else:
            node.prev = self._tail
            self._tail.next = node
            self._tail = node

    def __delitem__(self, key: K) -> None:
        if key not in self._dict:
            raise KeyError(key)
        node = self._dict.pop(key)
        self._unlink(node)
        self._invalidate_cache()

    def _unlink(self, node: _Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

    def __iter__(self) -> Iterator[K]:
        node = self._head
        while node:
            yield node.key
            node = node.next

    def __reversed__(self) -> Iterator[K]:
        node = self._tail
        while node:
            yield node.key
            node = node.prev

    def __repr__(self) -> str:
        if not self._dict:
            return "DequeDict()"
        items = [(n.key, n.value) for n in self._iter_nodes()]
        return f"DequeDict({items!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (dict, DequeDict)):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(not (k not in other or other[k] != v) for k, v in self.items())

    def _iter_nodes(self) -> Iterator[_Node]:
        node = self._head
        while node:
            yield node
            node = node.next

    def peekleft(self) -> V:
        """Return first value without removing."""
        if self._head is None:
            raise IndexError("peek from an empty DequeDict")
        return self._head.value

    def peekleftitem(self) -> tuple[K, V]:
        """Return first (key, value) without removing."""
        if self._head is None:
            raise IndexError("peek from an empty DequeDict")
        return (self._head.key, self._head.value)

    def peekleftkey(self) -> K:
        """Return first key without removing."""
        if self._head is None:
            raise IndexError("peek from an empty DequeDict")
        return self._head.key

    def peek(self) -> V:
        """Return last value without removing."""
        if self._tail is None:
            raise IndexError("peek from an empty DequeDict")
        return self._tail.value

    def peekitem(self) -> tuple[K, V]:
        """Return last (key, value) without removing."""
        if self._tail is None:
            raise IndexError("peek from an empty DequeDict")
        return (self._tail.key, self._tail.value)

    def popleft(self) -> V:
        """Remove and return first value."""
        if self._head is None:
            raise IndexError("pop from an empty DequeDict")
        node = self._head
        del self._dict[node.key]
        self._unlink(node)
        if self._cache is not None:
            self._cache_offset += 1
        return node.value

    def popleftitem(self) -> tuple[K, V]:
        """Remove and return first (key, value)."""
        if self._head is None:
            raise KeyError("popleftitem from an empty DequeDict")
        node = self._head
        del self._dict[node.key]
        self._unlink(node)
        if self._cache is not None:
            self._cache_offset += 1
        return (node.key, node.value)

    @overload
    def pop(self) -> V: ...
    @overload
    def pop(self, key: K) -> V: ...
    @overload
    def pop(self, key: K, default: V) -> V: ...

    def pop(self, key: K | None = None, default: V | None = None) -> V:  # type: ignore[misc]
        """Remove and return value by key, or from end if no key given."""
        if key is None:
            if self._tail is None:
                if default is not None:
                    return default
                raise IndexError("pop from an empty DequeDict")
            node = self._tail
            del self._dict[node.key]
            self._unlink(node)
            if self._cache is not None:
                self._cache.pop()
            return node.value
        if key not in self._dict:
            if default is not None:
                return default
            raise KeyError(key)
        node = self._dict.pop(key)
        self._unlink(node)
        self._invalidate_cache()
        return node.value

    def popitem(self) -> tuple[K, V]:
        """Remove and return last (key, value)."""
        if self._tail is None:
            raise KeyError("popitem from an empty DequeDict")
        node = self._tail
        del self._dict[node.key]
        self._unlink(node)
        if self._cache is not None:
            self._cache.pop()
        return (node.key, node.value)

    def appendleft(self, key: K, value: V) -> None:
        """Insert (key, value) at front."""
        if key in self._dict:
            raise KeyError("key already exists")
        node = self._Node(key, value)
        self._dict[key] = node
        self._invalidate_cache()
        if self._head is None:
            self._head = self._tail = node
        else:
            node.next = self._head
            self._head.prev = node
            self._head = node

    def move_to_end(self, key: K, last: bool = True) -> None:
        """Move existing key to front (last=False) or back (last=True)."""
        if key not in self._dict:
            raise KeyError(key)
        node = self._dict[key]
        if (last and node is self._tail) or (not last and node is self._head):
            return
        self._unlink(node)
        self._invalidate_cache()
        if last:
            node.prev = self._tail
            node.next = None
            if self._tail:
                self._tail.next = node
            else:
                self._head = node
            self._tail = node
        else:
            node.prev = None
            node.next = self._head
            if self._head:
                self._head.prev = node
            else:
                self._tail = node
            self._head = node

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return value for key, or default if key not present."""
        if key not in self._dict:
            return default
        return self._dict[key].value

    def keys(self) -> KeysView[K]:
        """Return view of keys in insertion order."""
        return _DequeDictKeysView(self)

    def values(self) -> ValuesView[V]:
        """Return view of values in insertion order."""
        return _DequeDictValuesView(self)

    def items(self) -> ItemsView[K, V]:
        """Return view of (key, value) pairs in insertion order."""
        return _DequeDictItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        self._dict.clear()
        self._head = None
        self._tail = None
        self._invalidate_cache()

    def copy(self) -> DequeDict[K, V]:
        """Return a shallow copy."""
        return DequeDict(self.items())

    def update(self, other: Mapping[K, V] | Iterable[tuple[K, V]] | None = None, **kwargs: V) -> None:
        """Update from dict, iterable of pairs, or keyword arguments."""
        if other is not None:
            if isinstance(other, Mapping):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v  # type: ignore[index]

    def setdefault(self, key: K, default: V | None = None) -> V | None:
        """Return value for key, setting default if not present."""
        if key in self._dict:
            return self._dict[key].value
        self[key] = default  # type: ignore[assignment]
        return default

    def at(self, index: int) -> V:
        """Return value at index position. Supports negative indexing. O(1) amortized."""
        cache = self._cache
        if cache is None:
            cache = self._build_cache()
        logical_size = len(cache) - self._cache_offset
        if index < 0:
            index += logical_size
        if index < 0 or index >= logical_size:
            raise IndexError("index out of range")
        return cache[self._cache_offset + index].value


class _DequeDictKeysView(KeysView[K]):
    __slots__ = ("_dd",)

    def __init__(self, dd: DequeDict[K, V]) -> None:
        self._dd = dd

    def __len__(self) -> int:
        return len(self._dd)

    def __iter__(self) -> Iterator[K]:
        return iter(self._dd)

    def __reversed__(self) -> Iterator[K]:
        return reversed(self._dd)

    def __contains__(self, key: object) -> bool:
        return key in self._dd


class _DequeDictValuesView(ValuesView[V]):
    __slots__ = ("_dd",)

    def __init__(self, dd: DequeDict[K, V]) -> None:
        self._dd = dd

    def __len__(self) -> int:
        return len(self._dd)

    def __iter__(self) -> Iterator[V]:
        for node in self._dd._iter_nodes():
            yield node.value

    def __reversed__(self) -> Iterator[V]:
        node = self._dd._tail
        while node:
            yield node.value
            node = node.prev

    def __contains__(self, value: object) -> bool:
        return any(node.value == value for node in self._dd._iter_nodes())


class _DequeDictItemsView(ItemsView[K, V]):
    __slots__ = ("_dd",)

    def __init__(self, dd: DequeDict[K, V]) -> None:
        self._dd = dd

    def __len__(self) -> int:
        return len(self._dd)

    def __iter__(self) -> Iterator[tuple[K, V]]:
        for node in self._dd._iter_nodes():
            yield (node.key, node.value)

    def __reversed__(self) -> Iterator[tuple[K, V]]:
        node = self._dd._tail
        while node:
            yield (node.key, node.value)
            node = node.prev

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        k, v = item
        if k not in self._dd:
            return False
        return self._dd[k] == v


class DefaultDequeDict(DequeDict[K, V]):
    """DequeDict with default_factory for missing keys, like collections.defaultdict."""

    __slots__ = ("default_factory",)

    def __init__(
        self,
        default_factory: Callable[[], V] | None = None,
        items: Mapping[K, V] | Iterable[tuple[K, V]] | None = None,
    ) -> None:
        self.default_factory = default_factory
        super().__init__(items)

    def __missing__(self, key: K) -> V:
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self[key] = value
        return value

    def __getitem__(self, key: K) -> V:
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)

    def __repr__(self) -> str:
        return f"DefaultDequeDict({self.default_factory}, {list(self.items())!r})"

    def copy(self) -> DefaultDequeDict[K, V]:
        return DefaultDequeDict(self.default_factory, self.items())


# Use C extension if available (disable with NOC=1 environment variable)
if not TYPE_CHECKING and not os.getenv("NOC"):
    with suppress(ImportError):
        from dequedict._dequedict import DequeDict

"""Tests for DequeDict following SETUP → EXPECTED → ACT → ASSERT pattern."""
from __future__ import annotations
import pytest
from dequedict import DequeDict, DefaultDequeDict


class TestDequeDictInit:
    """Tests for DequeDict initialization."""

    def test_init_empty_creates_empty_dequedict(self):
        # ACT
        dd = DequeDict()

        # ASSERT
        expected_len = 0
        assert len(dd) == expected_len

    def test_init_from_dict_preserves_items(self):
        # SETUP
        source = {"a": 1, "b": 2, "c": 3}

        # EXPECTED
        expected_len = 3
        expected_a = 1
        expected_b = 2

        # ACT
        dd = DequeDict(source)

        # ASSERT
        assert len(dd) == expected_len
        assert dd["a"] == expected_a
        assert dd["b"] == expected_b

    def test_init_from_pairs_preserves_order(self):
        # SETUP
        pairs = [("x", 10), ("y", 20), ("z", 30)]

        # EXPECTED
        expected_keys = ["x", "y", "z"]
        expected_first_key = "x"
        expected_last_key = "z"

        # ACT
        dd = DequeDict(pairs)

        # ASSERT
        assert list(dd.keys()) == expected_keys
        assert dd.peekleftkey() == expected_first_key
        assert dd.peekitem()[0] == expected_last_key


class TestDequeDictGetSet:
    """Tests for getting and setting items."""

    def test_setitem_adds_new_key_at_end(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_last_key = "b"
        expected_value = 2

        # ACT
        dd["b"] = 2

        # ASSERT
        assert dd.peekitem()[0] == expected_last_key
        assert dd["b"] == expected_value

    def test_setitem_updates_existing_key_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_value = 99
        expected_order = ["a", "b"]

        # ACT
        dd["a"] = 99

        # ASSERT
        assert dd["a"] == expected_value
        assert list(dd.keys()) == expected_order

    def test_getitem_raises_keyerror_for_missing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # ACT & ASSERT
        with pytest.raises(KeyError):
            _ = dd["missing"]

    def test_delitem_removes_key(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_len = 2
        expected_keys = ["a", "c"]

        # ACT
        del dd["b"]

        # ASSERT
        assert len(dd) == expected_len
        assert list(dd.keys()) == expected_keys
        assert "b" not in dd


class TestDequeDictContains:
    """Tests for membership checking."""

    def test_contains_returns_true_for_existing_key(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_result = True

        # ACT
        result = "a" in dd

        # ASSERT
        assert result == expected_result

    def test_contains_returns_false_for_missing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_result = False

        # ACT
        result = "missing" in dd

        # ASSERT
        assert result == expected_result


class TestDequeDictPeek:
    """Tests for peek operations."""

    def test_peekleft_returns_first_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_value = 1

        # ACT
        result = dd.peekleft()

        # ASSERT
        assert result == expected_value

    def test_peekleftitem_returns_first_pair(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_pair = ("a", 1)

        # ACT
        result = dd.peekleftitem()

        # ASSERT
        assert result == expected_pair

    def test_peekleftkey_returns_first_key(self):
        # SETUP
        dd = DequeDict([("first", 100), ("second", 200)])

        # EXPECTED
        expected_key = "first"

        # ACT
        result = dd.peekleftkey()

        # ASSERT
        assert result == expected_key

    def test_peek_returns_last_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_value = 3

        # ACT
        result = dd.peek()

        # ASSERT
        assert result == expected_value

    def test_peekitem_returns_last_pair(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_pair = ("b", 2)

        # ACT
        result = dd.peekitem()

        # ASSERT
        assert result == expected_pair

    def test_peekleft_raises_on_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        with pytest.raises(IndexError, match="empty"):
            dd.peekleft()

    def test_peek_raises_on_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        with pytest.raises(IndexError, match="empty"):
            dd.peek()


class TestDequeDictPop:
    """Tests for pop operations."""

    def test_popleft_removes_and_returns_first_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_value = 1
        expected_remaining_keys = ["b", "c"]

        # ACT
        result = dd.popleft()

        # ASSERT
        assert result == expected_value
        assert list(dd.keys()) == expected_remaining_keys

    def test_popleftitem_removes_and_returns_first_pair(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_pair = ("a", 1)
        expected_len = 1

        # ACT
        result = dd.popleftitem()

        # ASSERT
        assert result == expected_pair
        assert len(dd) == expected_len

    def test_pop_without_key_removes_last(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_value = 3
        expected_remaining_keys = ["a", "b"]

        # ACT
        result = dd.pop()

        # ASSERT
        assert result == expected_value
        assert list(dd.keys()) == expected_remaining_keys

    def test_pop_with_key_removes_specific_item(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_value = 2
        expected_remaining_keys = ["a", "c"]

        # ACT
        result = dd.pop("b")

        # ASSERT
        assert result == expected_value
        assert list(dd.keys()) == expected_remaining_keys

    def test_pop_with_default_returns_default_for_missing(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_value = "default"

        # ACT
        result = dd.pop("missing", "default")

        # ASSERT
        assert result == expected_value

    def test_popitem_removes_and_returns_last_pair(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_pair = ("b", 2)
        expected_len = 1

        # ACT
        result = dd.popitem()

        # ASSERT
        assert result == expected_pair
        assert len(dd) == expected_len

    def test_popleft_raises_on_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        with pytest.raises(IndexError):
            dd.popleft()

    def test_popitem_raises_on_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        with pytest.raises(KeyError):
            dd.popitem()


class TestDequeDictAppendLeft:
    """Tests for appendleft operation."""

    def test_appendleft_inserts_at_front(self):
        # SETUP
        dd = DequeDict([("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["a", "b", "c"]
        expected_first_key = "a"

        # ACT
        dd.appendleft("a", 1)

        # ASSERT
        assert list(dd.keys()) == expected_keys
        assert dd.peekleftkey() == expected_first_key

    def test_appendleft_raises_for_existing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # ACT & ASSERT
        with pytest.raises(KeyError, match="already exists"):
            dd.appendleft("a", 99)


class TestDequeDictMoveToEnd:
    """Tests for move_to_end operation."""

    def test_move_to_end_moves_to_back(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["b", "c", "a"]

        # ACT
        dd.move_to_end("a")

        # ASSERT
        assert list(dd.keys()) == expected_keys

    def test_move_to_end_with_last_false_moves_to_front(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["c", "a", "b"]

        # ACT
        dd.move_to_end("c", last=False)

        # ASSERT
        assert list(dd.keys()) == expected_keys

    def test_move_to_end_raises_for_missing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # ACT & ASSERT
        with pytest.raises(KeyError):
            dd.move_to_end("missing")


class TestDequeDictGet:
    """Tests for get method."""

    def test_get_returns_value_for_existing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_value = 1

        # ACT
        result = dd.get("a")

        # ASSERT
        assert result == expected_value

    def test_get_returns_none_for_missing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_value = None

        # ACT
        result = dd.get("missing")

        # ASSERT
        assert result == expected_value

    def test_get_returns_default_for_missing_key(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_value = "default"

        # ACT
        result = dd.get("missing", "default")

        # ASSERT
        assert result == expected_value


class TestDequeDictViews:
    """Tests for keys, values, and items views."""

    def test_keys_returns_keys_in_order(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["a", "b", "c"]

        # ACT
        result = list(dd.keys())

        # ASSERT
        assert result == expected_keys

    def test_values_returns_values_in_order(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_values = [1, 2, 3]

        # ACT
        result = list(dd.values())

        # ASSERT
        assert result == expected_values

    def test_items_returns_pairs_in_order(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_items = [("a", 1), ("b", 2)]

        # ACT
        result = list(dd.items())

        # ASSERT
        assert result == expected_items

    def test_keys_view_contains(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # ACT & ASSERT
        assert "a" in dd.keys()
        assert "c" not in dd.keys()

    def test_items_view_contains(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # ACT & ASSERT
        assert ("a", 1) in dd.items()
        assert ("a", 99) not in dd.items()

    def test_reversed_keys_view(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT
        result = list(reversed(dd.keys()))

        # ASSERT
        assert result == ["c", "b", "a"]

    def test_reversed_values_view(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT
        result = list(reversed(dd.values()))

        # ASSERT
        assert result == [3, 2, 1]

    def test_reversed_items_view(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT
        result = list(reversed(dd.items()))

        # ASSERT
        assert result == [("c", 3), ("b", 2), ("a", 1)]

    def test_reversed_views_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        assert list(reversed(dd.keys())) == []
        assert list(reversed(dd.values())) == []
        assert list(reversed(dd.items())) == []

    def test_reversed_views_next(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT & ASSERT — O(1) single element access
        assert next(reversed(dd.keys())) == "c"
        assert next(reversed(dd.values())) == 3
        assert next(reversed(dd.items())) == ("c", 3)


class TestDequeDictIteration:
    """Tests for iteration."""

    def test_iter_yields_keys_in_order(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["a", "b", "c"]

        # ACT
        result = list(dd)

        # ASSERT
        assert result == expected_keys

    def test_reversed_yields_keys_in_reverse_order(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # EXPECTED
        expected_keys = ["c", "b", "a"]

        # ACT
        result = list(reversed(dd))

        # ASSERT
        assert result == expected_keys


class TestDequeDictClearCopy:
    """Tests for clear and copy operations."""

    def test_clear_removes_all_items(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_len = 0

        # ACT
        dd.clear()

        # ASSERT
        assert len(dd) == expected_len

    def test_copy_creates_shallow_copy(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # EXPECTED
        expected_items = [("a", 1), ("b", 2)]

        # ACT
        copy = dd.copy()
        dd["c"] = 3

        # ASSERT
        assert list(copy.items()) == expected_items
        assert "c" not in copy


class TestDequeDictUpdate:
    """Tests for update method."""

    def test_update_from_dict(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_len = 2
        expected_b = 2

        # ACT
        dd.update({"b": 2})

        # ASSERT
        assert len(dd) == expected_len
        assert dd["b"] == expected_b

    def test_update_from_pairs(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_keys = ["a", "b", "c"]

        # ACT
        dd.update([("b", 2), ("c", 3)])

        # ASSERT
        assert list(dd.keys()) == expected_keys

    def test_update_with_kwargs(self):
        # SETUP
        dd = DequeDict()

        # EXPECTED
        expected_x = 10

        # ACT
        dd.update(x=10, y=20)

        # ASSERT
        assert dd["x"] == expected_x


class TestDequeDictSetDefault:
    """Tests for setdefault method."""

    def test_setdefault_returns_existing_value(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # EXPECTED
        expected_value = 1

        # ACT
        result = dd.setdefault("a", 99)

        # ASSERT
        assert result == expected_value
        assert dd["a"] == expected_value

    def test_setdefault_sets_and_returns_default(self):
        # SETUP
        dd = DequeDict()

        # EXPECTED
        expected_value = "default"

        # ACT
        result = dd.setdefault("a", "default")

        # ASSERT
        assert result == expected_value
        assert dd["a"] == expected_value


class TestDequeDictEquality:
    """Tests for equality comparison."""

    def test_equal_to_dict_with_same_contents(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])
        d = {"a": 1, "b": 2}

        # ACT & ASSERT
        assert dd == d

    def test_not_equal_to_dict_with_different_contents(self):
        # SETUP
        dd = DequeDict([("a", 1)])
        d = {"a": 2}

        # ACT & ASSERT
        assert dd != d

    def test_equal_to_another_dequedict(self):
        # SETUP
        dd1 = DequeDict([("a", 1), ("b", 2)])
        dd2 = DequeDict([("a", 1), ("b", 2)])

        # ACT & ASSERT
        assert dd1 == dd2


class TestDequeDictRepr:
    """Tests for repr."""

    def test_repr_empty(self):
        # SETUP
        dd = DequeDict()

        # EXPECTED
        expected_repr = "DequeDict()"

        # ACT
        result = repr(dd)

        # ASSERT
        assert result == expected_repr

    def test_repr_with_items(self):
        # SETUP
        dd = DequeDict([("a", 1)])

        # ACT
        result = repr(dd)

        # ASSERT
        assert "DequeDict" in result
        assert "a" in result


class TestDequeDictClassGetItem:
    """Tests for __class_getitem__ (generic subscript support)."""

    def test_class_getitem_returns_generic_alias(self):
        # ACT
        result = DequeDict[str, int]

        # ASSERT
        assert result is not None
        assert hasattr(result, '__origin__')
        assert result.__origin__ is DequeDict

    def test_class_getitem_with_single_param(self):
        # ACT
        result = DequeDict[str]

        # ASSERT
        assert result.__origin__ is DequeDict


class TestDequeDictAt:
    """Tests for at() positional access."""

    def test_at_returns_first_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT & ASSERT
        assert dd.at(0) == 1

    def test_at_returns_middle_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT & ASSERT
        assert dd.at(1) == 2

    def test_at_returns_last_value(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT & ASSERT
        assert dd.at(2) == 3

    def test_at_negative_index(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2), ("c", 3)])

        # ACT & ASSERT
        assert dd.at(-1) == 3
        assert dd.at(-2) == 2
        assert dd.at(-3) == 1

    def test_at_raises_on_empty(self):
        # SETUP
        dd = DequeDict()

        # ACT & ASSERT
        with pytest.raises(IndexError, match="index out of range"):
            dd.at(0)

    def test_at_raises_on_out_of_bounds(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # ACT & ASSERT
        with pytest.raises(IndexError, match="index out of range"):
            dd.at(2)

    def test_at_raises_on_negative_out_of_bounds(self):
        # SETUP
        dd = DequeDict([("a", 1), ("b", 2)])

        # ACT & ASSERT
        with pytest.raises(IndexError, match="index out of range"):
            dd.at(-3)

    def test_at_single_element(self):
        # SETUP
        dd = DequeDict([("only", 42)])

        # ACT & ASSERT
        assert dd.at(0) == 42
        assert dd.at(-1) == 42


class TestDefaultDequeDict:
    """Tests for DefaultDequeDict with default_factory."""

    def test_default_factory_list_creates_on_access(self):
        # SETUP
        dd = DefaultDequeDict(list)

        # EXPECTED
        expected_value = [1, 2]

        # ACT
        dd["a"].append(1)
        dd["a"].append(2)

        # ASSERT
        assert dd["a"] == expected_value

    def test_default_factory_int_starts_at_zero(self):
        # SETUP
        dd = DefaultDequeDict(int)

        # EXPECTED
        expected_value = 5

        # ACT
        dd["x"] += 5

        # ASSERT
        assert dd["x"] == expected_value

    def test_no_factory_raises_keyerror(self):
        # SETUP
        dd = DefaultDequeDict()

        # ACT & ASSERT
        with pytest.raises(KeyError):
            _ = dd["missing"]

    def test_deque_ops_still_work(self):
        # SETUP
        dd = DefaultDequeDict(int)
        dd["a"] = 1
        dd["b"] = 2

        # EXPECTED
        expected_first = 1

        # ACT
        result = dd.popleft()

        # ASSERT
        assert result == expected_first
        assert "a" not in dd

    def test_copy_preserves_factory(self):
        # SETUP
        dd = DefaultDequeDict(list)
        dd["a"].append(1)

        # ACT
        copy = dd.copy()
        copy["b"].append(2)

        # ASSERT
        assert copy["b"] == [2]
        assert copy.default_factory is list


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])


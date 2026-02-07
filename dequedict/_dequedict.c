/*
 * dequedict.c - Ordered dictionary with O(1) deque operations at both ends
 *
 * DequeDict combines the dict interface with deque-like operations:
 * - O(1) popleft/popleftitem/peekleft/peekleftitem
 * - O(1) pop/popitem/peek/peekitem (right side)
 * - O(1) appendleft (insert at front)
 * - O(1) lookup by key
 * - O(1) at(index) via incremental cache (C array of entry pointers)
 * - Maintains insertion order
 *
 * Cache stores entry pointers (not values). pop(key)/del maintains cache
 * in-place via memmove + index fixup, so at() stays O(1) always.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stdint.h>
#include <string.h>

/* Entry in the deque-dict */
typedef struct DequeDictEntry {
    PyObject *key;
    PyObject *value;
    struct DequeDictEntry *prev;
    struct DequeDictEntry *next;
    Py_ssize_t cache_idx;       /* Position in index_cache */
} DequeDictEntry;

/* Freelist for entry reuse - avoid malloc/free overhead */
#define FREELIST_MAX 128
static DequeDictEntry *freelist = NULL;
static int freelist_size = 0;

static inline DequeDictEntry *entry_alloc(void) {
    if (freelist) {
        DequeDictEntry *entry = freelist;
        freelist = entry->next;
        freelist_size--;
        return entry;
    }
    return PyMem_Malloc(sizeof(DequeDictEntry));
}

static inline void entry_free(DequeDictEntry *entry) {
    if (freelist_size < FREELIST_MAX) {
        entry->next = freelist;
        freelist = entry;
        freelist_size++;
    } else {
        PyMem_Free(entry);
    }
}

typedef struct {
    PyObject_HEAD
    PyObject *dict;                 /* Internal dict for O(1) key lookup -> entry */
    DequeDictEntry *head;           /* First entry (for popleft) */
    DequeDictEntry *tail;           /* Last entry (for pop) */
    Py_ssize_t size;
    DequeDictEntry **index_cache;   /* C array of entry pointers, or NULL */
    Py_ssize_t cache_size;          /* Number of entries in cache */
    Py_ssize_t cache_capacity;      /* Allocated capacity of cache */
    Py_ssize_t cache_offset;        /* Left offset into index_cache */
} DequeDictObject;

static PyTypeObject DequeDict_Type;

/* ========================================================================
 * Cache helpers
 * ======================================================================== */

static inline void
DequeDict_invalidate_cache(DequeDictObject *self)
{
    if (self->index_cache) {
        PyMem_Free(self->index_cache);
        self->index_cache = NULL;
    }
    self->cache_size = 0;
    self->cache_capacity = 0;
    self->cache_offset = 0;
}

static int
DequeDict_rebuild_cache(DequeDictObject *self)
{
    DequeDict_invalidate_cache(self);
    if (self->size == 0)
        return 0;

    self->cache_capacity = self->size;
    self->index_cache = PyMem_Malloc(sizeof(DequeDictEntry *) * self->cache_capacity);
    if (!self->index_cache) {
        PyErr_NoMemory();
        return -1;
    }
    self->cache_size = self->size;
    self->cache_offset = 0;

    DequeDictEntry *entry = self->head;
    Py_ssize_t i = 0;
    while (entry) {
        entry->cache_idx = i;
        self->index_cache[i] = entry;
        entry = entry->next;
        i++;
    }
    return 0;
}

/* Ensure cache has room for one more entry. Returns 0 on success, -1 on failure. */
static int
DequeDict_cache_ensure_capacity(DequeDictObject *self)
{
    if (self->cache_size < self->cache_capacity)
        return 0;

    Py_ssize_t new_cap = self->cache_capacity * 2;
    if (new_cap < 8) new_cap = 8;

    DequeDictEntry **new_cache = PyMem_Realloc(self->index_cache,
        sizeof(DequeDictEntry *) * new_cap);
    if (!new_cache) {
        DequeDict_invalidate_cache(self);
        return -1;
    }
    self->index_cache = new_cache;
    self->cache_capacity = new_cap;
    return 0;
}

/* ======================================================================== */

static int
DequeDict_traverse(DequeDictObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->dict);
    /* Visit all entries in the linked list */
    DequeDictEntry *entry = self->head;
    while (entry) {
        Py_VISIT(entry->key);
        Py_VISIT(entry->value);
        entry = entry->next;
    }
    return 0;
}

static int
DequeDict_clear(DequeDictObject *self)
{
    /* Free all entries */
    DequeDictEntry *entry = self->head;
    while (entry) {
        DequeDictEntry *next = entry->next;
        Py_CLEAR(entry->key);
        Py_CLEAR(entry->value);
        entry_free(entry);
        entry = next;
    }
    self->head = NULL;
    self->tail = NULL;
    self->size = 0;
    Py_CLEAR(self->dict);
    DequeDict_invalidate_cache(self);
    return 0;
}

static void
DequeDict_dealloc(DequeDictObject *self)
{
    PyObject_GC_UnTrack(self);
    DequeDict_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
DequeDict_init(DequeDictObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"items", NULL};
    PyObject *items = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &items))
        return -1;

    /* Untrack before modifying (safe for re-init) */
    PyObject_GC_UnTrack(self);

    /* Clear existing state */
    DequeDictEntry *entry = self->head;
    while (entry) {
        DequeDictEntry *next = entry->next;
        Py_DECREF(entry->key);
        Py_DECREF(entry->value);
        entry_free(entry);
        entry = next;
    }
    Py_XDECREF(self->dict);
    DequeDict_invalidate_cache(self);

    self->dict = PyDict_New();
    if (!self->dict) return -1;
    self->head = NULL;
    self->tail = NULL;
    self->size = 0;

    /* Initialize from items if provided */
    if (items) {
        if (PyDict_Check(items)) {
            /* Initialize from dict */
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(items, &pos, &key, &value)) {
                /* Create entry */
                DequeDictEntry *new_entry = entry_alloc();
                if (!new_entry) {
                    PyErr_NoMemory();
                    return -1;
                }

                Py_INCREF(key);
                Py_INCREF(value);
                new_entry->key = key;
                new_entry->value = value;
                new_entry->prev = self->tail;
                new_entry->next = NULL;
                new_entry->cache_idx = -1;

                if (self->tail) {
                    self->tail->next = new_entry;
                } else {
                    self->head = new_entry;
                }
                self->tail = new_entry;
                self->size++;

                /* Store capsule in dict for O(1) lookup */
                PyObject *capsule = PyLong_FromVoidPtr(new_entry);
                if (!capsule) {
                    entry_free(new_entry);
                    return -1;
                }
                if (PyDict_SetItem(self->dict, key, capsule) < 0) {
                    Py_DECREF(capsule);
                    entry_free(new_entry);
                    return -1;
                }
                Py_DECREF(capsule);
            }
        } else {
            /* Initialize from iterable of (key, value) pairs */
            PyObject *iter = PyObject_GetIter(items);
            if (!iter) return -1;

            PyObject *pair;
            while ((pair = PyIter_Next(iter)) != NULL) {
                if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    PyErr_SetString(PyExc_ValueError,
                        "DequeDict requires sequence of (key, value) pairs");
                    return -1;
                }

                PyObject *key = PyTuple_GET_ITEM(pair, 0);
                PyObject *value = PyTuple_GET_ITEM(pair, 1);

                DequeDictEntry *new_entry = entry_alloc();
                if (!new_entry) {
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    PyErr_NoMemory();
                    return -1;
                }

                Py_INCREF(key);
                Py_INCREF(value);
                new_entry->key = key;
                new_entry->value = value;
                new_entry->prev = self->tail;
                new_entry->next = NULL;
                new_entry->cache_idx = -1;

                if (self->tail) {
                    self->tail->next = new_entry;
                } else {
                    self->head = new_entry;
                }
                self->tail = new_entry;
                self->size++;

                PyObject *capsule = PyLong_FromVoidPtr(new_entry);
                if (!capsule) {
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    entry_free(new_entry);
                    return -1;
                }
                if (PyDict_SetItem(self->dict, key, capsule) < 0) {
                    Py_DECREF(capsule);
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    entry_free(new_entry);
                    return -1;
                }
                Py_DECREF(capsule);
                Py_DECREF(pair);
            }
            Py_DECREF(iter);
            if (PyErr_Occurred()) return -1;
        }
    }

    PyObject_GC_Track(self);
    return 0;
}

static Py_ssize_t
DequeDict_len(DequeDictObject *self)
{
    return self->size;
}

/* __getitem__ - O(1) lookup */
static PyObject *
DequeDict_getitem(DequeDictObject *self, PyObject *key)
{
    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (!capsule) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
    if (!entry) return NULL;

    Py_INCREF(entry->value);
    return entry->value;
}

/* __setitem__ - O(1) if key exists, append if new */
static int
DequeDict_setitem(DequeDictObject *self, PyObject *key, PyObject *value)
{
    if (value == NULL) {
        /* __delitem__ — remove from cache in-place */
        PyObject *capsule = PyDict_GetItem(self->dict, key);
        if (!capsule) {
            PyErr_SetObject(PyExc_KeyError, key);
            return -1;
        }

        DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
        if (!entry) return -1;

        /* Unlink from list */
        if (entry->prev) {
            entry->prev->next = entry->next;
        } else {
            self->head = entry->next;
        }
        if (entry->next) {
            entry->next->prev = entry->prev;
        } else {
            self->tail = entry->prev;
        }

        PyDict_DelItem(self->dict, key);
        Py_DECREF(entry->key);
        Py_DECREF(entry->value);
        entry_free(entry);
        self->size--;
        DequeDict_invalidate_cache(self);
        return 0;
    }

    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (capsule) {
        /* Update existing entry — cache stores entry pointers, no update needed */
        DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
        if (!entry) return -1;

        PyObject *old_value = entry->value;
        Py_INCREF(value);
        entry->value = value;
        Py_DECREF(old_value);
        return 0;
    }

    /* New key - append to end */
    DequeDictEntry *new_entry = entry_alloc();
    if (!new_entry) {
        PyErr_NoMemory();
        return -1;
    }

    Py_INCREF(key);
    Py_INCREF(value);
    new_entry->key = key;
    new_entry->value = value;
    new_entry->prev = self->tail;
    new_entry->next = NULL;
    new_entry->cache_idx = -1;

    if (self->tail) {
        self->tail->next = new_entry;
    } else {
        self->head = new_entry;
    }
    self->tail = new_entry;
    self->size++;

    /* Append to cache if it exists */
    if (self->index_cache) {
        if (DequeDict_cache_ensure_capacity(self) == 0 && self->index_cache) {
            new_entry->cache_idx = self->cache_size;
            self->index_cache[self->cache_size] = new_entry;
            self->cache_size++;
        }
    }

    capsule = PyLong_FromVoidPtr(new_entry);
    if (!capsule) {
        entry_free(new_entry);
        return -1;
    }
    if (PyDict_SetItem(self->dict, key, capsule) < 0) {
        Py_DECREF(capsule);
        entry_free(new_entry);
        return -1;
    }
    Py_DECREF(capsule);
    return 0;
}

/* __contains__ - O(1) */
static int
DequeDict_contains(DequeDictObject *self, PyObject *key)
{
    return PyDict_Contains(self->dict, key);
}

/* peekleft() - O(1) return first value without removing */
static PyObject *
DequeDict_peekleft(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->head) {
        PyErr_SetString(PyExc_IndexError, "peek from an empty DequeDict");
        return NULL;
    }
    Py_INCREF(self->head->value);
    return self->head->value;
}

/* peekleftitem() - O(1) return first (key, value) without removing */
static PyObject *
DequeDict_peekleftitem(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->head) {
        PyErr_SetString(PyExc_IndexError, "peek from an empty DequeDict");
        return NULL;
    }
    return PyTuple_Pack(2, self->head->key, self->head->value);
}

/* peekleftkey() - O(1) return first key without removing */
static PyObject *
DequeDict_peekleftkey(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->head) {
        PyErr_SetString(PyExc_IndexError, "peek from an empty DequeDict");
        return NULL;
    }
    Py_INCREF(self->head->key);
    return self->head->key;
}

/* peek() / peekright() - O(1) return last value without removing */
static PyObject *
DequeDict_peek(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->tail) {
        PyErr_SetString(PyExc_IndexError, "peek from an empty DequeDict");
        return NULL;
    }
    Py_INCREF(self->tail->value);
    return self->tail->value;
}

/* peekitem() / peekrightitem() - O(1) return last (key, value) without removing */
static PyObject *
DequeDict_peekitem(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->tail) {
        PyErr_SetString(PyExc_IndexError, "peek from an empty DequeDict");
        return NULL;
    }
    return PyTuple_Pack(2, self->tail->key, self->tail->value);
}

/* popleft() - O(1) remove and return first value */
static PyObject *
DequeDict_popleft(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->head) {
        PyErr_SetString(PyExc_IndexError, "pop from an empty DequeDict");
        return NULL;
    }

    DequeDictEntry *entry = self->head;
    PyObject *value = entry->value;
    Py_INCREF(value);

    /* Unlink head */
    self->head = entry->next;
    if (self->head) {
        self->head->prev = NULL;
    } else {
        self->tail = NULL;
    }

    PyDict_DelItem(self->dict, entry->key);
    Py_DECREF(entry->key);
    Py_DECREF(entry->value);
    entry_free(entry);
    self->size--;

    /* Bump cache offset instead of invalidating */
    if (self->index_cache) {
        self->cache_offset++;
    }

    return value;
}

/* popleftitem() - O(1) remove and return first (key, value) */
static PyObject *
DequeDict_popleftitem(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->head) {
        PyErr_SetString(PyExc_KeyError, "popleftitem from an empty DequeDict");
        return NULL;
    }

    DequeDictEntry *entry = self->head;
    PyObject *key = entry->key;
    PyObject *value = entry->value;
    Py_INCREF(key);
    Py_INCREF(value);

    /* Unlink head */
    self->head = entry->next;
    if (self->head) {
        self->head->prev = NULL;
    } else {
        self->tail = NULL;
    }

    PyDict_DelItem(self->dict, key);
    Py_DECREF(entry->key);
    Py_DECREF(entry->value);
    entry_free(entry);
    self->size--;

    /* Bump cache offset instead of invalidating */
    if (self->index_cache) {
        self->cache_offset++;
    }

    PyObject *result = PyTuple_Pack(2, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    return result;
}

/* pop(key=None, default=UNSET) - O(1) remove by key or from end */
static PyObject *
DequeDict_pop(DequeDictObject *self, PyObject *args)
{
    PyObject *key = NULL;
    PyObject *default_val = NULL;

    if (!PyArg_ParseTuple(args, "|OO", &key, &default_val))
        return NULL;

    if (key == NULL) {
        /* Pop from right (like deque) */
        if (!self->tail) {
            if (default_val) {
                Py_INCREF(default_val);
                return default_val;
            }
            PyErr_SetString(PyExc_IndexError, "pop from an empty DequeDict");
            return NULL;
        }

        DequeDictEntry *entry = self->tail;
        PyObject *value = entry->value;
        Py_INCREF(value);

        /* Unlink tail */
        self->tail = entry->prev;
        if (self->tail) {
            self->tail->next = NULL;
        } else {
            self->head = NULL;
        }

        PyDict_DelItem(self->dict, entry->key);
        Py_DECREF(entry->key);
        Py_DECREF(entry->value);
        entry_free(entry);
        self->size--;

        /* Pop from cache right side */
        if (self->index_cache && self->cache_size > 0) {
            self->cache_size--;
        }

        return value;
    }

    /* Pop by key — remove from cache in-place */
    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (!capsule) {
        if (default_val) {
            Py_INCREF(default_val);
            return default_val;
        }
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
    if (!entry) return NULL;

    PyObject *value = entry->value;
    Py_INCREF(value);

    /* Unlink from list */
    if (entry->prev) {
        entry->prev->next = entry->next;
    } else {
        self->head = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    } else {
        self->tail = entry->prev;
    }

    PyDict_DelItem(self->dict, key);
    Py_DECREF(entry->key);
    Py_DECREF(entry->value);
    entry_free(entry);
    self->size--;
    DequeDict_invalidate_cache(self);

    return value;
}

/* popitem() - O(1) remove and return last (key, value) */
static PyObject *
DequeDict_popitem(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    if (!self->tail) {
        PyErr_SetString(PyExc_KeyError, "popitem from an empty DequeDict");
        return NULL;
    }

    DequeDictEntry *entry = self->tail;
    PyObject *key = entry->key;
    PyObject *value = entry->value;
    Py_INCREF(key);
    Py_INCREF(value);

    /* Unlink tail */
    self->tail = entry->prev;
    if (self->tail) {
        self->tail->next = NULL;
    } else {
        self->head = NULL;
    }

    PyDict_DelItem(self->dict, key);
    Py_DECREF(entry->key);
    Py_DECREF(entry->value);
    entry_free(entry);
    self->size--;

    /* Pop from cache right side */
    if (self->index_cache && self->cache_size > 0) {
        self->cache_size--;
    }

    PyObject *result = PyTuple_Pack(2, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    return result;
}

/* appendleft(key, value) - O(1) insert at front — invalidates cache */
static PyObject *
DequeDict_appendleft(DequeDictObject *self, PyObject *args)
{
    PyObject *key, *value;

    if (!PyArg_ParseTuple(args, "OO", &key, &value))
        return NULL;

    /* Check if key exists */
    if (PyDict_Contains(self->dict, key)) {
        PyErr_SetString(PyExc_KeyError, "key already exists");
        return NULL;
    }

    DequeDictEntry *new_entry = entry_alloc();
    if (!new_entry) {
        return PyErr_NoMemory();
    }

    Py_INCREF(key);
    Py_INCREF(value);
    new_entry->key = key;
    new_entry->value = value;
    new_entry->prev = NULL;
    new_entry->next = self->head;
    new_entry->cache_idx = -1;

    if (self->head) {
        self->head->prev = new_entry;
    } else {
        self->tail = new_entry;
    }
    self->head = new_entry;
    self->size++;
    DequeDict_invalidate_cache(self);

    PyObject *capsule = PyLong_FromVoidPtr(new_entry);
    if (!capsule) {
        entry_free(new_entry);
        return NULL;
    }
    if (PyDict_SetItem(self->dict, key, capsule) < 0) {
        Py_DECREF(capsule);
        entry_free(new_entry);
        return NULL;
    }
    Py_DECREF(capsule);

    Py_RETURN_NONE;
}

/* get(key, default=None) */
static PyObject *
DequeDict_get(DequeDictObject *self, PyObject *args)
{
    PyObject *key;
    PyObject *default_val = Py_None;

    if (!PyArg_ParseTuple(args, "O|O", &key, &default_val))
        return NULL;

    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (!capsule) {
        Py_INCREF(default_val);
        return default_val;
    }

    DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
    if (!entry) {
        Py_INCREF(default_val);
        return default_val;
    }

    Py_INCREF(entry->value);
    return entry->value;
}

/* ========================================================================
 * Keys/Values/Items Views - O(1) creation, lazy iteration
 * ======================================================================== */

typedef struct {
    PyObject_HEAD
    DequeDictObject *dd;
    int kind;  /* 0=keys, 1=values, 2=items */
} DequeDictViewObject;

static PyTypeObject DequeDictKeysView_Type;
static PyTypeObject DequeDictValuesView_Type;
static PyTypeObject DequeDictItemsView_Type;

static int
DequeDictView_traverse(DequeDictViewObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->dd);
    return 0;
}

static int
DequeDictView_clear(DequeDictViewObject *self)
{
    Py_CLEAR(self->dd);
    return 0;
}

static void
DequeDictView_dealloc(DequeDictViewObject *self)
{
    PyObject_GC_UnTrack(self);
    DequeDictView_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static Py_ssize_t
DequeDictView_len(DequeDictViewObject *self)
{
    return self->dd->size;
}

/* View iterator */
typedef struct {
    PyObject_HEAD
    DequeDictObject *dd;
    DequeDictEntry *current;
    int kind;
    int reverse;  /* 0=forward (->next), 1=reverse (->prev) */
} DequeDictViewIterObject;

static PyTypeObject DequeDictViewIter_Type;

static int
DequeDictViewIter_traverse(DequeDictViewIterObject *it, visitproc visit, void *arg)
{
    Py_VISIT(it->dd);
    return 0;
}

static int
DequeDictViewIter_clear(DequeDictViewIterObject *it)
{
    Py_CLEAR(it->dd);
    return 0;
}

static void
DequeDictViewIter_dealloc(DequeDictViewIterObject *it)
{
    PyObject_GC_UnTrack(it);
    DequeDictViewIter_clear(it);
    Py_TYPE(it)->tp_free(it);
}

static PyObject *
DequeDictViewIter_next(DequeDictViewIterObject *it)
{
    if (!it->current) return NULL;

    PyObject *result;
    if (it->kind == 0) {
        result = it->current->key;
        Py_INCREF(result);
    } else if (it->kind == 1) {
        result = it->current->value;
        Py_INCREF(result);
    } else {
        result = PyTuple_Pack(2, it->current->key, it->current->value);
    }

    it->current = it->reverse ? it->current->prev : it->current->next;
    return result;
}

static PyObject *
DequeDictView_iter(DequeDictViewObject *self)
{
    DequeDictViewIterObject *it = PyObject_GC_New(DequeDictViewIterObject, &DequeDictViewIter_Type);
    if (!it) return NULL;

    Py_INCREF(self->dd);
    it->dd = self->dd;
    it->current = self->dd->head;
    it->kind = self->kind;
    it->reverse = 0;
    PyObject_GC_Track(it);
    return (PyObject *)it;
}

static PyObject *
DequeDictView_reversed(DequeDictViewObject *self)
{
    DequeDictViewIterObject *it = PyObject_GC_New(DequeDictViewIterObject, &DequeDictViewIter_Type);
    if (!it) return NULL;

    Py_INCREF(self->dd);
    it->dd = self->dd;
    it->current = self->dd->tail;
    it->kind = self->kind;
    it->reverse = 1;
    PyObject_GC_Track(it);
    return (PyObject *)it;
}

static int
DequeDictKeysView_contains(DequeDictViewObject *self, PyObject *key)
{
    return PyDict_Contains(self->dd->dict, key);
}

static int
DequeDictValuesView_contains(DequeDictViewObject *self, PyObject *value)
{
    DequeDictEntry *entry = self->dd->head;
    while (entry) {
        int cmp = PyObject_RichCompareBool(entry->value, value, Py_EQ);
        if (cmp != 0) return cmp;
        entry = entry->next;
    }
    return 0;
}

static int
DequeDictItemsView_contains(DequeDictViewObject *self, PyObject *item)
{
    if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2)
        return 0;

    PyObject *key = PyTuple_GET_ITEM(item, 0);
    PyObject *value = PyTuple_GET_ITEM(item, 1);

    PyObject *capsule = PyDict_GetItem(self->dd->dict, key);
    if (!capsule) return 0;

    DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
    if (!entry) return 0;

    return PyObject_RichCompareBool(entry->value, value, Py_EQ);
}

static PySequenceMethods DequeDictKeysView_as_seq = {
    .sq_length = (lenfunc)DequeDictView_len,
    .sq_contains = (objobjproc)DequeDictKeysView_contains,
};

static PySequenceMethods DequeDictValuesView_as_seq = {
    .sq_length = (lenfunc)DequeDictView_len,
    .sq_contains = (objobjproc)DequeDictValuesView_contains,
};

static PySequenceMethods DequeDictItemsView_as_seq = {
    .sq_length = (lenfunc)DequeDictView_len,
    .sq_contains = (objobjproc)DequeDictItemsView_contains,
};

static PyMethodDef DequeDictKeysView_methods[] = {
    {"__reversed__", (PyCFunction)DequeDictView_reversed, METH_NOARGS, NULL},
    {NULL}
};

static PyMethodDef DequeDictValuesView_methods[] = {
    {"__reversed__", (PyCFunction)DequeDictView_reversed, METH_NOARGS, NULL},
    {NULL}
};

static PyMethodDef DequeDictItemsView_methods[] = {
    {"__reversed__", (PyCFunction)DequeDictView_reversed, METH_NOARGS, NULL},
    {NULL}
};

static PyTypeObject DequeDictViewIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictViewIter",
    .tp_basicsize = sizeof(DequeDictViewIterObject),
    .tp_dealloc = (destructor)DequeDictViewIter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictViewIter_traverse,
    .tp_clear = (inquiry)DequeDictViewIter_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)DequeDictViewIter_next,
};

static PyTypeObject DequeDictKeysView_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictKeys",
    .tp_basicsize = sizeof(DequeDictViewObject),
    .tp_dealloc = (destructor)DequeDictView_dealloc,
    .tp_as_sequence = &DequeDictKeysView_as_seq,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictView_traverse,
    .tp_clear = (inquiry)DequeDictView_clear,
    .tp_iter = (getiterfunc)DequeDictView_iter,
    .tp_methods = DequeDictKeysView_methods,
};

static PyTypeObject DequeDictValuesView_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictValues",
    .tp_basicsize = sizeof(DequeDictViewObject),
    .tp_dealloc = (destructor)DequeDictView_dealloc,
    .tp_as_sequence = &DequeDictValuesView_as_seq,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictView_traverse,
    .tp_clear = (inquiry)DequeDictView_clear,
    .tp_iter = (getiterfunc)DequeDictView_iter,
    .tp_methods = DequeDictValuesView_methods,
};

static PyTypeObject DequeDictItemsView_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictItems",
    .tp_basicsize = sizeof(DequeDictViewObject),
    .tp_dealloc = (destructor)DequeDictView_dealloc,
    .tp_as_sequence = &DequeDictItemsView_as_seq,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictView_traverse,
    .tp_clear = (inquiry)DequeDictView_clear,
    .tp_iter = (getiterfunc)DequeDictView_iter,
    .tp_methods = DequeDictItemsView_methods,
};

/* keys() - O(1) returns view */
static PyObject *
DequeDict_keys(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    DequeDictViewObject *view = PyObject_GC_New(DequeDictViewObject, &DequeDictKeysView_Type);
    if (!view) return NULL;
    Py_INCREF(self);
    view->dd = self;
    view->kind = 0;
    PyObject_GC_Track(view);
    return (PyObject *)view;
}

/* values() - O(1) returns view */
static PyObject *
DequeDict_values(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    DequeDictViewObject *view = PyObject_GC_New(DequeDictViewObject, &DequeDictValuesView_Type);
    if (!view) return NULL;
    Py_INCREF(self);
    view->dd = self;
    view->kind = 1;
    PyObject_GC_Track(view);
    return (PyObject *)view;
}

/* items() - O(1) returns view */
static PyObject *
DequeDict_items(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    DequeDictViewObject *view = PyObject_GC_New(DequeDictViewObject, &DequeDictItemsView_Type);
    if (!view) return NULL;
    Py_INCREF(self);
    view->dd = self;
    view->kind = 2;
    PyObject_GC_Track(view);
    return (PyObject *)view;
}

/* clear() - Python method */
static PyObject *
DequeDict_clear_method(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    DequeDictEntry *entry = self->head;
    while (entry) {
        DequeDictEntry *next = entry->next;
        Py_DECREF(entry->key);
        Py_DECREF(entry->value);
        entry_free(entry);
        entry = next;
    }

    PyDict_Clear(self->dict);
    self->head = NULL;
    self->tail = NULL;
    self->size = 0;
    DequeDict_invalidate_cache(self);

    Py_RETURN_NONE;
}

/* copy() */
static PyObject *
DequeDict_copy(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    PyObject *items = DequeDict_items(self, NULL);
    if (!items) return NULL;

    PyObject *result = PyObject_CallFunctionObjArgs((PyObject *)&DequeDict_Type, items, NULL);
    Py_DECREF(items);
    return result;
}

/* update(other) */
static PyObject *
DequeDict_update(DequeDictObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *other = NULL;

    if (!PyArg_ParseTuple(args, "|O", &other))
        return NULL;

    if (other) {
        if (PyDict_Check(other)) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(other, &pos, &key, &value)) {
                if (DequeDict_setitem(self, key, value) < 0)
                    return NULL;
            }
        } else {
            PyObject *iter = PyObject_GetIter(other);
            if (!iter) return NULL;

            PyObject *pair;
            while ((pair = PyIter_Next(iter)) != NULL) {
                if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    PyErr_SetString(PyExc_ValueError,
                        "update requires sequence of (key, value) pairs");
                    return NULL;
                }

                PyObject *key = PyTuple_GET_ITEM(pair, 0);
                PyObject *value = PyTuple_GET_ITEM(pair, 1);

                if (DequeDict_setitem(self, key, value) < 0) {
                    Py_DECREF(pair);
                    Py_DECREF(iter);
                    return NULL;
                }
                Py_DECREF(pair);
            }
            Py_DECREF(iter);
            if (PyErr_Occurred()) return NULL;
        }
    }

    /* Process keyword arguments */
    if (kwds) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            if (DequeDict_setitem(self, key, value) < 0)
                return NULL;
        }
    }

    Py_RETURN_NONE;
}

/* setdefault(key, default=None) */
static PyObject *
DequeDict_setdefault(DequeDictObject *self, PyObject *args)
{
    PyObject *key;
    PyObject *default_val = Py_None;

    if (!PyArg_ParseTuple(args, "O|O", &key, &default_val))
        return NULL;

    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (capsule) {
        DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
        if (entry) {
            Py_INCREF(entry->value);
            return entry->value;
        }
    }

    /* Key doesn't exist - add it */
    if (DequeDict_setitem(self, key, default_val) < 0)
        return NULL;

    Py_INCREF(default_val);
    return default_val;
}

/* __class_getitem__(params) - support generic subscript syntax e.g. DequeDict[str, int] */
static PyObject *
DequeDict_class_getitem(PyObject *cls, PyObject *args)
{
    return Py_GenericAlias(cls, args);
}

/* at(index) - O(1) via entry pointer cache */
static PyObject *
DequeDict_at(DequeDictObject *self, PyObject *args)
{
    Py_ssize_t index;

    if (!PyArg_ParseTuple(args, "n", &index))
        return NULL;

    /* Build cache if needed */
    if (!self->index_cache) {
        if (DequeDict_rebuild_cache(self) < 0)
            return NULL;
        if (!self->index_cache) {
            /* Empty dict, size==0, no cache allocated */
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
    }

    Py_ssize_t logical_size = self->cache_size - self->cache_offset;

    if (index < 0)
        index += logical_size;

    if (index < 0 || index >= logical_size) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    DequeDictEntry *entry = self->index_cache[self->cache_offset + index];
    Py_INCREF(entry->value);
    return entry->value;
}

/* move_to_end(key, last=True) - O(1) move key to front or back — invalidates cache */
static PyObject *
DequeDict_move_to_end(DequeDictObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"key", "last", NULL};
    PyObject *key;
    int last = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist, &key, &last))
        return NULL;

    PyObject *capsule = PyDict_GetItem(self->dict, key);
    if (!capsule) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    DequeDictEntry *entry = (DequeDictEntry *)PyLong_AsVoidPtr(capsule);
    if (!entry) return NULL;

    /* Already at the right position? */
    if ((last && entry == self->tail) || (!last && entry == self->head)) {
        Py_RETURN_NONE;
    }

    /* Unlink entry */
    if (entry->prev) {
        entry->prev->next = entry->next;
    } else {
        self->head = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    } else {
        self->tail = entry->prev;
    }

    DequeDict_invalidate_cache(self);

    if (last) {
        /* Move to end */
        entry->prev = self->tail;
        entry->next = NULL;
        if (self->tail) {
            self->tail->next = entry;
        } else {
            self->head = entry;
        }
        self->tail = entry;
    } else {
        /* Move to front */
        entry->prev = NULL;
        entry->next = self->head;
        if (self->head) {
            self->head->prev = entry;
        } else {
            self->tail = entry;
        }
        self->head = entry;
    }

    Py_RETURN_NONE;
}

/* Iterator */
typedef struct {
    PyObject_HEAD
    DequeDictObject *dequedict;
    DequeDictEntry *current;
} DequeDictIterObject;

static PyTypeObject DequeDictIter_Type;

static int
DequeDictIter_traverse(DequeDictIterObject *it, visitproc visit, void *arg)
{
    Py_VISIT(it->dequedict);
    return 0;
}

static int
DequeDictIter_clear(DequeDictIterObject *it)
{
    Py_CLEAR(it->dequedict);
    return 0;
}

static void
DequeDictIter_dealloc(DequeDictIterObject *it)
{
    PyObject_GC_UnTrack(it);
    DequeDictIter_clear(it);
    Py_TYPE(it)->tp_free(it);
}

static PyObject *
DequeDictIter_next(DequeDictIterObject *it)
{
    if (!it->current) return NULL;

    PyObject *key = it->current->key;
    Py_INCREF(key);
    it->current = it->current->next;
    return key;
}

static PyObject *
DequeDict_iter(DequeDictObject *self)
{
    DequeDictIterObject *it = PyObject_GC_New(DequeDictIterObject, &DequeDictIter_Type);
    if (!it) return NULL;

    Py_INCREF(self);
    it->dequedict = self;
    it->current = self->head;
    PyObject_GC_Track(it);
    return (PyObject *)it;
}

/* Reverse iterator */
typedef struct {
    PyObject_HEAD
    DequeDictObject *dequedict;
    DequeDictEntry *current;
} DequeDictRevIterObject;

static PyTypeObject DequeDictRevIter_Type;

static int
DequeDictRevIter_traverse(DequeDictRevIterObject *it, visitproc visit, void *arg)
{
    Py_VISIT(it->dequedict);
    return 0;
}

static int
DequeDictRevIter_clear(DequeDictRevIterObject *it)
{
    Py_CLEAR(it->dequedict);
    return 0;
}

static void
DequeDictRevIter_dealloc(DequeDictRevIterObject *it)
{
    PyObject_GC_UnTrack(it);
    DequeDictRevIter_clear(it);
    Py_TYPE(it)->tp_free(it);
}

static PyObject *
DequeDictRevIter_next(DequeDictRevIterObject *it)
{
    if (!it->current) return NULL;

    PyObject *key = it->current->key;
    Py_INCREF(key);
    it->current = it->current->prev;
    return key;
}

static PyObject *
DequeDict_reversed(DequeDictObject *self, PyObject *Py_UNUSED(args))
{
    DequeDictRevIterObject *it = PyObject_GC_New(DequeDictRevIterObject, &DequeDictRevIter_Type);
    if (!it) return NULL;

    Py_INCREF(self);
    it->dequedict = self;
    it->current = self->tail;
    PyObject_GC_Track(it);
    return (PyObject *)it;
}

/* __eq__ */
static PyObject *
DequeDict_richcompare(DequeDictObject *self, PyObject *other, int op)
{
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyMapping_Check(other)) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }

    Py_ssize_t other_len = PyMapping_Size(other);
    if (other_len == -1) {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self->size != other_len) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }

    /* Compare each key-value pair */
    DequeDictEntry *entry = self->head;
    while (entry) {
        PyObject *other_val = PyObject_GetItem(other, entry->key);
        if (!other_val) {
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                PyErr_Clear();
                if (op == Py_EQ) Py_RETURN_FALSE;
                Py_RETURN_TRUE;
            }
            return NULL;
        }

        int cmp = PyObject_RichCompareBool(entry->value, other_val, Py_EQ);
        Py_DECREF(other_val);

        if (cmp < 0) return NULL;
        if (!cmp) {
            if (op == Py_EQ) Py_RETURN_FALSE;
            Py_RETURN_TRUE;
        }

        entry = entry->next;
    }

    if (op == Py_EQ) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
DequeDict_repr(DequeDictObject *self)
{
    if (self->size == 0) {
        return PyUnicode_FromString("DequeDict()");
    }

    PyObject *items = DequeDict_items(self, NULL);
    if (!items) return NULL;

    PyObject *repr = PyUnicode_FromFormat("DequeDict(%R)", items);
    Py_DECREF(items);
    return repr;
}

static PyMethodDef DequeDict_methods[] = {
    /* Deque-like operations - O(1) */
    {"peekleft", (PyCFunction)DequeDict_peekleft, METH_NOARGS,
     "Return first value without removing - O(1)"},
    {"peekleftitem", (PyCFunction)DequeDict_peekleftitem, METH_NOARGS,
     "Return first (key, value) without removing - O(1)"},
    {"peekleftkey", (PyCFunction)DequeDict_peekleftkey, METH_NOARGS,
     "Return first key without removing - O(1)"},
    {"peek", (PyCFunction)DequeDict_peek, METH_NOARGS,
     "Return last value without removing - O(1)"},
    {"peekitem", (PyCFunction)DequeDict_peekitem, METH_NOARGS,
     "Return last (key, value) without removing - O(1)"},
    {"popleft", (PyCFunction)DequeDict_popleft, METH_NOARGS,
     "Remove and return first value - O(1)"},
    {"popleftitem", (PyCFunction)DequeDict_popleftitem, METH_NOARGS,
     "Remove and return first (key, value) - O(1)"},
    {"pop", (PyCFunction)DequeDict_pop, METH_VARARGS,
     "Remove and return value by key or from end - O(1)"},
    {"popitem", (PyCFunction)DequeDict_popitem, METH_NOARGS,
     "Remove and return last (key, value) - O(1)"},
    {"appendleft", (PyCFunction)DequeDict_appendleft, METH_VARARGS,
     "Insert (key, value) at front - O(1)"},
    {"move_to_end", (PyCFunction)DequeDict_move_to_end, METH_VARARGS | METH_KEYWORDS,
     "Move key to front (last=False) or back (last=True) - O(1)"},

    /* Dict-like operations */
    {"get", (PyCFunction)DequeDict_get, METH_VARARGS, "D.get(k[,d]) -> D[k] if k in D, else d"},
    {"keys", (PyCFunction)DequeDict_keys, METH_NOARGS, "D.keys() -> list of keys in order"},
    {"values", (PyCFunction)DequeDict_values, METH_NOARGS, "D.values() -> list of values in order"},
    {"items", (PyCFunction)DequeDict_items, METH_NOARGS, "D.items() -> list of (key, value) in order"},
    {"clear", (PyCFunction)DequeDict_clear_method, METH_NOARGS, "D.clear() -- remove all items"},
    {"copy", (PyCFunction)DequeDict_copy, METH_NOARGS, "D.copy() -> a shallow copy"},
    {"update", (PyCFunction)DequeDict_update, METH_VARARGS | METH_KEYWORDS, "D.update([E, ]**F)"},
    {"setdefault", (PyCFunction)DequeDict_setdefault, METH_VARARGS, "D.setdefault(k[,d])"},
    {"at", (PyCFunction)DequeDict_at, METH_VARARGS,
     "Return value at index position. O(1) via entry pointer cache."},
    {"__reversed__", (PyCFunction)DequeDict_reversed, METH_NOARGS, "D.__reversed__() -- return reverse iterator"},
    {"__class_getitem__", (PyCFunction)DequeDict_class_getitem, METH_O | METH_CLASS,
     "See PEP 585"},
    {NULL}
};

static PySequenceMethods DequeDict_as_sequence = {
    .sq_contains = (objobjproc)DequeDict_contains,
};

static PyMappingMethods DequeDict_as_mapping = {
    .mp_length = (lenfunc)DequeDict_len,
    .mp_subscript = (binaryfunc)DequeDict_getitem,
    .mp_ass_subscript = (objobjargproc)DequeDict_setitem,
};

static PyTypeObject DequeDictIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictIter",
    .tp_basicsize = sizeof(DequeDictIterObject),
    .tp_dealloc = (destructor)DequeDictIter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictIter_traverse,
    .tp_clear = (inquiry)DequeDictIter_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)DequeDictIter_next,
};

static PyTypeObject DequeDictRevIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDictRevIter",
    .tp_basicsize = sizeof(DequeDictRevIterObject),
    .tp_dealloc = (destructor)DequeDictRevIter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)DequeDictRevIter_traverse,
    .tp_clear = (inquiry)DequeDictRevIter_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)DequeDictRevIter_next,
};

static PyTypeObject DequeDict_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dequedict.DequeDict",
    .tp_basicsize = sizeof(DequeDictObject),
    .tp_dealloc = (destructor)DequeDict_dealloc,
    .tp_hash = PyObject_HashNotImplemented,
    .tp_repr = (reprfunc)DequeDict_repr,
    .tp_as_sequence = &DequeDict_as_sequence,
    .tp_as_mapping = &DequeDict_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_doc = "Ordered dictionary with O(1) deque operations at both ends.\n\n"
              "Provides dict-like key lookup plus efficient popleft/peekleft operations.\n"
              "Similar to collections.OrderedDict but with deque-like operations.",
    .tp_traverse = (traverseproc)DequeDict_traverse,
    .tp_clear = (inquiry)DequeDict_clear,
    .tp_richcompare = (richcmpfunc)DequeDict_richcompare,
    .tp_iter = (getiterfunc)DequeDict_iter,
    .tp_methods = DequeDict_methods,
    .tp_init = (initproc)DequeDict_init,
    .tp_new = PyType_GenericNew,
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dequedict._dequedict",
    .m_doc = "Ordered dictionary with O(1) deque operations",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__dequedict(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    if (PyType_Ready(&DequeDictIter_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDictRevIter_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDictViewIter_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDictKeysView_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDictValuesView_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDictItemsView_Type) < 0) return NULL;
    if (PyType_Ready(&DequeDict_Type) < 0) return NULL;

    Py_INCREF(&DequeDict_Type);
    PyModule_AddObject(m, "DequeDict", (PyObject *)&DequeDict_Type);

    return m;
}

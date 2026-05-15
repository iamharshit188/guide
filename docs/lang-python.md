# Python — Internals + Production Patterns

---

## Quick Reference

| Topic | Key Facts |
|-------|-----------|
| Reference counting | CPython's primary GC; `sys.getrefcount(x)` shows count + 1 (argument copy) |
| Cyclic GC | Runs periodically; finds reference cycles unreachable from roots |
| GIL | One thread executes Python bytecode at a time; released on I/O and C extensions |
| `__slots__` | Replaces per-instance `__dict__` with fixed-offset C struct; saves 40–200 bytes/instance |
| Generator | `yield` suspends frame; `send(val)` resumes and passes value to `yield` expression |
| `asyncio` | Single-threaded cooperative concurrency; event loop drives coroutines |
| Descriptor | Object with `__get__`/`__set__`/`__delete__`; powers `property`, `classmethod`, `staticmethod` |
| Metaclass | Class of a class; `type` by default; controls class creation |
| `functools.wraps` | Copies `__name__`, `__doc__`, `__annotations__`, `__wrapped__` to wrapper |
| `dataclass` | Auto-generates `__init__`, `__repr__`, `__eq__`; `frozen=True` adds `__hash__` |
| `NamedTuple` | Immutable; tuple subclass; index + attribute access; memory = tuple |
| `TypedDict` | Type-checking only; runtime = plain `dict`; no enforcement |
| List comprehension | $O(n)$; faster than equivalent `for` + `append` (avoids attr lookup each iter) |
| `is` vs `==` | `is` checks identity (same object in memory); `==` calls `__eq__` |

---

## Core Concepts

### CPython Memory Model

**Reference counting:**

Every object has a `ob_refcnt` field. Incremented on assignment, function call, container insertion. Decremented when variable goes out of scope, reassigned, or container element removed. When count reaches zero, `tp_dealloc` is called immediately — deterministic resource release for non-cyclic objects.

```python
import sys
x = [1, 2, 3]
sys.getrefcount(x)   # 2: one for x, one for the getrefcount argument
y = x
sys.getrefcount(x)   # 3
del y
sys.getrefcount(x)   # 2
```

**Integer caching:** CPython caches integers in [-5, 256] as singletons. `a = 256; b = 256; a is b` → `True`. `a = 257; b = 257; a is b` → `False` (unless in same compilation unit — implementation detail).

**String interning:** Short strings that look like identifiers are interned (`sys.intern`). `'hello' is 'hello'` → usually `True`; `'hello world' is 'hello world'` → implementation-defined.

**Cyclic garbage collector:**

Reference counting cannot collect cycles: if `A` references `B` and `B` references `A`, neither reaches zero even if both are unreachable from the root set. CPython's `gc` module runs a generational collector with three generations (threshold: 700/10/10 new objects by default). It detects cycles by temporarily decrementing ref counts along edges — objects with count > 0 after traversal are reachable from outside; those that reach 0 are garbage.

```python
import gc
gc.get_threshold()   # (700, 10, 10)
gc.collect()         # force full collection; returns number of unreachable objects
gc.disable()         # disable for performance-critical sections
```

**Memory allocator layers:**

```
Python object  → pymalloc (arena allocator for objects ≤ 512 bytes)
                      ↓
               → system malloc (for larger objects)
                      ↓
               → OS virtual memory
```

`pymalloc` uses 256 KB arenas divided into 4 KB pools of fixed-size blocks. Avoids per-object calls to system `malloc` — critical for throughput.

### GIL and Its Implications

The Global Interpreter Lock is a mutex protecting the CPython interpreter state. Only one thread holds the GIL and executes bytecode at a time. It is released:
- Every 100 bytecodes by default (`sys.setswitchinterval`, default 5ms in Python 3.2+)
- During I/O calls (`socket.recv`, `open().read()`, etc.)
- During calls to C extensions that explicitly release it (`numpy`, `pandas`, `ctypes`)

**Implication table:**

| Workload | Threading | Multiprocessing | asyncio |
|---------|-----------|----------------|---------|
| CPU-bound pure Python | No speedup (GIL blocks) | Yes — each process has own GIL | No |
| I/O-bound | Yes — GIL released during I/O | Yes, overkill | Yes, best choice |
| CPU-bound with numpy | Yes — numpy releases GIL for BLAS | Yes | No |
| Shared state needed | Easy with threading | Expensive (IPC) | Easy (single-threaded) |

Python 3.13+ introduces an optional no-GIL build (`--disable-gil`). Single-threaded performance is ~10% slower; multi-threaded CPU-bound code scales with core count.

**`concurrent.futures`:**
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
with ThreadPoolExecutor(max_workers=8) as ex:   # I/O bound
    results = list(ex.map(fetch_url, urls))
with ProcessPoolExecutor(max_workers=4) as ex:  # CPU bound
    results = list(ex.map(crunch, data_chunks))
```

### Generators & Coroutines

A generator function contains `yield`. Calling it returns a generator object without executing the body. Each `next()` call resumes execution until the next `yield`.

**Generator internals:** Python saves the entire frame (local variables, bytecode pointer, stack) in a `PyFrameObject` on the heap. `yield` suspends the frame; `next()` restores it. Memory: $O(1)$ for the generator object itself, regardless of how many values it can produce.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

gen = fibonacci()
vals = [next(gen) for _ in range(10)]
```

**`send` and `throw`:**

```python
def accumulator():
    total = 0
    while True:
        val = yield total    # yield sends total out; receives next send() arg
        if val is None:
            break
        total += val

acc = accumulator()
next(acc)          # prime: advance to first yield
acc.send(10)       # total = 10
acc.send(20)       # total = 30
```

**`yield from`:** Delegates to a sub-generator, transparently forwarding `send`, `throw`, and `close`. Return value of sub-generator becomes the value of the `yield from` expression.

**Native coroutines (`async def`):**

`async def f()` creates a coroutine object. `await expr` suspends `f` if `expr` is not ready, returning control to the event loop. Under the hood, coroutine objects implement the same `__next__`/`send`/`throw` protocol as generators — they are generators with `CO_COROUTINE` flag.

### asyncio Event Loop Internals

`asyncio` is single-threaded cooperative concurrency. The event loop maintains:
- A **ready queue** of callbacks to run now
- A **scheduled queue** (min-heap by time) for `call_later`/`call_at`
- A **selector** (`epoll` on Linux, `kqueue` on macOS) watching I/O readiness

**One iteration of the event loop:**
1. Drain the ready queue — run all pending callbacks
2. Poll the selector with a timeout = time until next scheduled callback
3. Schedule I/O-ready callbacks onto the ready queue
4. Run time-due scheduled callbacks

`asyncio.sleep(0)` yields control to the event loop without waiting — lets other coroutines run.

```python
import asyncio

async def fetch(url: str, delay: float) -> str:
    await asyncio.sleep(delay)    # yields to event loop; no thread blocked
    return f"data from {url}"

async def main():
    tasks = [
        asyncio.create_task(fetch("a.com", 0.3)),
        asyncio.create_task(fetch("b.com", 0.1)),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

asyncio.run(main())
```

**Blocking calls in asyncio:** Any synchronous blocking call (file I/O, `time.sleep`, CPU computation) blocks the entire event loop. Use `loop.run_in_executor(None, blocking_fn, arg)` to offload to a thread pool.

### Decorators & Descriptors

**Decorator:** A callable that takes a function and returns a replacement. Applied at class/function definition time.

```python
import functools
import time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-t0:.4f}s")
        return result
    return wrapper

@timed
def slow(n):
    return sum(range(n))
```

Decorator with arguments requires an extra layer: `@retry(times=3)` means `retry(times=3)` returns the actual decorator.

**Descriptor protocol:** Any object defining `__get__`, `__set__`, or `__delete__`. Stored as a class attribute. When accessed on an instance, Python's attribute lookup calls the descriptor methods instead of returning the object directly.

- **Non-data descriptor:** only `__get__` (e.g., `staticmethod`, `classmethod`, functions). Instance `__dict__` shadows them.
- **Data descriptor:** `__get__` + `__set__` (e.g., `property`). Takes priority over instance `__dict__`.

Lookup order: data descriptors → instance `__dict__` → non-data descriptors → class `__dict__`.

### Metaclasses

A metaclass is the class of a class. `type` is the default metaclass. `type(name, bases, namespace)` creates a new class object.

Metaclass hooks:
- `__prepare__(mcs, name, bases)` — returns the namespace dict (allows ordered or custom dicts)
- `__new__(mcs, name, bases, namespace)` — creates the class object
- `__init__(cls, name, bases, namespace)` — initializes the class
- `__call__(cls, *args, **kwargs)` — called when creating instances of `cls`

Common use cases: registering subclasses automatically, enforcing abstract methods, ORM field collection (Django `ModelBase`), singleton enforcement.

### `__slots__`

By default, each instance has a `__dict__` (a hash table) for storing attributes. `__slots__` replaces this with a fixed-size C-level struct of typed slots.

```python
class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Benefits: ~40–200 bytes less memory per instance, faster attribute access (direct offset vs hash lookup), prevents accidental attribute creation. Limitation: cannot add arbitrary attributes; inheritance requires re-declaring slots; `__weakref__` and `__dict__` must be explicitly included if needed.

### Context Managers

Implement `__enter__` and `__exit__`. `__exit__(self, exc_type, exc_val, exc_tb)` — if it returns a truthy value, the exception is suppressed.

```python
class Transaction:
    def __enter__(self):
        self.conn.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False   # do not suppress exceptions
```

`contextlib.contextmanager`: implement with a generator yielding exactly once — code before `yield` is `__enter__`, code after is `__exit__`.

### Type Hints & mypy

Type hints are annotations stored in `__annotations__`; not enforced at runtime by default.

```python
from typing import Optional, Union, Callable, TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: list[T] = []
    def push(self, item: T) -> None:
        self._data.append(item)
    def pop(self) -> T:
        return self._data.pop()
```

**Key typing constructs:**

| Construct | Meaning |
|-----------|---------|
| `Optional[T]` | `T \| None` |
| `Union[A, B]` | `A \| B` (Python 3.10+: `A \| B` directly) |
| `Callable[[A, B], R]` | Function taking A, B returning R |
| `TypeVar('T', bound=X)` | T must be X or a subtype |
| `Protocol` | Structural subtyping (duck typing with type checking) |
| `Final` | Variable must not be reassigned |
| `Literal['a', 'b']` | Only specific literal values |
| `TypedDict` | Dict with specific string keys and typed values |
| `ParamSpec` (3.10+) | Captures parameter types for decorator typing |

Run `mypy --strict src/` for full type checking.

### dataclasses vs NamedTuple vs TypedDict

| Feature | `@dataclass` | `NamedTuple` | `TypedDict` |
|---------|-------------|-------------|------------|
| Mutability | Mutable by default; `frozen=True` for immutable | Always immutable | Mutable (plain dict) |
| Inheritance | Yes (Python class) | Limited | Yes (TypedDict) |
| Memory | Like regular class | Like tuple (compact) | Like dict |
| Runtime type check | No | No | No |
| Index access | No | Yes (`p[0]`) | By key |
| Default values | Yes (`field(default=...)`) | Yes | No |
| `__hash__` | Only if `frozen=True` or `eq=False` | Yes (tuple hash) | No |
| Use case | Mutable structured data, OOP patterns | Immutable records, CSV rows | API response schemas |

### Comprehension Complexity & itertools/functools

**Complexity:**
- List comprehension `[f(x) for x in lst]` — $O(n)$
- Dict comprehension `{k: v for k, v in items}` — $O(n)$ average; $O(n^2)$ if hash collisions
- Set comprehension — $O(n)$ average
- Generator expression — $O(1)$ creation, $O(n)$ total iteration

**itertools:**

| Function | Description |
|---------|-------------|
| `chain(*iters)` | Concatenate iterables lazily |
| `islice(it, n)` | Take first n elements without materializing |
| `groupby(it, key)` | Group consecutive equal keys (sort first) |
| `product(*iters)` | Cartesian product |
| `combinations(it, r)` | $\binom{n}{r}$ combinations |
| `accumulate(it, func)` | Running aggregate (default: sum) |
| `cycle(it)` | Infinite rotation |
| `takewhile(pred, it)` | Take while predicate holds |

**functools:**

| Function | Description |
|---------|-------------|
| `lru_cache(maxsize=128)` | Memoize with LRU eviction; `maxsize=None` = infinite |
| `cache` (3.9+) | `lru_cache(maxsize=None)` shorthand |
| `reduce(f, it, init)` | Left fold |
| `partial(f, *args)` | Partially apply arguments |
| `total_ordering` | Define `__eq__` + one comparison; get rest |
| `singledispatch` | Function overloading by first argument type |

### Multiprocessing vs Threading vs asyncio

| Criterion | `threading` | `multiprocessing` | `asyncio` |
|-----------|------------|-------------------|----------|
| GIL bypass | No (for Python code) | Yes (separate processes) | No |
| CPU-bound speedup | No | Yes | No |
| I/O-bound | Yes | Yes (overkill) | Yes (best) |
| Memory | Shared | Copied (fork) or piped | Shared |
| Overhead | Low (~8 KB stack) | High (~50 MB process) | Minimal |
| Communication | Shared objects (lock required) | Queue/Pipe/Value | Queues/primitives |
| Fault isolation | None (crash = crash all) | Yes | None |
| Debugging | Hard (race conditions) | Moderate | Moderate |

**Decision rule:**
- Network I/O at scale → `asyncio`
- CPU-bound Python → `multiprocessing` (or `numba`/`cython`/C extension)
- Wrapping blocking C lib → `threading` (if lib releases GIL) or `multiprocessing`
- Mixed I/O + CPU → `asyncio` + `run_in_executor` for CPU portions

---

## Code Examples

### Custom Decorator with `functools.wraps`

```python
import functools
import time
import logging
from typing import Callable, TypeVar, Any

F = TypeVar('F', bound=Callable[..., Any])

def retry(times: int = 3, delay: float = 0.5, exceptions: tuple = (Exception,)):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logging.warning(f"{func.__name__} attempt {attempt}/{times} failed: {e}")
                    if attempt < times:
                        time.sleep(delay * attempt)
            raise last_exc
        wrapper.retry_times = times
        return wrapper  # type: ignore[return-value]
    return decorator

@retry(times=3, delay=0.1, exceptions=(ValueError, RuntimeError))
def unstable(x: int) -> int:
    import random
    if random.random() < 0.6:
        raise ValueError("transient failure")
    return x * 2

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        print(unstable(21))
    except ValueError as e:
        print(f"all retries exhausted: {e}")
```

### Descriptor Protocol

```python
class Validated:
    def __set_name__(self, owner, name):
        self.name = name
        self.private = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private, None)

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private, value)

    def validate(self, value):
        return value

class Positive(Validated):
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric, got {type(value)}")
        if value <= 0:
            raise ValueError(f"{self.name} must be > 0, got {value}")
        return value

class BoundedStr(Validated):
    def __init__(self, max_len: int):
        self.max_len = max_len

    def validate(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be str")
        if len(value) > self.max_len:
            raise ValueError(f"{self.name} max length {self.max_len}, got {len(value)}")
        return value

class Product:
    price = Positive()
    name  = BoundedStr(max_len=50)

    def __init__(self, name: str, price: float):
        self.name  = name
        self.price = price

    def __repr__(self):
        return f"Product(name={self.name!r}, price={self.price})"

if __name__ == "__main__":
    p = Product("Widget", 9.99)
    print(p)
    try:
        p.price = -1
    except ValueError as e:
        print(f"caught: {e}")
    try:
        p.name = "x" * 60
    except ValueError as e:
        print(f"caught: {e}")
```

### Async Generator

```python
import asyncio
from typing import AsyncIterator

async def rate_limited_range(n: int, per_second: float) -> AsyncIterator[int]:
    delay = 1.0 / per_second
    for i in range(n):
        yield i
        await asyncio.sleep(delay)

async def paginate(url: str, pages: int) -> AsyncIterator[dict]:
    for page in range(1, pages + 1):
        await asyncio.sleep(0.05)
        yield {"page": page, "url": url, "items": list(range(page * 10, (page + 1) * 10))}

async def main():
    print("Rate limited output:")
    async for val in rate_limited_range(5, per_second=20):
        print(f"  val={val}")

    print("\nPaginated fetch:")
    async for page_data in paginate("https://api.example.com/data", pages=3):
        print(f"  page {page_data['page']}: {page_data['items'][:3]}...")

asyncio.run(main())
```

### Metaclass Example

```python
class RegistryMeta(type):
    _registry: dict[str, type] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:
            tag = namespace.get('tag') or name.lower()
            mcs._registry[tag] = cls
        return cls

    @classmethod
    def get(mcs, tag: str) -> type:
        if tag not in mcs._registry:
            raise KeyError(f"no handler registered for tag '{tag}'")
        return mcs._registry[tag]

class Handler(metaclass=RegistryMeta):
    def handle(self, payload: dict) -> str:
        raise NotImplementedError

class JsonHandler(Handler):
    tag = 'json'
    def handle(self, payload: dict) -> str:
        import json
        return json.dumps(payload)

class XmlHandler(Handler):
    tag = 'xml'
    def handle(self, payload: dict) -> str:
        items = ''.join(f'<{k}>{v}</{k}>' for k, v in payload.items())
        return f'<root>{items}</root>'

if __name__ == "__main__":
    data = {"status": "ok", "count": 42}
    for fmt in ('json', 'xml'):
        handler_cls = RegistryMeta.get(fmt)
        print(f"{fmt}: {handler_cls().handle(data)}")
```

---

## Interview Q&A

**Q1: Explain how CPython's reference counting and cyclic GC interact, and when the cyclic GC runs.**

Reference counting deallocates objects immediately when `ob_refcnt` drops to zero — deterministic, no pause. Cyclic GC runs when the number of new objects since the last collection exceeds the generation-0 threshold (default 700). It finds cycles by building a graph of container objects (`list`, `dict`, `set`, etc.), temporarily decrementing reference counts along edges, then identifying objects whose count reaches zero (unreachable). It promotes survivors to older generations (gen 1 threshold: 10 gen-0 collections; gen 2: 10 gen-1 collections). The cyclic GC does not handle non-container types (integers, strings). You can disable it (`gc.disable()`) for short-lived data-processing scripts where cycles are impossible — avoids pauses.

---

**Q2: What is the GIL, why does it exist, and what workloads does it hurt most?**

The GIL (Global Interpreter Lock) is a mutex that ensures only one thread executes Python bytecodes at a time. It exists because CPython's memory management (reference counting, object allocator) is not thread-safe — making every increment/decrement atomic would add overhead and complexity. It hurts CPU-bound multithreaded code most severely: threads cannot run Python code in parallel even on multi-core machines. I/O-bound code is largely unaffected because the GIL is released during system calls. Workarounds: `multiprocessing` (each process has its own GIL), C extensions that release the GIL (numpy, etc.), or Python 3.13+ no-GIL build.

---

**Q3: What is the difference between a generator and a coroutine in Python?**

A generator (function with `yield`) produces a sequence of values — control flows producer-to-consumer. A coroutine (function with `async def` / `await`, or `yield` used with `send`) is a two-way communication channel: the caller sends values in and receives values out. Coroutines are generators with the `CO_COROUTINE` flag and are driven by an event loop, not direct `next()` calls. `async def` functions return a coroutine object that must be `await`ed or wrapped in a task. Generators are pull-based; coroutines support cooperative multitasking via `await` suspension points.

---

**Q4: How does the descriptor protocol work, and what is the lookup priority order?**

When accessing `instance.attr`, Python's `object.__getattribute__` checks: (1) data descriptors in the class (and MRO) — objects with both `__get__` and `__set__`; (2) instance `__dict__`; (3) non-data descriptors and other class attributes. Data descriptors win over instance dict, which is why `property` (a data descriptor) cannot be shadowed by setting an instance attribute. Functions are non-data descriptors — they have `__get__` but not `__set__`. When accessed via an instance, `function.__get__(instance, cls)` returns a bound method. The `__set_name__` hook (called at class creation) lets descriptors know their attribute name without needing it passed manually.

---

**Q5: What are `__slots__` and when should you not use them?**

`__slots__` replaces the instance `__dict__` with a fixed C-level array of slots. Benefits: ~40–200 bytes less memory per instance, faster attribute access (direct offset), prevents typos creating new attributes. Avoid when: (1) you need dynamic attributes (`__dict__` is not available unless you include `'__dict__'` in `__slots__`); (2) you use multiple inheritance where multiple classes define non-overlapping slots (works but is fragile); (3) you need pickling without custom `__getstate__`/`__setstate__`; (4) you use mixins that rely on `__dict__`. Best applied to value objects created in tight loops (Point, Vector, graph nodes).

---

**Q6: Explain Python's asyncio event loop architecture and what "cooperative" means.**

The event loop is a single OS thread running an infinite loop: drain the ready queue (callbacks added by resolved futures, I/O events, `call_soon`), query the I/O selector for ready events, advance time-scheduled callbacks. "Cooperative" means a coroutine must explicitly yield control via `await` — there is no preemption. If a coroutine runs CPU-bound code for 100ms without `await`, all other coroutines stall for 100ms. This is fundamentally different from threading (preemptive — OS interrupts threads). The advantage: no race conditions on shared data (only one coroutine runs at a time), no locks needed for simple state. Blocking calls must be offloaded via `run_in_executor` or C extensions that release the GIL.

---

**Q7: What is the MRO (Method Resolution Order) and how does C3 linearization work?**

MRO determines the order in which base classes are searched for attribute/method lookup. Python uses C3 linearization, defined recursively:

$L[C] = C + merge(L[B_1], L[B_2], \ldots, [B_1, B_2, \ldots])$

At each step, take the head of the first list if it does not appear in the tail of any other list; remove it from all lists. This guarantees monotonicity (class before its bases), local precedence (left-to-right among bases), and consistency. `ClassName.__mro__` or `ClassName.mro()` shows the computed order. Inconsistent hierarchies raise `TypeError` at class definition time.

---

**Q8: What is the difference between `deepcopy` and shallow copy, and when does each matter?**

Shallow copy (`copy.copy`, `list[:]`, `dict.copy()`) creates a new container but does not recurse — elements are the same objects. Modifying a mutable element affects both copies. Deep copy (`copy.deepcopy`) recursively copies all objects, creating an independent tree. Deep copy handles cycles (via a memo dict of already-copied objects). Matters when: data contains mutable nested objects (lists of lists, dicts of lists). Does not matter for immutable objects (strings, tuples of immutables, numbers). `deepcopy` is significantly slower — avoid on large data; prefer explicit copying at the required depth.

---

**Q9: How does `functools.lru_cache` work internally, and what are its limitations?**

`lru_cache` wraps a function; on call, it hashes all positional and keyword arguments to produce a cache key. If the key exists in the cache (implemented as an ordered dict), it returns the cached value and moves the entry to the most-recently-used position. If not, it calls the function, stores the result, and evicts the least-recently-used entry if `maxsize` is exceeded. Limitations: (1) Arguments must be hashable — fails on list/dict args. (2) Cache is per-function, unbounded with `maxsize=None` — potential memory leak on large key spaces. (3) Not thread-safe in a fine-grained sense (CPython's GIL protects dict operations but cache hit/miss logic is non-atomic). (4) Does not handle `self` well on methods — caches across all instances; use `functools.cached_property` or a per-instance cache.

---

**Q10: Explain the difference between `multiprocessing`, `threading`, and `asyncio` for a web scraping workload hitting 1000 URLs.**

`asyncio` is the right choice: network I/O releases the event loop at each `await`, so thousands of in-flight requests can be managed with a single thread and minimal memory (each coroutine uses a few hundred bytes vs ~8 KB per thread). `threading` works but 1000 threads are expensive (8 GB stack space minimum) and the OS scheduler overhead is significant; use a pool of 50–100 threads with `ThreadPoolExecutor` if using requests (sync library). `multiprocessing` with 1000 processes is impractical — process creation overhead and memory cost dominate. Optimal: `asyncio` with `aiohttp` and a semaphore limiting concurrent connections to respect server rate limits (`asyncio.Semaphore(50)`).

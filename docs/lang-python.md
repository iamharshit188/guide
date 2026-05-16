# Python — Zero to Production Expert

---

## 1. Why Python

Python trades execution speed for development speed: its expressive syntax and vast standard library let you write in hours what takes days in C++. It dominates data science, ML, and backend engineering because numpy/torch/sklearn expose C/Fortran kernels behind a clean Python API, giving C-level throughput with Python ergonomics.

---

## 2. Running Python

### REPL vs Script

The REPL (Read-Eval-Print Loop) starts with `python3` — each expression is evaluated immediately. Scripts are `.py` files executed top-to-bottom.

```python
# REPL — type directly into the terminal
>>> 2 + 2
4
>>> x = [1, 2, 3]
>>> x
[1, 2, 3]
```

```python
# script.py — saved file
x = [1, 2, 3]
print(sum(x))
```

Run a script:

```bash
python3 script.py
```

### Shebang Line

The shebang makes a script directly executable on Unix without typing `python3`:

```python
#!/usr/bin/env python3
print("runs directly if chmod +x is set")
```

```bash
chmod +x script.py
./script.py
```

`/usr/bin/env python3` finds whichever `python3` is first on `PATH` — avoids hardcoding `/usr/bin/python3`.

### `if __name__ == "__main__":`

When Python imports a module, it sets `__name__` to the module's filename (without `.py`). When it runs a file directly, it sets `__name__` to `"__main__"`. The guard prevents top-level code from executing on import:

```python
def compute():
    return 42

def main():
    print(compute())

if __name__ == "__main__":
    main()
```

Without the guard, `main()` would fire every time another module does `import this_file`, which is almost never desired.

---

## 3. Variables & Types

### Dynamic Typing

Python variables are labels that point to objects. The object carries the type, not the variable. A variable can point to any type at any time.

```python
x = 10       # x points to an int object
x = "hello"  # x now points to a str object — no error
x = [1, 2]   # x now points to a list
```

### Built-in Types

| Type | Example | Notes |
|------|---------|-------|
| `int` | `42`, `-7`, `10_000` | Arbitrary precision; no overflow |
| `float` | `3.14`, `1e-5`, `float('inf')` | IEEE 754 double (64-bit) |
| `complex` | `3+4j`, `complex(1, 2)` | Real + imaginary parts |
| `bool` | `True`, `False` | Subclass of `int`; `True == 1`, `False == 0` |
| `str` | `"hi"`, `'hi'`, `"""multi"""` | Immutable sequence of Unicode code points |
| `bytes` | `b"hi"`, `bytes([72, 105])` | Immutable sequence of integers 0–255 |
| `NoneType` | `None` | Singleton; only value is `None` |

### `type()` and `isinstance()`

```python
type(42)          # <class 'int'>
type(42) is int   # True
isinstance(42, int)        # True
isinstance(True, int)      # True — bool is a subclass of int
isinstance(42, (int, str)) # True — accepts a tuple of types
```

Prefer `isinstance()` over `type() ==` because `isinstance` respects inheritance.

### Integer Interning (-5 to 256)

CPython pre-allocates integer objects for the range -5 through 256 as singletons. Assigning any integer in that range returns the same object.

```python
a = 100
b = 100
a is b   # True — same cached object

a = 300
b = 300
a is b   # False — new objects created each time

a = -5
b = -5
a is b   # True

a = -6
b = -6
a is b   # False
```

This is a CPython implementation detail — do not rely on it for correctness.

### String Interning

Python interns string literals that look like identifiers (letters, digits, underscores only). Strings with spaces or special characters may or may not be interned depending on context.

```python
a = "hello"
b = "hello"
a is b   # True — interned

a = "hello world"
b = "hello world"
a is b   # False (usually) — not automatically interned

import sys
a = sys.intern("hello world")
b = sys.intern("hello world")
a is b   # True — manually interned
```

### `id()` and `is` vs `==`

`id(obj)` returns the memory address of the object (in CPython). `is` compares identity (same object). `==` calls `__eq__` and compares value.

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True — same values
a is b   # False — different objects in memory

c = a
a is c   # True — same object
id(a) == id(c)   # True
```

Rule: use `==` for value comparison. Use `is` only for `None`, `True`, `False`, or sentinel checks.

---

## 4. Operators & Expressions

### Arithmetic

| Operator | Operation | Example | Result |
|----------|-----------|---------|--------|
| `+` | Addition | `3 + 2` | `5` |
| `-` | Subtraction | `3 - 2` | `1` |
| `*` | Multiplication | `3 * 2` | `6` |
| `/` | True division | `7 / 2` | `3.5` |
| `//` | Floor division | `7 // 2` | `3` |
| `%` | Modulo | `7 % 2` | `1` |
| `**` | Exponentiation | `2 ** 10` | `1024` |

### Comparison

```python
x = 5
x == 5   # True
x != 4   # True
x > 3    # True
x >= 5   # True
x < 10   # True
x <= 5   # True
```

### Logical

```python
True and False   # False
True or False    # True
not True         # False
```

### Bitwise

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `&` | AND | `5 & 3` | `1` |
| `\|` | OR | `5 \| 3` | `7` |
| `^` | XOR | `5 ^ 3` | `6` |
| `~` | NOT | `~5` | `-6` |
| `<<` | Left shift | `1 << 3` | `8` |
| `>>` | Right shift | `8 >> 2` | `2` |

### Identity and Membership

```python
x = [1, 2, 3]
x is None      # False
x is not None  # True
2 in x         # True
5 not in x     # True
```

### Walrus Operator `:=` (Python 3.8+)

Assigns and returns a value inside an expression. Avoids calling a function twice:

```python
import re

# without walrus — calls re.match twice if match is truthy
if re.match(r"\d+", s):
    m = re.match(r"\d+", s)
    print(m.group())

# with walrus
if m := re.match(r"\d+", s):
    print(m.group())
```

Useful in `while` loops:

```python
while chunk := file.read(8192):
    process(chunk)
```

### Chained Comparisons

Python allows chaining comparisons directly — each operand is evaluated once:

```python
1 < x < 10       # equivalent to (1 < x) and (x < 10)
0 <= i < len(lst)
a == b == c       # all three equal
```

### Short-Circuit Evaluation

`and` returns the first falsy operand or the last operand. `or` returns the first truthy operand or the last operand. The right side is never evaluated if the result is determined by the left.

```python
0 and expensive()   # 0 — expensive() never called
1 or expensive()    # 1 — expensive() never called

None or "default"   # "default"
"value" or "default"  # "value"

# common pattern for default values
config = user_config or {}
```

---

## 5. Control Flow

### if/elif/else

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

### for Loop

`for` iterates over any iterable — any object implementing `__iter__`:

```python
for item in [1, 2, 3]:
    print(item)

for ch in "hello":
    print(ch)

for key in {"a": 1, "b": 2}:
    print(key)
```

`enumerate` yields `(index, value)` pairs:

```python
for i, val in enumerate(["a", "b", "c"], start=1):
    print(i, val)   # 1 a, 2 b, 3 c
```

`zip` pairs elements from multiple iterables, stopping at the shortest:

```python
names = ["Alice", "Bob"]
scores = [95, 87]
for name, score in zip(names, scores):
    print(name, score)
```

`range(start, stop, step)` generates integers lazily:

```python
range(5)          # 0 1 2 3 4
range(2, 10, 2)   # 2 4 6 8
range(10, 0, -1)  # 10 9 8 ... 1
```

### while Loop

```python
n = 10
total = 0
while n > 0:
    total += n
    n -= 1
```

### break, continue, else on Loops

`break` exits the loop immediately. `continue` skips to the next iteration. The `else` clause on a loop runs if the loop completes without hitting `break` — almost universally unknown:

```python
for n in range(2, 20):
    for factor in range(2, n):
        if n % factor == 0:
            break   # n is composite
    else:
        print(n, "is prime")   # only runs if inner loop didn't break
```

Same pattern on `while`:

```python
while condition:
    if found:
        break
else:
    # ran to completion without break
    handle_not_found()
```

### pass

`pass` is a no-op statement. Used as a placeholder for empty blocks:

```python
class NotImplementedYet:
    pass

def todo():
    pass
```

### Ternary Expression

```python
x = "even" if n % 2 == 0 else "odd"
result = value if value is not None else default
```

---

## 6. Functions

### Definition and Return

```python
def add(a, b):
    return a + b

def nothing():
    pass   # implicitly returns None
```

### Default Arguments and Keyword Arguments

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")              # "Hello, Alice!"
greet("Bob", "Hi")          # "Hi, Bob!"
greet(greeting="Hey", name="Carol")  # keyword — order doesn't matter
```

### `*args` and `**kwargs`

`*args` collects extra positional arguments as a tuple. `**kwargs` collects extra keyword arguments as a dict.

```python
def variadic(*args, **kwargs):
    print(args)    # tuple
    print(kwargs)  # dict

variadic(1, 2, 3, x=10, y=20)
# (1, 2, 3)
# {'x': 10, 'y': 20}
```

Keyword-only arguments (after `*`):

```python
def connect(host, *, port=8080, timeout=30):
    pass

connect("localhost", port=9090)   # ok
connect("localhost", 9090)        # TypeError — port must be keyword
```

### Argument Passing: Pass-by-Object-Reference

Python does not pass by value (no copy) and not by reference (no pointer to variable). It passes by object reference — the function receives a reference to the same object.

- Immutable objects (int, str, tuple): cannot be mutated; rebinding inside the function doesn't affect the caller.
- Mutable objects (list, dict): mutations inside the function affect the caller; rebinding does not.

```python
def mutate(lst):
    lst.append(4)        # mutates the original

def rebind(lst):
    lst = [99, 100]      # local rebind — caller unaffected

x = [1, 2, 3]
mutate(x)
print(x)    # [1, 2, 3, 4]

rebind(x)
print(x)    # [1, 2, 3, 4] — unchanged
```

### Mutable Default Argument Trap

Default argument values are evaluated once at function definition, not on each call. A mutable default accumulates state across calls:

```python
# BUG
def append_to(item, lst=[]):
    lst.append(item)
    return lst

append_to(1)   # [1]
append_to(2)   # [1, 2] — NOT [2], same list reused
append_to(3)   # [1, 2, 3]

# FIX — use None as sentinel
def append_to(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### `global` and `nonlocal`

`global` declares that a name refers to the module-level variable, not a local one. `nonlocal` reaches into the enclosing (non-global) scope.

```python
count = 0

def increment():
    global count
    count += 1

def make_counter():
    n = 0
    def counter():
        nonlocal n
        n += 1
        return n
    return counter
```

### First-Class Functions

Functions are objects — they can be stored in variables, passed as arguments, and returned from functions.

```python
def square(x):
    return x * x

def apply(func, values):
    return [func(v) for v in values]

apply(square, [1, 2, 3, 4])   # [1, 4, 9, 16]

# returned from a function
def multiplier(factor):
    def inner(x):
        return x * factor
    return inner

triple = multiplier(3)
triple(10)   # 30
```

### Closures and Captured Variables

A closure is a function that captures variables from its enclosing scope. The captured variable is a reference to the cell object, not a snapshot of the value at creation time.

```python
def make_adder(n):
    def add(x):
        return x + n   # n is captured from make_adder's scope
    return add

add5 = make_adder(5)
add5(3)   # 8

# late-binding gotcha in loops
fns = [lambda: i for i in range(3)]
[f() for f in fns]   # [2, 2, 2] — all capture the same i

# fix: bind at creation time with default arg
fns = [lambda i=i: i for i in range(3)]
[f() for f in fns]   # [0, 1, 2]
```

### `functools.wraps`

When you wrap a function in a decorator, the wrapper replaces the original. Without `wraps`, the original's `__name__`, `__doc__`, and `__annotations__` are lost. `functools.wraps` copies them to the wrapper:

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    "Returns a greeting."
    return f"Hello, {name}"

greet.__name__   # "greet" — not "wrapper"
greet.__doc__    # "Returns a greeting."
```

### Docstrings

The first string literal in a function/class/module body becomes the `__doc__` attribute. Access with `help()`:

```python
def compute(x):
    "Returns x squared."
    return x ** 2

help(compute)
compute.__doc__   # "Returns x squared."
```

---

## 7. Data Structures (Built-in)

### list

Ordered, mutable sequence. Backed by a dynamic array — elements are stored as pointers to objects.

```python
lst = [1, 2, 3]
lst[0]        # 1 — indexing
lst[-1]       # 3 — negative index
lst[1:3]      # [2, 3] — slice
lst[::2]      # [1, 3] — step
lst[::-1]     # [3, 2, 1] — reverse
```

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Index `lst[i]` | O(1) | Direct pointer offset |
| Append | O(1) amortized | Over-allocates; occasional O(n) resize |
| Insert at i | O(n) | Shifts elements right |
| Pop from end | O(1) | |
| Pop from i | O(n) | Shifts elements left |
| Remove (value) | O(n) | Scans linearly |
| `in` | O(n) | Linear scan |
| `sort()` | O(n log n) | Timsort; stable |
| `len()` | O(1) | Stored as field |

```python
lst = [3, 1, 4, 1, 5]
lst.append(9)           # [3, 1, 4, 1, 5, 9]
lst.insert(2, 99)       # insert 99 at index 2
lst.pop()               # removes and returns last
lst.pop(0)              # removes and returns index 0
lst.remove(1)           # removes first occurrence of 1
lst.sort()              # in-place, ascending
lst.sort(reverse=True)  # in-place, descending
sorted(lst)             # returns new sorted list
lst.reverse()           # in-place reverse
lst.index(4)            # index of first occurrence
lst.count(1)            # count occurrences
lst.extend([6, 7])      # concatenate in-place
```

`sort()` vs `sorted()`: `sort()` is in-place and returns `None`; `sorted()` returns a new list, works on any iterable.

### tuple

Ordered, immutable sequence. Slightly smaller than list (no over-allocation). Use when data should not change.

```python
t = (1, 2, 3)
t[0]       # 1
t[-1]      # 3

# single-element tuple needs trailing comma
single = (42,)
not_tuple = (42)   # just int with parentheses
```

Unpacking:

```python
a, b, c = (1, 2, 3)
first, *rest = (1, 2, 3, 4)   # first=1, rest=[2,3,4]
*init, last = (1, 2, 3, 4)    # init=[1,2,3], last=4
```

`namedtuple`: tuple with named fields, zero overhead over plain tuple.

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
p.x     # 3
p[0]    # 3 — still indexable
p._asdict()   # OrderedDict
```

### dict

Unordered (insertion-ordered since Python 3.7) mapping from hashable keys to values. Implemented as a hash table with open addressing.

```python
d = {"a": 1, "b": 2}
d["a"]                  # 1
d["c"]                  # KeyError
d.get("c")              # None
d.get("c", 0)           # 0 — default
d["c"] = 3              # insert/update
del d["a"]              # delete
"b" in d                # True
```

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Get/Set/Delete | O(1) average | O(n) worst case (hash collision) |
| `in` | O(1) average | |
| Iteration | O(n) | |
| `len()` | O(1) | |

```python
d.keys()    # view of keys
d.values()  # view of values
d.items()   # view of (key, value) pairs
d.update({"x": 10})   # merge another dict
d.pop("b")            # remove and return
d.setdefault("y", 0)  # insert if missing, return value
```

Merge operator (Python 3.9+):

```python
merged = d1 | d2        # new dict
d1 |= d2                # update in place
```

Dict comprehension:

```python
squares = {x: x**2 for x in range(5)}
filtered = {k: v for k, v in d.items() if v > 0}
```

**Hash table internals:** Python computes `hash(key)` and maps it to a slot. Collision resolution uses open addressing with pseudo-random probing. Load factor is kept below ~2/3 by resizing. Keys must be hashable (immutable by convention). Two objects that compare equal must have the same hash — this is why mutable objects cannot be keys.

**`defaultdict`:** Calls a factory function for missing keys instead of raising `KeyError`:

```python
from collections import defaultdict

dd = defaultdict(list)
dd["a"].append(1)   # no KeyError — creates [] automatically
dd["a"].append(2)
# {"a": [1, 2]}

word_count = defaultdict(int)
for word in text.split():
    word_count[word] += 1
```

**`Counter`:** Subclass of dict for counting hashable objects:

```python
from collections import Counter

c = Counter("abracadabra")
c.most_common(3)    # [('a', 5), ('b', 2), ('r', 2)]
c["a"]              # 5
c["z"]              # 0 — no KeyError
c1 + c2             # add counts
c1 - c2             # subtract counts (drops negatives)
```

**`OrderedDict`:** Pre-3.7 solution for ordered dicts. Still useful for `move_to_end()` and order-sensitive equality:

```python
from collections import OrderedDict

od = OrderedDict()
od["a"] = 1
od["b"] = 2
od.move_to_end("a")   # moves "a" to the end
od.popitem(last=False) # removes first item (FIFO)
```

### set and frozenset

Unordered collection of unique hashable elements. Backed by a hash table.

```python
s = {1, 2, 3}
s.add(4)
s.discard(10)    # no error if missing
s.remove(10)     # KeyError if missing
len(s)
3 in s
```

| Operation | Syntax | Method |
|-----------|--------|--------|
| Union | `a \| b` | `a.union(b)` |
| Intersection | `a & b` | `a.intersection(b)` |
| Difference | `a - b` | `a.difference(b)` |
| Symmetric difference | `a ^ b` | `a.symmetric_difference(b)` |
| Subset | `a <= b` | `a.issubset(b)` |
| Superset | `a >= b` | `a.issuperset(b)` |
| Disjoint | — | `a.isdisjoint(b)` |

```python
a = {1, 2, 3}
b = {2, 3, 4}
a | b    # {1, 2, 3, 4}
a & b    # {2, 3}
a - b    # {1}
a ^ b    # {1, 4}
```

`frozenset` is immutable and hashable — can be used as a dict key or set element:

```python
fs = frozenset([1, 2, 3])
d = {fs: "value"}
```

`in` on a set is O(1). `in` on a list is O(n). Prefer set for membership testing on large collections.

### str

Immutable sequence of Unicode code points (Python 3). Indexing returns a one-character string, not a codepoint integer.

```python
s = "hello"
s[0]       # "h"
s[-1]      # "o"
s[1:4]     # "ell"
len(s)     # 5
```

| Method | Example | Result |
|--------|---------|--------|
| `upper()` | `"hi".upper()` | `"HI"` |
| `lower()` | `"HI".lower()` | `"hi"` |
| `strip()` | `" hi ".strip()` | `"hi"` |
| `lstrip()` / `rstrip()` | `" hi ".lstrip()` | `"hi "` |
| `split(sep)` | `"a,b".split(",")` | `["a","b"]` |
| `join(iterable)` | `",".join(["a","b"])` | `"a,b"` |
| `replace(old,new)` | `"hi".replace("h","H")` | `"Hi"` |
| `startswith(s)` | `"hello".startswith("he")` | `True` |
| `endswith(s)` | `"hello".endswith("lo")` | `True` |
| `find(s)` | `"hello".find("ll")` | `2` |
| `count(s)` | `"hello".count("l")` | `2` |
| `isdigit()` | `"123".isdigit()` | `True` |
| `isalpha()` | `"abc".isalpha()` | `True` |
| `zfill(n)` | `"42".zfill(5)` | `"00042"` |
| `center(n)` | `"hi".center(6)` | `"  hi  "` |

**f-strings (Python 3.6+):**

```python
name = "Alice"
score = 95.678

f"Hello, {name}!"              # "Hello, Alice!"
f"{score:.2f}"                 # "95.68"
f"{score:>10.2f}"              # "     95.68"
f"{1_000_000:,}"               # "1,000,000"
f"{'hi':^10}"                  # "    hi    "
f"{2**10 = }"                  # "2**10 = 1024" — debug format (3.8+)
```

**format():**

```python
"{} + {} = {}".format(1, 2, 3)         # positional
"{name} is {age}".format(name="Alice", age=30)
"{0:.2f}".format(3.14159)
```

String concatenation with `+` is O(n) per operation because each creates a new string. Use `"".join(list_of_strings)` for bulk concatenation — O(n) total.

---

## 8. Comprehensions

### Syntax Table

| Type | Syntax | Result type |
|------|--------|-------------|
| List | `[expr for x in iter if cond]` | `list` |
| Dict | `{k: v for x in iter if cond}` | `dict` |
| Set | `{expr for x in iter if cond}` | `set` |
| Generator | `(expr for x in iter if cond)` | `generator` |

```python
squares = [x**2 for x in range(10)]
even_sq  = [x**2 for x in range(10) if x % 2 == 0]
sq_dict  = {x: x**2 for x in range(5)}
sq_set   = {x**2 for x in range(-3, 4)}
sq_gen   = (x**2 for x in range(10))   # lazy
```

### Nested Comprehensions

Outer loop first, inner loop second — mirrors reading order of nested `for` loops:

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
```

### When NOT to Use Comprehensions

- When the logic requires multiple statements or is non-trivial — a regular `for` loop with named variables reads better.
- When you need to handle exceptions inside the loop.
- When you need to break/continue — comprehensions cannot do this.
- Deeply nested comprehensions (more than 2 levels) — extract to a function instead.

---

## 9. Object-Oriented Python

### Class Definition

```python
class Dog:
    species = "Canis familiaris"   # class variable — shared by all instances

    def __init__(self, name, age):
        self.name = name           # instance variable
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"
```

`self` is the instance — it is passed automatically when calling an instance method but must be declared explicitly as the first parameter.

### Instance vs Class vs Static Methods

| Method Type | Decorator | First Param | Access |
|-------------|-----------|-------------|--------|
| Instance | none | `self` | Instance state + class state |
| Class | `@classmethod` | `cls` | Class state only |
| Static | `@staticmethod` | none | Neither instance nor class |

```python
class Circle:
    pi = 3.14159

    def __init__(self, radius):
        self.radius = radius

    def area(self):                     # instance method
        return Circle.pi * self.radius ** 2

    @classmethod
    def from_diameter(cls, diameter):   # alternative constructor
        return cls(diameter / 2)

    @staticmethod
    def validate_radius(r):             # utility — no self or cls needed
        return r > 0

c1 = Circle(5)
c2 = Circle.from_diameter(10)
Circle.validate_radius(5)
```

### `@property` Getter/Setter/Deleter

Exposes a computed attribute as if it were a plain attribute — no parentheses on access:

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @celsius.deleter
    def celsius(self):
        del self._celsius

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

t = Temperature(25)
t.celsius     # 25 — calls getter
t.celsius = 30   # calls setter
t.fahrenheit  # 86.0 — computed property
```

### Inheritance and `super()`

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

class Cat(Animal):
    def speak(self):
        return f"{self.name} meows"

class GoldenRetriever(Dog):
    def __init__(self, name, color):
        super().__init__(name)   # calls Dog.__init__ (which calls Animal.__init__)
        self.color = color
```

### Method Resolution Order (MRO) — C3 Linearization

Python's MRO determines which class's method is used in multiple inheritance. Python uses the C3 linearization algorithm, which produces a consistent and monotonic ordering.

C3 rule: L[C] = C + merge(L[B1], L[B2], ..., [B1, B2, ...]) where B1, B2 are bases.

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

D.__mro__
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

D().method()   # "B" — first in MRO after D
```

Inspect MRO:

```python
D.__mro__         # tuple of classes
D.mro()           # same as list
```

### `__str__` vs `__repr__`

| Method | Called by | Purpose |
|--------|-----------|---------|
| `__repr__` | `repr()`, REPL display | Unambiguous developer representation; ideally `eval(repr(x)) == x` |
| `__str__` | `str()`, `print()` | Human-readable output |

If only `__repr__` is defined, `__str__` falls back to `__repr__`.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x!r}, {self.y!r})"

    def __str__(self):
        return f"({self.x}, {self.y})"
```

### Dunder Methods Reference Table

| Method | Triggered by | Notes |
|--------|-------------|-------|
| `__len__` | `len(obj)` | Return integer |
| `__getitem__` | `obj[key]` | Enables indexing and iteration fallback |
| `__setitem__` | `obj[key] = val` | |
| `__delitem__` | `del obj[key]` | |
| `__contains__` | `x in obj` | Falls back to `__iter__` if missing |
| `__iter__` | `iter(obj)`, `for x in obj` | Return an iterator |
| `__next__` | `next(obj)` | Raise `StopIteration` when done |
| `__enter__` | `with obj as x:` | Returns value bound to `as` target |
| `__exit__` | End of `with` block | Args: `(exc_type, exc_val, tb)`. Return truthy to suppress exception |
| `__call__` | `obj()` | Makes instances callable |
| `__eq__` | `obj == other` | Defining `__eq__` sets `__hash__ = None` unless you also define `__hash__` |
| `__hash__` | `hash(obj)`, dict key | Must be consistent with `__eq__` |
| `__lt__` | `obj < other` | Define all six, or use `@functools.total_ordering` |
| `__add__` | `obj + other` | |
| `__bool__` | `bool(obj)`, `if obj:` | Falls back to `__len__` |
| `__getattr__` | Attribute not found normally | Last-resort attribute lookup |
| `__setattr__` | `obj.attr = val` | Intercepts all attribute writes |

### Dataclasses

`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` based on annotated fields:

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0   # default value
    tags: list = field(default_factory=list)  # mutable default

@dataclass(frozen=True)   # immutable — generates __hash__
class ImmutablePoint:
    x: float
    y: float

@dataclass
class Vector:
    components: list

    def __post_init__(self):   # runs after generated __init__
        if not self.components:
            raise ValueError("Cannot have empty vector")
        self.magnitude = sum(c**2 for c in self.components) ** 0.5

p = Point(1.0, 2.0)
p.x    # 1.0
repr(p)  # "Point(x=1.0, y=2.0, z=0.0, tags=[])"
```

### `__slots__`

By default, each instance has a `__dict__` for storing attributes — a hash table with ~200+ bytes overhead per instance. `__slots__` replaces `__dict__` with fixed C-struct offsets, reducing memory by 40–200 bytes per instance and speeding up attribute access:

```python
class WithDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Tradeoffs with `__slots__`:

| Factor | Without slots | With slots |
|--------|--------------|------------|
| Memory per instance | ~400 bytes | ~200 bytes |
| Attribute access | Dict lookup | Direct offset |
| Dynamic attributes | Allowed | Not allowed |
| `__weakref__` | Supported | Need to add to `__slots__` |
| Multiple inheritance | Works freely | Complications with non-empty slots in both bases |

---

## 10. Iterators & Generators

### Iterator Protocol

An **iterable** is any object with `__iter__`. An **iterator** is an object with both `__iter__` (returns `self`) and `__next__` (returns next value or raises `StopIteration`).

```python
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        val = self.current
        self.current -= 1
        return val

for n in Countdown(3):
    print(n)   # 3 2 1
```

`for x in obj` is equivalent to:

```python
it = iter(obj)    # calls obj.__iter__()
while True:
    try:
        x = next(it)   # calls it.__next__()
    except StopIteration:
        break
```

### Generator Functions

A function containing `yield` is a generator function. Calling it returns a generator object — it does not execute the body immediately. Execution advances to the next `yield` only when `next()` is called. State (local variables, execution position) is preserved between calls.

```python
def squares(n):
    for i in range(n):
        yield i ** 2

gen = squares(5)
next(gen)   # 0
next(gen)   # 1
next(gen)   # 4

for s in squares(5):
    print(s)
```

Generators are memory-efficient — they produce one value at a time instead of building the full sequence:

```python
# list: computes all 10M values, stores in memory
big_list = [x**2 for x in range(10_000_000)]

# generator: computes one at a time
big_gen = (x**2 for x in range(10_000_000))
sum(big_gen)   # processes one element at a time
```

### Generator Expressions

Same as list comprehension but with `()` — returns a generator:

```python
gen = (x**2 for x in range(10) if x % 2 == 0)
```

### `send()`, `throw()`, `close()`

Generators are coroutines — they can receive values:

```python
def accumulator():
    total = 0
    while True:
        value = yield total   # yield current total, receive next value
        if value is None:
            break
        total += value

gen = accumulator()
next(gen)        # 0 — prime the generator (advance to first yield)
gen.send(10)     # 10
gen.send(20)     # 30
gen.send(5)      # 35
gen.close()      # raises GeneratorExit inside generator
```

`throw(exc)` raises an exception at the point where the generator is suspended:

```python
gen.throw(ValueError, "bad input")
```

### `yield from`

Delegates to a sub-generator. Handles `send()`, `throw()`, `close()` automatically.

```python
def chain(*iterables):
    for it in iterables:
        yield from it

list(chain([1, 2], [3, 4], [5]))   # [1, 2, 3, 4, 5]

# equivalent to
def chain(*iterables):
    for it in iterables:
        for item in it:
            yield item
```

### Infinite Generators

```python
def naturals(start=0):
    n = start
    while True:
        yield n
        n += 1

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

Use `itertools.islice` to take a finite slice of an infinite generator.

### itertools

```python
import itertools

list(itertools.chain([1,2], [3,4], [5]))   # [1, 2, 3, 4, 5]
list(itertools.islice(naturals(), 5))      # [0, 1, 2, 3, 4]
list(itertools.product("AB", repeat=2))   # [('A','A'),('A','B'),('B','A'),('B','B')]
list(itertools.combinations("ABC", 2))    # [('A','B'),('A','C'),('B','C')]
list(itertools.permutations("AB"))        # [('A','B'),('B','A')]
list(itertools.cycle("AB"))              # A B A B A B ... (infinite)
list(itertools.repeat(7, 3))             # [7, 7, 7]

# groupby — groups consecutive identical keys (input must be sorted by key)
data = [("a",1),("a",2),("b",3)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# a [('a', 1), ('a', 2)]
# b [('b', 3)]

# accumulate
list(itertools.accumulate([1,2,3,4]))          # [1, 3, 6, 10]
list(itertools.accumulate([1,2,3,4], max))     # [1, 2, 3, 4]

# starmap
list(itertools.starmap(pow, [(2,3),(3,2)]))    # [8, 9]

# takewhile / dropwhile
list(itertools.takewhile(lambda x: x<5, [1,3,5,7]))  # [1, 3]
list(itertools.dropwhile(lambda x: x<5, [1,3,5,7]))  # [5, 7]
```

---

## 11. Decorators

### How Function Decorators Work

A decorator is a callable that takes a function and returns a replacement. `@decorator` syntax is syntactic sugar:

```python
@decorator
def func():
    pass

# exactly equivalent to:
def func():
    pass
func = decorator(func)
```

Step-by-step example:

```python
import functools

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(2, 3)
# Calling add
# add returned 5
```

### Decorator with Arguments (Factory Pattern)

Add an extra outer function that accepts the arguments and returns the actual decorator:

```python
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}")

greet("Alice")   # prints three times
```

### Class Decorators

A class can be a decorator if it implements `__call__`:

```python
class Retry:
    def __init__(self, max_attempts):
        self.max_attempts = max_attempts

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts - 1:
                        raise
        return wrapper

@Retry(max_attempts=3)
def unstable():
    pass
```

### Stacking Decorators — Execution Order

Decorators apply bottom-up (innermost first) but execute top-down (outermost first at call time):

```python
@A
@B
@C
def func():
    pass
# equivalent to: func = A(B(C(func)))
# call order: A's wrapper → B's wrapper → C's wrapper → original func
```

### `functools.lru_cache` and `functools.cache`

`lru_cache` memoizes a function's return values keyed on arguments. Evicts least-recently-used entries when `maxsize` is exceeded. `cache` (Python 3.9+) is `lru_cache(maxsize=None)`:

```python
import functools

@functools.lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

fib(100)        # computed instantly via memoization
fib.cache_info()   # CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)
fib.cache_clear()  # clear the cache

@functools.cache   # unbounded cache
def fib_v2(n):
    if n < 2:
        return n
    return fib_v2(n-1) + fib_v2(n-2)
```

Requirements: arguments must be hashable (no lists, dicts as args).

### `functools.partial`

Creates a new callable with some arguments pre-filled:

```python
import functools

def power(base, exponent):
    return base ** exponent

square = functools.partial(power, exponent=2)
cube   = functools.partial(power, exponent=3)

square(5)   # 25
cube(3)     # 27

# useful with map/filter
double = functools.partial(lambda x, n: x * n, n=2)
list(map(double, [1, 2, 3]))   # [2, 4, 6]
```

---

## 12. Context Managers

### `with` Statement Mechanics

```python
with expression as target:
    body
```

Equivalent to:

```python
mgr = expression
target = mgr.__enter__()
try:
    body
except:
    if not mgr.__exit__(*sys.exc_info()):
        raise
else:
    mgr.__exit__(None, None, None)
```

### `__enter__` and `__exit__` Protocol

```python
class ManagedFile:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.file = open(self.path, self.mode)
        return self.file        # bound to 'as' target

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False            # False = do not suppress exceptions

with ManagedFile("data.txt", "r") as f:
    content = f.read()
```

### Exception Suppression in `__exit__`

`__exit__` receives `(exc_type, exc_val, traceback)`. Return `True` to suppress the exception; return `False` (or `None`) to propagate it:

```python
class SuppressZeroDivision:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is ZeroDivisionError:
            print("Suppressed ZeroDivisionError")
            return True     # suppress
        return False        # reraise anything else

with SuppressZeroDivision():
    x = 1 / 0   # suppressed — execution continues after the with block
print("still running")
```

### `contextlib.contextmanager`

Convert a generator function into a context manager. Code before `yield` is `__enter__`; code after is `__exit__`:

```python
from contextlib import contextmanager

@contextmanager
def managed_file(path, mode):
    f = open(path, mode)
    try:
        yield f            # value bound to 'as' target
    finally:
        f.close()

@contextmanager
def timer():
    import time
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.4f}s")

with timer():
    total = sum(range(10_000_000))
```

### `contextlib.suppress`

Suppresses specified exceptions:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    open("missing.txt")   # no error raised — execution continues
```

### ExitStack for Dynamic Context Managers

`ExitStack` allows entering a variable number of context managers or registering cleanup callbacks:

```python
from contextlib import ExitStack

files = ["a.txt", "b.txt", "c.txt"]

with ExitStack() as stack:
    handles = [stack.enter_context(open(f)) for f in files]
    # all three files open; all closed when block exits

# register arbitrary cleanup
with ExitStack() as stack:
    stack.callback(print, "cleanup ran")
    stack.callback(some_cleanup_function)
    # callbacks run LIFO on exit
```

---

## 13. Exceptions

### Exception Hierarchy (ASCII)

```
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── AttributeError
    ├── ImportError
    │   └── ModuleNotFoundError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── NameError
    │   └── UnboundLocalError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── TimeoutError
    ├── RuntimeError
    │   └── RecursionError
    ├── StopIteration
    ├── TypeError
    ├── ValueError
    │   └── UnicodeError
    └── Warning
        ├── DeprecationWarning
        └── UserWarning
```

### try/except/else/finally

```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except (TypeError, KeyError) as e:
    print(f"Type or key error: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
    raise   # re-raise the same exception
else:
    # runs ONLY if no exception was raised in try
    print("Success:", result)
finally:
    # ALWAYS runs — exception or not
    cleanup()
```

`else` clause: confirms that the code in `try` succeeded without exception. Keeps the happy path separate from exception handling.

`finally` clause: always runs — even if `return` is in the `try` block or an exception propagates. Use for resource cleanup (though context managers are usually cleaner).

### `raise` and `raise from`

```python
raise ValueError("bad input")          # raise new exception
raise                                  # re-raise current exception (inside except)

try:
    int("abc")
except ValueError as e:
    raise RuntimeError("parsing failed") from e   # chain: RuntimeError.__cause__ = e
    # traceback shows both exceptions
```

`raise X from None` suppresses the original exception context.

### Custom Exception Classes

```python
class AppError(Exception):
    pass

class ValidationError(AppError):
    def __init__(self, field, message):
        super().__init__(f"{field}: {message}")
        self.field = field

try:
    raise ValidationError("email", "invalid format")
except ValidationError as e:
    print(e.field)     # "email"
    print(str(e))      # "email: invalid format"
except AppError:
    pass   # catches all AppError subclasses
```

### Bare `except` — Never Use

```python
# WRONG — catches SystemExit, KeyboardInterrupt, and everything else
try:
    something()
except:
    pass

# CORRECT — catch only what you can handle
try:
    something()
except Exception:
    pass

# BEST — catch the specific exception
try:
    something()
except ValueError:
    handle_value_error()
```

---

## 14. Modules & Packages

### `import` Mechanics

When you `import foo`:

1. Python checks `sys.modules` — if `"foo"` is already there, return it (no re-execution).
2. Searches `sys.path` directories in order.
3. Compiles to bytecode (`.pyc` in `__pycache__`), executes module body, stores in `sys.modules["foo"]`.

```python
import sys
print(sys.path)         # list of directories searched for modules
print(sys.modules)      # dict of cached modules

import math
sys.modules["math"]     # same object as `math`

import importlib
importlib.reload(math)  # force re-execution (rarely needed)
```

### Relative vs Absolute Imports

```
mypackage/
    __init__.py
    utils.py
    models/
        __init__.py
        user.py
        order.py
```

```python
# absolute import — always works
from mypackage import utils
from mypackage.models import user

# relative import — only inside a package
# in mypackage/models/order.py:
from . import user           # same package
from ..utils import helper   # parent package
```

Absolute imports are preferred (PEP 8). Use relative imports inside packages when refactoring structure and you want to keep the imports independent of the package's install name.

### `__init__.py` and Package Structure

`__init__.py` marks a directory as a Python package. It runs when the package is imported. Use it to expose a public API:

```python
# mypackage/__init__.py
from .models.user import User
from .models.order import Order
from .utils import helper

# consumers can now do:
from mypackage import User   # instead of mypackage.models.user.User
```

Namespace packages (Python 3.3+): directories without `__init__.py` are still importable as namespace packages.

### `__all__`

Controls what `from module import *` exports. Explicit is always better — `__all__` also serves as documentation of the public API:

```python
# utils.py
__all__ = ["public_func", "PublicClass"]

def public_func():
    pass

def _private_func():   # leading underscore = private by convention
    pass

class PublicClass:
    pass

class _InternalClass:
    pass
```

Without `__all__`, `import *` imports all names not starting with `_`.

---

## 15. Concurrency & Parallelism

### GIL — Global Interpreter Lock

The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. Reasons for its existence:

1. CPython's reference counting is not thread-safe without the GIL.
2. Many C extensions (numpy, SQLite) assume single-threaded execution.

What the GIL blocks: CPU-bound Python threads cannot run in true parallel.

What the GIL does NOT block: I/O operations release the GIL while waiting, allowing other threads to run. C extensions can release the GIL explicitly (numpy does this for most operations).

### threading

```python
import threading

def worker(n):
    print(f"Thread {n} running")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()   # wait for all threads to finish

# Thread-safe data sharing
lock = threading.Lock()
counter = 0

def increment():
    global counter
    with lock:
        counter += 1

# thread-safe queue for producer-consumer
from queue import Queue
q = Queue()
q.put("item")
item = q.get()
```

### multiprocessing

Spawns separate OS processes — each has its own GIL and memory space. True parallelism for CPU-bound work.

```python
import multiprocessing

def square(x):
    return x ** 2

if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(square, range(20))

    # Process class for finer control
    p = multiprocessing.Process(target=worker, args=(arg,))
    p.start()
    p.join()

    # Shared memory
    shared_val = multiprocessing.Value("i", 0)   # int
    shared_arr = multiprocessing.Array("d", [1.0, 2.0, 3.0])  # doubles
```

`Pool.map` blocks until all tasks complete. `Pool.imap` returns results lazily. `Pool.starmap` for functions with multiple arguments.

### asyncio

Single-threaded cooperative concurrency. A coroutine runs until it `await`s something, then the event loop runs another coroutine. No OS thread switching overhead — suitable for thousands of concurrent I/O operations.

```python
import asyncio
import aiohttp   # pip install aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ["https://httpbin.org/get"] * 10
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # run concurrently
    return results

asyncio.run(main())   # entry point — creates and runs the event loop
```

`asyncio.gather` runs coroutines concurrently and collects results. `asyncio.create_task` schedules a coroutine without waiting. `asyncio.wait` for finer control over completion handling.

```python
async def producer(queue):
    for i in range(5):
        await queue.put(i)
        await asyncio.sleep(0.1)

async def consumer(queue):
    while True:
        item = await queue.get()
        print(item)
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(producer(queue), consumer(queue))
```

### Decision Matrix

| Scenario | Use |
|----------|-----|
| CPU-bound, pure Python | `multiprocessing` |
| CPU-bound, numpy/C extensions | `multiprocessing` or release GIL in C extension |
| I/O-bound, many connections | `asyncio` |
| I/O-bound, legacy blocking code | `threading` |
| Mixed CPU + I/O | `asyncio` + `ProcessPoolExecutor` for CPU parts |
| Simple parallelism | `concurrent.futures.ProcessPoolExecutor` |
| Simple threading | `concurrent.futures.ThreadPoolExecutor` |

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# thread pool
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(io_task, url) for url in urls]
    results = [f.result() for f in futures]

# process pool
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_task, data))
```

---

## 16. Python Internals

### CPython Reference Counting

Every Python object has a `ob_refcnt` field. The count increments on: assignment, passing to a function, inserting into a container. Decrements on: variable goes out of scope, variable reassigned, deleted, or container element removed. When `ob_refcnt` reaches zero, `tp_dealloc` is called immediately.

```python
import sys

x = [1, 2, 3]
sys.getrefcount(x)   # 2 (x + the getrefcount arg)

y = x
sys.getrefcount(x)   # 3

del y
sys.getrefcount(x)   # 2

def f(obj):
    print(sys.getrefcount(obj))   # +1 for the function argument

f(x)   # 3
```

`getrefcount` always shows one more than expected because the call itself creates a reference.

### Cyclic GC

Reference counting cannot collect cycles. Example: `a.ref = b; b.ref = a; del a; del b` — both objects have count 1 but are unreachable.

CPython's `gc` module runs a generational collector:

```
Generation 0 (youngest): threshold 700 new objects
Generation 1: threshold 10 Gen0 collections
Generation 2 (oldest): threshold 10 Gen1 collections
```

Algorithm: temporarily decrement reference counts following all references from a candidate set. Objects whose count reaches 0 have no external references — they form a cycle. Objects with count > 0 are reachable from outside the candidate set.

```python
import gc

gc.collect()           # force collection, return number of unreachable objects
gc.get_count()         # (gen0, gen1, gen2) counts
gc.get_threshold()     # (700, 10, 10)
gc.set_threshold(1000, 15, 15)
gc.disable()           # disable automatic collection (manual only)
```

Objects that implement `__del__` (finalizers) complicate collection — CPython 3.4+ handles them correctly via `tp_finalize`.

### Memory Management: pymalloc

CPython's object allocator (`pymalloc`) manages objects up to 512 bytes:

```
Arena (256 KB) — requested from OS via malloc
└── Pool (4 KB) — holds objects of one size class
    └── Block (8–512 bytes in 8-byte steps) — one object
```

Objects larger than 512 bytes go directly to the OS via `malloc`. Freed small objects return to their pool — pools are not returned to the OS until the arena is entirely free (rare for long-running processes — this is why Python processes seem to hold memory even after deleting objects).

### Bytecode: `dis` Module

CPython compiles source to bytecode before execution. Bytecode is a sequence of 2-byte instructions (opcode + argument):

```python
import dis

def add(a, b):
    return a + b

dis.dis(add)
#   2           0 RESUME                   0
#
#   3           2 LOAD_FAST                0 (a)
#               4 LOAD_FAST                1 (b)
#               6 BINARY_OP               0 (+)
#              10 RETURN_VALUE

# inspect code object
add.__code__.co_varnames    # ('a', 'b')
add.__code__.co_consts      # (None,)
add.__code__.co_filename    # '<stdin>'
add.__code__.co_argcount    # 2
```

`LOAD_FAST` is faster than `LOAD_GLOBAL` (locals stored in a C array, globals require a dict lookup).

### Frame Objects and Call Stack

Each function call creates a frame object (`PyFrameObject`). The frame holds: local variables, reference to the code object, reference to globals dict, the evaluation stack, and the current instruction pointer.

```python
import sys

def inner():
    frame = sys._getframe()       # current frame
    print(frame.f_code.co_name)  # "inner"
    print(frame.f_back.f_code.co_name)  # calling function's name

def outer():
    inner()

outer()
```

### GIL Implementation

The GIL is implemented as a mutex around the eval loop. Python 3.2+ uses a request-based mechanism: every 5ms (the check interval), the holding thread checks if another thread wants the GIL and may release it.

```python
import sys

sys.getswitchinterval()    # 0.005 (5 ms) — default
sys.setswitchinterval(0.001)   # switch more often (more overhead)
```

The GIL is released during:
- Blocking I/O syscalls (`read`, `write`, `recv`, `send`)
- `time.sleep`
- Most numpy operations
- Any C extension that explicitly releases it with `Py_BEGIN_ALLOW_THREADS`

---

## 17. Type Hints & mypy

### Basic Annotations

```python
x: int = 10
name: str = "Alice"
pi: float = 3.14

def greet(name: str) -> str:
    return f"Hello, {name}"

def nothing() -> None:
    pass
```

Type hints are not enforced at runtime — they are metadata for static analyzers.

### Optional, Union, and Collection Types

```python
from typing import Optional, Union, List, Dict, Tuple, Set, FrozenSet

def find(items: List[int], target: int) -> Optional[int]:
    for i, item in enumerate(items):
        if item == target:
            return i
    return None

def process(value: Union[int, str]) -> str:
    return str(value)

# Python 3.10+ — use | instead of Union
def process(value: int | str) -> str:
    return str(value)

# Python 3.9+ — use built-in types directly
def f(x: list[int]) -> dict[str, int]:
    pass
```

### `Any`, `TypeVar`, `Generic`

```python
from typing import Any, TypeVar, Generic

def log(x: Any) -> None:   # opt out of type checking for x
    print(x)

T = TypeVar("T")

def first(lst: list[T]) -> T:   # generic function — preserves type
    return lst[0]

first([1, 2, 3])     # inferred return type: int
first(["a", "b"])    # inferred return type: str

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

s: Stack[int] = Stack()
```

### `Protocol` for Structural Subtyping

`Protocol` enables duck-typing with type checking — any class that implements the required methods satisfies the protocol, without explicit inheritance:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("drawing circle")

class Square:
    def draw(self) -> None:
        print("drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

render(Circle())   # mypy: OK — Circle satisfies Drawable
render(Square())   # mypy: OK
```

### TypedDict and NamedTuple

```python
from typing import TypedDict, NamedTuple

class User(TypedDict):
    name: str
    age: int
    email: str   # all keys required by default

class PartialUser(TypedDict, total=False):
    name: str    # all keys optional
    age: int

user: User = {"name": "Alice", "age": 30, "email": "a@example.com"}

class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0   # default value

p = Point(1.0, 2.0)
p.x       # 1.0
p[0]      # 1.0 — still a tuple
```

### `Final`, `ClassVar`, `Literal`

```python
from typing import Final, ClassVar, Literal

MAX_SIZE: Final = 100          # cannot be reassigned
MAX_SIZE = 200                 # mypy error

class Config:
    count: ClassVar[int] = 0   # class variable, not instance variable

Direction = Literal["north", "south", "east", "west"]

def move(direction: Direction) -> None:
    pass

move("north")    # OK
move("up")       # mypy error
```

### Running mypy

```bash
pip install mypy
mypy script.py                    # check one file
mypy mypackage/                   # check a package
mypy --strict script.py           # maximum strictness
mypy --ignore-missing-imports script.py
```

Create `mypy.ini` or `pyproject.toml` for project-wide config:

```ini
[mypy]
python_version = 3.11
strict = True
ignore_missing_imports = True
```

---

## 18. Testing

### unittest Basics

```python
import unittest

class TestMath(unittest.TestCase):
    def setUp(self):          # runs before each test
        self.data = [1, 2, 3]

    def tearDown(self):       # runs after each test
        pass

    def test_sum(self):
        self.assertEqual(sum(self.data), 6)

    def test_empty(self):
        self.assertEqual(sum([]), 0)

    def test_raises(self):
        with self.assertRaises(TypeError):
            sum("abc")

    def test_approx(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=10)

if __name__ == "__main__":
    unittest.main()
```

```bash
python3 -m unittest test_module.py
python3 -m unittest discover -s tests/   # discover all test_*.py files
```

### pytest

pytest is the industry standard. Tests are plain functions — no class required.

```python
# test_math.py

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
```

```bash
pip install pytest
pytest                    # discover and run all test_*.py files
pytest test_math.py       # specific file
pytest test_math.py::test_add  # specific test
pytest -v                 # verbose output
pytest -x                 # stop on first failure
pytest -k "add"           # run tests matching name pattern
```

**Fixtures:**

```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

@pytest.fixture
def db_connection():
    conn = create_db_connection()
    yield conn          # setup above yield, teardown below
    conn.close()

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_db(db_connection):
    assert db_connection.is_connected()
```

**Parametrize:**

```python
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
])
def test_add(a, b, expected):
    assert a + b == expected
```

**Marks:**

```python
@pytest.mark.slow
def test_expensive():
    pass

@pytest.mark.skip(reason="not implemented yet")
def test_future():
    pass

@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_unix():
    pass
```

```bash
pytest -m slow          # run only slow tests
pytest -m "not slow"    # skip slow tests
```

### `mock.patch` and `MagicMock`

```python
from unittest.mock import patch, MagicMock, call

def get_user(user_id):
    response = requests.get(f"/api/users/{user_id}")
    return response.json()

def test_get_user():
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "name": "Alice"}

    with patch("requests.get", return_value=mock_response) as mock_get:
        user = get_user(1)

    mock_get.assert_called_once_with("/api/users/1")
    assert user["name"] == "Alice"

# patch as decorator
@patch("module.requests.get")
def test_get_user_v2(mock_get):
    mock_get.return_value.json.return_value = {"id": 1}
    result = get_user(1)
    assert result["id"] == 1

# MagicMock — auto-creates attributes on access
m = MagicMock()
m.foo.bar.baz()          # no AttributeError
m.foo.return_value = 42
m.foo()                  # 42
m.assert_called_once()
```

### Coverage with pytest-cov

```bash
pip install pytest-cov
pytest --cov=mypackage tests/
pytest --cov=mypackage --cov-report=html tests/   # generates htmlcov/
pytest --cov=mypackage --cov-fail-under=80        # fail if < 80%
```

Create `.coveragerc`:

```ini
[run]
omit =
    tests/*
    setup.py
    */__init__.py
```

---

## 19. Interview Q&A — 20 Questions

**Q1. What is the GIL, and how does it affect Python programs?**

The GIL (Global Interpreter Lock) is a mutex in CPython that prevents multiple threads from executing Python bytecode simultaneously. It simplifies CPython's memory management (reference counting) at the cost of true CPU parallelism. For CPU-bound work, use `multiprocessing`. For I/O-bound work, `threading` or `asyncio` work fine — the GIL is released during I/O waits.

Follow-up: *Which operations release the GIL?* — Blocking syscalls (read/write/sleep), most numpy operations, any C extension using `Py_BEGIN_ALLOW_THREADS`.

---

**Q2. Explain Python's memory management.**

CPython uses three mechanisms: (1) reference counting — immediate deallocation when count hits zero; (2) cyclic garbage collector — finds and collects reference cycles; (3) pymalloc — custom allocator for small objects (<= 512 bytes) using arenas/pools/blocks to reduce fragmentation.

Follow-up: *Why does Python sometimes not release memory to the OS?* — Pools are returned to the OS only when the entire arena is free, which is rare for long-lived processes.

---

**Q3. What is the difference between `is` and `==`?**

`is` checks object identity (same memory address, `id(a) == id(b)`). `==` calls `__eq__` and checks value equality. Two different list objects `[1, 2]` are `==` but not `is`. Integer caching (-5 to 256) makes small integers `is`-identical in CPython, but this is an implementation detail.

```python
a = [1, 2]
b = [1, 2]
a == b   # True
a is b   # False

a = 100; b = 100
a is b   # True (cached integer)
```

Follow-up: *When should you use `is`?* — Only for `None`, `True`, `False`, or sentinel values.

---

**Q4. What are decorators and how do they work?**

A decorator is a callable that takes a function and returns a new function. `@decorator` is syntactic sugar for `func = decorator(func)`. Decorators are used for cross-cutting concerns: logging, authentication, caching, rate limiting. Always use `functools.wraps` to preserve the original function's metadata.

```python
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.4f}s")
        return result
    return wrapper
```

Follow-up: *How do you write a decorator that takes arguments?* — Add an outer function: `def repeat(n): def decorator(func): ...`

---

**Q5. What is the mutable default argument trap?**

Default argument values are evaluated once at function definition time and stored in `func.__defaults__`. A mutable default (list, dict) persists across calls and accumulates state.

```python
def broken(lst=[]):
    lst.append(1)
    return lst

broken()   # [1]
broken()   # [1, 1] — same list

def fixed(lst=None):
    if lst is None:
        lst = []
    lst.append(1)
    return lst
```

Follow-up: *How can you inspect default values?* — `func.__defaults__` is a tuple of default values.

---

**Q6. How does Python's `for` loop work internally?**

Python calls `iter(obj)` to get an iterator, then calls `next()` on it repeatedly until `StopIteration` is raised. Any object with `__iter__` is iterable. Any object with both `__iter__` and `__next__` is an iterator.

```python
# for x in lst: body
_iter = iter(lst)
while True:
    try:
        x = next(_iter)
    except StopIteration:
        break
    body
```

Follow-up: *What is the difference between an iterable and an iterator?* — An iterable has `__iter__`. An iterator has both `__iter__` (returns self) and `__next__`.

---

**Q7. Explain generators and their advantages over lists.**

Generators produce values lazily — one at a time — using `yield`. They preserve their execution state between calls. Advantages: O(1) memory regardless of sequence length; useful for infinite sequences; composable via `yield from`.

```python
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

# processes one line at a time — no matter how large the file
for line in read_large_file("huge.log"):
    process(line)
```

Follow-up: *What is `send()` used for?* — To pass a value back into the generator at the `yield` expression point, enabling two-way communication.

---

**Q8. What is MRO and how does Python resolve multiple inheritance?**

MRO (Method Resolution Order) is the order in which Python searches base classes for a method. Python uses C3 linearization: `L[C] = C + merge(L[B1], L[B2], [B1, B2])`. The merge algorithm takes the head of each list if it doesn't appear in the tail of any other list.

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

D.__mro__
# (D, B, C, A, object)
```

`super()` follows the MRO — it calls the next class in the MRO chain, not the direct parent.

Follow-up: *What happens if you have a diamond inheritance conflict?* — C3 raises a `TypeError` if no consistent linearization exists.

---

**Q9. What are `*args` and `**kwargs`, and how are they used?**

`*args` collects extra positional arguments as a tuple. `**kwargs` collects extra keyword arguments as a dict. They allow functions to accept arbitrary arguments. The names `args` and `kwargs` are conventions — what matters is `*` and `**`.

```python
def f(*args, **kwargs):
    print(args)    # tuple
    print(kwargs)  # dict

# also used to unpack:
def add(a, b, c): return a + b + c
args = [1, 2, 3]
add(*args)         # add(1, 2, 3)
d = {"a": 1, "b": 2, "c": 3}
add(**d)           # add(a=1, b=2, c=3)
```

Follow-up: *What is the order of parameters in a function signature?* — `positional, *args, keyword-only, **kwargs`.

---

**Q10. What are context managers and when would you write a custom one?**

A context manager is an object with `__enter__` and `__exit__`. It guarantees cleanup code runs even if an exception occurs. Write custom ones for: managing resources (connections, locks, files), timing code sections, temporarily modifying state, or suppressing specific exceptions.

```python
@contextmanager
def temp_directory():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)
```

Follow-up: *How does `__exit__` suppress exceptions?* — Return a truthy value; Python checks the return value and skips re-raising if truthy.

---

**Q11. What is the difference between `deepcopy` and `copy`?**

`copy.copy()` creates a shallow copy — a new container but the same element references. `copy.deepcopy()` recursively copies all objects. Modifying a mutable element of a shallow copy affects the original.

```python
import copy

original = [[1, 2], [3, 4]]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

shallow[0].append(99)
print(original)   # [[1, 2, 99], [3, 4]] — affected

deep[0].append(99)
print(original)   # [[1, 2, 99], [3, 4]] — unaffected by deep copy
```

Follow-up: *When is `deepcopy` a bad idea?* — Objects with circular references, open files, sockets, or database connections — deepcopy will fail or produce nonsensical results.

---

**Q12. Explain list slicing and the `slice` object.**

`lst[start:stop:step]` returns a new list. Omitted values default to start=0, stop=len, step=1. Negative step reverses direction. The slice creates a shallow copy.

```python
lst = [0, 1, 2, 3, 4, 5]
lst[2:5]     # [2, 3, 4]
lst[::2]     # [0, 2, 4]
lst[::-1]    # [5, 4, 3, 2, 1, 0]
lst[1:5:2]   # [1, 3]

# slice object
s = slice(1, 5, 2)
lst[s]       # [1, 3]
```

Follow-up: *How do you implement `__getitem__` for slices?* — Check `isinstance(key, slice)` and use `key.indices(len(self))`.

---

**Q13. What is the difference between `staticmethod` and `classmethod`?**

`staticmethod` receives no implicit first argument — it's a regular function namespaced in the class. `classmethod` receives the class as the first argument (`cls`), allowing it to create instances of the class or access class-level state.

```python
class Date:
    def __init__(self, year, month, day):
        self.year, self.month, self.day = year, month, day

    @classmethod
    def from_string(cls, s):   # alternative constructor
        y, m, d = map(int, s.split("-"))
        return cls(y, m, d)   # works correctly for subclasses too

    @staticmethod
    def is_valid(year, month, day):   # utility — no state needed
        return 1 <= month <= 12 and 1 <= day <= 31
```

Follow-up: *Why use `cls` instead of hardcoding the class name in `classmethod`?* — Subclasses that call the classmethod will create instances of themselves, not the parent.

---

**Q14. How do Python closures work, and what is the late-binding gotcha?**

A closure captures variables from the enclosing scope by reference (via a cell object), not by value at definition time. This means the variable's value at call time is used, not at definition time.

```python
def make_multipliers():
    return [lambda x: x * i for i in range(5)]

fns = make_multipliers()
[f(1) for f in fns]   # [4, 4, 4, 4, 4] — all see i=4 at call time

# fix: bind value at definition
def make_multipliers():
    return [lambda x, i=i: x * i for i in range(5)]

[f(1) for f in make_multipliers()]   # [0, 1, 2, 3, 4]
```

Follow-up: *How do you inspect what a closure has captured?* — `func.__closure__` is a tuple of cell objects; `cell.cell_contents` gives the value.

---

**Q15. What are Python's built-in sorting capabilities?**

`list.sort()` — in-place, returns `None`, O(n log n) Timsort (stable). `sorted()` — returns new list, works on any iterable. Both accept `key=` (transform before comparison) and `reverse=True`.

```python
words = ["banana", "apple", "cherry"]
words.sort(key=len)              # sort by length
sorted(words, key=str.lower)     # case-insensitive

# sort by multiple keys (primary, secondary)
data = [("Alice", 30), ("Bob", 25), ("Carol", 30)]
data.sort(key=lambda x: (x[1], x[0]))   # by age, then name

import operator
data.sort(key=operator.itemgetter(1, 0))  # same, no lambda
```

Follow-up: *What is Timsort?* — A hybrid of merge sort and insertion sort, optimized for partially sorted data. O(n) best case, O(n log n) worst case, O(n) space.

---

**Q16. Explain `__hash__` and why mutable objects are unhashable.**

The hash of an object must be constant over its lifetime. If `a == b`, then `hash(a)` must equal `hash(b)`. Mutable objects can change value, which would change their hash, breaking dict/set invariants. Python sets `__hash__ = None` when you define `__eq__` without `__hash__`, making the object unhashable.

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))   # consistent with __eq__

p = Point(1, 2)
{p: "value"}   # works because __hash__ is defined
```

Follow-up: *How does Python handle hash collisions in dicts?* — Open addressing with pseudo-random probing based on the hash value.

---

**Q17. What is `asyncio` and when should you use it?**

`asyncio` provides single-threaded cooperative concurrency. Coroutines (`async def`) yield control at `await` points. The event loop schedules and drives coroutines. Use for: many concurrent network connections, WebSocket servers, HTTP clients, any I/O-heavy code where you need thousands of concurrent operations without the overhead of OS threads.

```python
import asyncio

async def fetch(url):
    # simulate I/O
    await asyncio.sleep(1)
    return f"data from {url}"

async def main():
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
    # all three run concurrently — total time ~1s, not 3s
    print(results)

asyncio.run(main())
```

Follow-up: *Can asyncio use multiple CPU cores?* — No, it's single-threaded. For CPU + I/O, use `loop.run_in_executor(ProcessPoolExecutor(), cpu_task)`.

---

**Q18. How do you profile Python code?**

```python
# cProfile — function-level timing
python3 -m cProfile -s cumulative script.py

# in code
import cProfile
cProfile.run("my_function()", sort="cumulative")

# line_profiler — line-by-line (pip install line_profiler)
# @profile decorator, then: kernprof -l -v script.py

# memory_profiler — memory usage (pip install memory_profiler)
# @profile decorator, then: python3 -m memory_profiler script.py

# timeit — microbenchmarks
import timeit
timeit.timeit("sum(range(1000))", number=10000)

# in IPython/Jupyter
%timeit sum(range(1000))
%memit sum(range(1000))
```

Follow-up: *What is the difference between wall time and CPU time in profiling?* — Wall time includes I/O wait; CPU time is only time spent executing instructions.

---

**Q19. What are descriptors in Python?**

A descriptor is a class that defines `__get__`, `__set__`, or `__delete__`. When accessed via a class or instance, Python calls these methods instead of returning/setting the attribute directly. `property`, `classmethod`, and `staticmethod` are all descriptors.

```python
class Validated:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self   # class-level access
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise TypeError(f"{self.name} must be int")
        obj.__dict__[self.name] = value

class MyClass:
    age = Validated()

m = MyClass()
m.age = 25    # calls Validated.__set__
m.age         # calls Validated.__get__
m.age = "x"  # TypeError
```

Follow-up: *What is the difference between a data descriptor and a non-data descriptor?* — Data descriptors define `__set__` or `__delete__` and take precedence over instance `__dict__`. Non-data descriptors (only `__get__`) can be shadowed by instance attributes.

---

**Q20. How does Python import work, and how do you avoid circular imports?**

`import foo` triggers: check `sys.modules` cache → find in `sys.path` → execute module body → store in `sys.modules`. Circular imports occur when `a.py` imports `b.py` which imports `a.py`. By the time `b` tries to import `a`, `a` is in `sys.modules` but not fully initialized.

Fixes:
1. Move the import inside the function that needs it (deferred import).
2. Restructure: move shared code to a third module that neither `a` nor `b` imports.
3. Import the module, not the name: `import a` instead of `from a import X` — the attribute is resolved at call time, not import time.

```python
# a.py — deferred import
def get_b_thing():
    from b import thing   # imported when function is called, not at module load
    return thing()
```

Follow-up: *What is `importlib.import_module` used for?* — Dynamic imports where the module name is determined at runtime: `importlib.import_module(module_name)`.

---

## 20. Solved Practice Problems

### Problem 1: Retry Decorator

Implement a decorator that retries a function N times on exception.

**Approach:** Outer function takes `times` and optional `exceptions` to catch. Inner decorator wraps the function in a loop.

```python
import functools
import time

def retry(times=3, exceptions=(Exception,), delay=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times and delay:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

# usage
@retry(times=3, exceptions=(ConnectionError,), delay=1)
def fetch_data(url):
    import random
    if random.random() < 0.7:
        raise ConnectionError("network down")
    return "data"

if __name__ == "__main__":
    try:
        print(fetch_data("http://example.com"))
    except ConnectionError as e:
        print(f"Failed after 3 attempts: {e}")
```

**Complexity:** O(1) overhead per call, O(N) worst case calls.

---

### Problem 2: Infinite Fibonacci Generator

Write a generator that produces Fibonacci numbers infinitely.

**Approach:** Maintain two variables `a`, `b`. Yield `a`, then update simultaneously.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

import itertools

def fib_up_to(limit):
    return list(itertools.takewhile(lambda x: x <= limit, fibonacci()))

def fib_first_n(n):
    return list(itertools.islice(fibonacci(), n))

if __name__ == "__main__":
    print(fib_first_n(10))       # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    print(fib_up_to(100))        # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
```

**Complexity:** O(1) space per value, O(n) time for first n values.

---

### Problem 3: LRU Cache from Scratch

Implement LRU Cache without `functools.lru_cache`.

**Approach:** Doubly linked list (O(1) insert/delete from any position) + dict (O(1) lookup). On access, move node to front. On capacity overflow, evict from rear.

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)  # dummy head (most recent)
        self.tail = Node(0, 0)  # dummy tail (least recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert_front(node)
        return node.value

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self._insert_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

if __name__ == "__main__":
    cache = LRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    print(cache.get(1))   # "a" — moves 1 to front
    cache.put(4, "d")     # evicts 2 (LRU)
    print(cache.get(2))   # -1 — evicted
```

**Complexity:** O(1) for both `get` and `put`.

---

### Problem 4: Flatten Nested List

Flatten a nested list of arbitrary depth using recursion.

**Approach:** Recurse on each element; yield if not a list, else yield from recursed sub-list.

```python
def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def flatten_to_list(lst):
    return list(flatten(lst))

if __name__ == "__main__":
    nested = [1, [2, [3, [4, [5]]], 6], 7, [8, 9]]
    print(flatten_to_list(nested))
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]

    deep = [[[[[42]]]]]
    print(flatten_to_list(deep))   # [42]

    empty = [[], [[], []], []]
    print(flatten_to_list(empty))  # []
```

**Complexity:** O(n) time where n is total number of elements. O(d) stack space where d is maximum nesting depth.

---

### Problem 5: Word Frequency Count

Count word frequency with Counter, then without.

```python
from collections import Counter
import re

def word_frequency_counter(text):
    words = re.findall(r"[a-z]+", text.lower())
    return Counter(words)

def word_frequency_manual(text):
    words = re.findall(r"[a-z]+", text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    text = "the quick brown fox jumps over the lazy dog the fox"

    c = word_frequency_counter(text)
    print(c.most_common(3))   # [('the', 3), ('fox', 2), ...]

    m = word_frequency_manual(text)
    print(list(m.items())[:3])
```

**Complexity:** O(n) time and space where n is word count.

---

### Problem 6: Publish-Subscribe Event System

Implement a simple pub-sub event system using closures.

**Approach:** Central dict maps event names to lists of handler functions. `subscribe` appends a handler; `publish` calls all handlers for the event.

```python
def make_event_system():
    handlers = {}

    def subscribe(event, handler):
        if event not in handlers:
            handlers[event] = []
        handlers[event].append(handler)

        def unsubscribe():
            handlers[event].remove(handler)

        return unsubscribe

    def publish(event, *args, **kwargs):
        for handler in handlers.get(event, [])[:]:  # copy — handler may unsubscribe
            handler(*args, **kwargs)

    def clear(event=None):
        if event:
            handlers.pop(event, None)
        else:
            handlers.clear()

    return subscribe, publish, clear

if __name__ == "__main__":
    subscribe, publish, clear = make_event_system()

    def on_login(user):
        print(f"User logged in: {user}")

    def on_login_audit(user):
        print(f"Audit: login event for {user}")

    unsub = subscribe("login", on_login)
    subscribe("login", on_login_audit)

    publish("login", "Alice")
    # User logged in: Alice
    # Audit: login event for Alice

    unsub()   # remove on_login
    publish("login", "Bob")
    # Audit: login event for Bob
```

**Complexity:** O(1) subscribe, O(k) publish where k is number of handlers for the event.

---

### Problem 7: String Permutations

Find all permutations of a string — recursive approach and itertools comparison.

```python
import itertools

def permutations_recursive(s):
    if len(s) <= 1:
        return [s]
    result = []
    for i, ch in enumerate(s):
        rest = s[:i] + s[i+1:]
        for perm in permutations_recursive(rest):
            result.append(ch + perm)
    return result

def permutations_itertools(s):
    return ["".join(p) for p in itertools.permutations(s)]

def permutations_unique(s):
    return list(set(permutations_itertools(s)))

if __name__ == "__main__":
    s = "abc"
    rec = sorted(permutations_recursive(s))
    it  = sorted(permutations_itertools(s))
    print(rec == it)   # True
    print(rec)         # ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

    s2 = "aab"
    print(permutations_unique(s2))   # ['aab', 'aba', 'baa']
```

**Complexity:** O(n! * n) time — there are n! permutations, each of length n. O(n) recursion stack depth.

---

### Problem 8: Binary Search

Binary search on a sorted list — iterative and recursive.

```python
def binary_search_iterative(lst, target):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2   # avoids integer overflow (relevant in C; Python ints unbounded)
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def binary_search_recursive(lst, target, lo=0, hi=None):
    if hi is None:
        hi = len(lst) - 1
    if lo > hi:
        return -1
    mid = lo + (hi - lo) // 2
    if lst[mid] == target:
        return mid
    elif lst[mid] < target:
        return binary_search_recursive(lst, target, mid + 1, hi)
    else:
        return binary_search_recursive(lst, target, lo, mid - 1)

def binary_search_leftmost(lst, target):
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(lst) and lst[lo] == target else -1

import bisect

def binary_search_bisect(lst, target):
    i = bisect.bisect_left(lst, target)
    return i if i < len(lst) and lst[i] == target else -1

if __name__ == "__main__":
    lst = [1, 3, 5, 7, 9, 11, 13, 15]
    print(binary_search_iterative(lst, 7))   # 3
    print(binary_search_iterative(lst, 4))   # -1
    print(binary_search_bisect(lst, 13))     # 6
```

**Complexity:** O(log n) time, O(1) space (iterative), O(log n) space (recursive).

---

### Problem 9: Timing Context Manager

Implement a context manager that times code execution.

```python
import time
from contextlib import contextmanager

class Timer:
    def __init__(self, label=""):
        self.label = label
        self.elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, tb):
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.label}] " if self.label else ""
        print(f"{label}Elapsed: {self.elapsed:.6f}s")
        return False   # do not suppress exceptions

@contextmanager
def timer(label=""):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Elapsed: {elapsed:.6f}s")

if __name__ == "__main__":
    with Timer("sum") as t:
        total = sum(range(10_000_000))
    print(f"Result: {total}, took {t.elapsed:.4f}s")

    with timer("list comprehension"):
        result = [x**2 for x in range(1_000_000)]
```

**Complexity:** O(1) overhead for the context manager itself.

---

### Problem 10: Parse CSV Manually

Parse a CSV file manually (no csv module) and compute column averages.

```python
def parse_csv(filepath, delimiter=","):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            fields = parse_line(line, delimiter)
            rows.append(fields)
    return rows

def parse_line(line, delimiter=","):
    fields = []
    current = []
    in_quotes = False

    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == delimiter and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    fields.append("".join(current).strip())
    return fields

def column_averages(filepath, delimiter=","):
    rows = parse_csv(filepath, delimiter)
    if not rows:
        return {}

    headers = rows[0]
    data_rows = rows[1:]

    totals = {h: 0.0 for h in headers}
    counts = {h: 0 for h in headers}

    for row in data_rows:
        for header, value in zip(headers, row):
            try:
                totals[header] += float(value)
                counts[header] += 1
            except ValueError:
                pass   # skip non-numeric values

    return {
        h: totals[h] / counts[h] if counts[h] > 0 else None
        for h in headers
    }

def create_sample_csv(filepath):
    with open(filepath, "w") as f:
        f.write("name,age,score\n")
        f.write("Alice,30,95.5\n")
        f.write("Bob,25,87.0\n")
        f.write("Carol,35,92.3\n")
        f.write("Dave,28,78.9\n")

if __name__ == "__main__":
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = tmp.name

    create_sample_csv(path)

    rows = parse_csv(path)
    print("Headers:", rows[0])
    print("First row:", rows[1])

    avgs = column_averages(path)
    for col, avg in avgs.items():
        if avg is not None:
            print(f"{col}: {avg:.2f}")
        else:
            print(f"{col}: (non-numeric)")

    os.unlink(path)
```

**Complexity:** O(n * m) time where n is rows and m is columns. O(n * m) space to store all rows. Handles quoted fields containing commas or newlines within quotes.

---

## Section 21 — Python for ML & AI: Everything You Need

This section covers all Python knowledge required to work effectively in ML/AI/data science — from numerical foundations through production deployment patterns. Every subsection includes runnable code, not pseudocode.

---

### 21.1 NumPy — The Foundation

NumPy provides the n-dimensional array (`ndarray`) that underpins every ML framework. Understanding it at the memory level prevents shape bugs and performance bottlenecks.

#### Array Creation and Dtypes

```python
import numpy as np

# Creation from Python structures
a = np.array([1, 2, 3])                    # int64 on 64-bit systems
b = np.array([1.0, 2.0, 3.0])             # float64
c = np.array([1, 2, 3], dtype=np.float32) # explicit dtype — matches PyTorch default

# Factory functions
zeros   = np.zeros((3, 4))                 # shape (3,4), dtype float64
ones    = np.ones((2, 3), dtype=np.int8)
eye     = np.eye(4)                        # 4×4 identity
linsp   = np.linspace(0, 1, 100)          # 100 evenly spaced from 0 to 1
arange  = np.arange(0, 10, 2)            # [0,2,4,6,8]
empty   = np.empty((3, 3))               # uninitialized — fast but values garbage

# Dtype inspection and casting
print(a.dtype, a.itemsize, a.nbytes)      # int64  8  24
a32 = a.astype(np.float32)               # explicit cast — copy
```

#### Shape, Reshape, Transpose

```python
rng = np.random.default_rng(42)          # always seed for reproducibility
x = rng.standard_normal((4, 6))         # shape (4, 6)

print(x.shape)   # (4, 6)
print(x.ndim)    # 2
print(x.size)    # 24

# Reshape — total elements must match
x_r = x.reshape(2, 12)      # (2, 12)
x_r = x.reshape(24)         # (24,)  — 1D
x_r = x.reshape(4, -1)      # (4, 6) — infer last dim
x_r = x.reshape(-1, 3)      # (8, 3)

# Transpose
x_T = x.T                   # (6, 4) — view, not copy for C-contiguous
x_T2 = x.transpose(1, 0)   # same

# Adding/removing dimensions
x_3d = x[:, :, np.newaxis]  # (4, 6, 1) — needed for broadcasting
x_3d = x[..., np.newaxis]   # same, ... matches all leading dims
x_sq = x_3d.squeeze()       # (4, 6) — remove all size-1 dims
```

#### Broadcasting Rules

Broadcasting lets NumPy operate on arrays of different shapes without copying data. Two shapes are compatible if, aligned from the right, each dimension pair is either equal or one of them is 1.

```python
# Compatible — broadcast works
a = np.ones((4, 1))   # (4, 1)
b = np.ones((1, 6))   # (1, 6)
c = a + b             # (4, 6) — a broadcasts along cols, b along rows

# Subtract column mean from each column: (N, D) - (D,) → (N, D)
X = rng.standard_normal((100, 10))
X_centered = X - X.mean(axis=0)   # mean shape (10,) broadcasts over 100 rows

# Outer product via broadcasting
u = np.array([1, 2, 3])       # (3,)
v = np.array([10, 20])        # (2,)
outer = u[:, np.newaxis] * v[np.newaxis, :]  # (3, 2)

# FAILS — shapes (3, 4) and (2,) incompatible: 4 != 2 and neither is 1
try:
    fail = np.ones((3, 4)) + np.ones((2,))
except ValueError as e:
    print(e)   # operands could not be broadcast together with shapes (3,4) (2,)
```

| Left shape | Right shape | Result shape | Reason |
|---|---|---|---|
| `(4, 6)` | `(6,)` | `(4, 6)` | `(6,)` → `(1,6)` → `(4,6)` |
| `(4, 1)` | `(1, 6)` | `(4, 6)` | both dims expand |
| `(3,)` | `(3,)` | `(3,)` | equal |
| `(3, 4)` | `(2,)` | **error** | 4≠2, neither is 1 |
| `(5, 1, 3)` | `(4, 3)` | `(5, 4, 3)` | middle dim expands |

#### Vectorized Operations vs Python Loops

```python
import time

N = 1_000_000
rng = np.random.default_rng(42)
x = rng.standard_normal(N)

# Python loop — avoid this
t0 = time.perf_counter()
result_loop = [xi**2 for xi in x]
t_loop = time.perf_counter() - t0

# NumPy vectorized
t0 = time.perf_counter()
result_vec = x**2
t_vec = time.perf_counter() - t0

print(f"Loop: {t_loop:.3f}s   NumPy: {t_vec:.4f}s   Speedup: {t_loop/t_vec:.0f}x")
# typical: Loop: 0.180s   NumPy: 0.003s   Speedup: 60x
```

Why: NumPy calls compiled C/Fortran code (BLAS/LAPACK) and operates on contiguous memory blocks. Python loops have per-iteration interpreter overhead (~100ns/iter) plus object boxing/unboxing for floats.

#### Indexing: Basic, Boolean, Fancy

```python
x = np.arange(12).reshape(3, 4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

# Basic — slices return views (no copy)
print(x[1, :])        # row 1: [4 5 6 7]
print(x[:, 2])        # col 2: [2 6 10]
print(x[0:2, 1:3])   # submatrix (2,2): [[1,2],[5,6]]

# Boolean — always returns copy
mask = x > 5
print(x[mask])        # [6 7 8 9 10 11] — 1D result
x[x % 2 == 0] = -1   # in-place conditional assignment

# Fancy indexing — integer arrays, always returns copy
rows = np.array([0, 2])
cols = np.array([1, 3])
print(x[rows, cols])    # elements (0,1) and (2,3): [1, 11]
print(x[rows])          # rows 0 and 2: shape (2, 4)
print(x[rows[:, None], cols])  # 2×2 submatrix at all combinations
```

#### Linear Algebra

```python
A = rng.standard_normal((4, 4))
b = rng.standard_normal(4)

# Matrix multiply (prefer @ operator)
C = A @ A.T                          # (4,4) — positive semi-definite
dot = np.dot(A, A.T)                 # identical to @

# Solve linear system Ax = b
x = np.linalg.solve(A, b)           # more stable than inv(A) @ b

# Inverse — use only when you need the matrix itself
A_inv = np.linalg.inv(A)

# Eigendecomposition: A = Q diag(λ) Q⁻¹
eigenvalues, eigenvectors = np.linalg.eig(A)       # may be complex for non-symmetric
eigenvalues_s, eigenvectors_s = np.linalg.eigh(C)  # guaranteed real for symmetric

# SVD: A = U S V^T  (shapes: (m,m), (min(m,n),), (n,n))
U, s, Vt = np.linalg.svd(A, full_matrices=True)
U, s, Vt = np.linalg.svd(A, full_matrices=False)   # economy SVD — usually preferred
A_reconstructed = U * s @ Vt        # broadcasting: multiply each col of U by s_i

# Rank, determinant, norms
rank = np.linalg.matrix_rank(A)
det  = np.linalg.det(A)
frob = np.linalg.norm(A, 'fro')     # Frobenius
l2   = np.linalg.norm(b)            # Euclidean for vectors

# PCA via SVD (manual — matches sklearn result)
X = rng.standard_normal((100, 5))
X_c = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_pca = X_c @ Vt[:2].T              # project to top-2 components
```

#### Memory Layout: C vs Fortran Contiguous

```python
A = np.ones((1000, 1000))
print(A.flags['C_CONTIGUOUS'])    # True — row-major (C order)
print(A.flags['F_CONTIGUOUS'])    # False

A_F = np.asfortranarray(A)        # column-major (Fortran order)
print(A_F.flags['F_CONTIGUOUS'])  # True

# Why it matters: BLAS (used by np.dot / @) is optimized for column-major.
# For matrix multiply C = A @ B, if A is F-contiguous NumPy can call BLAS
# without transposing internally. For row iteration, C-contiguous is faster.

# After transpose, x.T is F-contiguous (no data copy)
x = np.ones((3, 4))              # C-contiguous
print(x.T.flags['F_CONTIGUOUS']) # True — same buffer, different strides
print(x.T.flags['C_CONTIGUOUS']) # False

# Force a contiguous copy when passing to C extensions
x_T_contig = np.ascontiguousarray(x.T)  # C-contiguous copy of transposed
```

---

### 21.2 Pandas — Data Wrangling

#### Series and DataFrame

```python
import pandas as pd
import numpy as np

# Series: 1D with index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['b'])       # 20 — label access
print(s.iloc[1])    # 20 — positional access

# DataFrame: 2D heterogeneous table
df = pd.DataFrame({
    'age':    [25, 32, 28, 45],
    'salary': [50000, 80000, 60000, 120000],
    'dept':   ['eng', 'eng', 'mkt', 'eng'],
})
print(df.dtypes)   # age int64, salary int64, dept object
print(df.shape)    # (4, 3)
print(df.info())   # memory usage, non-null counts
```

#### Loading Data

```python
# CSV with dtype specification (avoids object columns for known types)
df = pd.read_csv('data.csv', dtype={'id': np.int32, 'score': np.float32},
                 parse_dates=['timestamp'], index_col='id')

# JSON (records or lines format)
df = pd.read_json('data.jsonl', lines=True)

# Inspect after load — always do this
print(df.head())
print(df.dtypes)
print(df.isna().sum())   # missing count per column
print(df.describe())     # percentiles for numeric cols
```

#### Selection: loc vs iloc vs []

```python
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]}, index=['x','y','z'])

# [] — column selection (label only for columns)
df['A']            # Series for column A
df[['A','B']]      # DataFrame with subset columns

# loc — label-based: rows and columns by label
df.loc['x']                # row with index label 'x'
df.loc['x':'y', 'A':'B']  # label-inclusive slice (both ends included!)
df.loc[df['A'] > 1, 'B']  # boolean row mask + column label

# iloc — position-based: rows and columns by integer position
df.iloc[0]                  # first row
df.iloc[0:2, 0:1]          # positional slice (end exclusive)
df.iloc[[0, 2], :]          # rows 0 and 2
```

**SettingWithCopyWarning**: occurs when you set values on a slice that may be a view.

```python
# WRONG — may silently not update original df
subset = df[df['A'] > 1]
subset['B'] = 99              # SettingWithCopyWarning

# CORRECT — use .loc on the original DataFrame
df.loc[df['A'] > 1, 'B'] = 99

# Or use .copy() explicitly when you want a detached copy
subset = df[df['A'] > 1].copy()
subset['B'] = 99              # no warning — subset is independent
```

#### GroupBy: Split-Apply-Combine

```python
df = pd.DataFrame({
    'dept':   ['eng','eng','mkt','mkt','eng'],
    'salary': [80000, 90000, 60000, 70000, 85000],
    'level':  ['mid','senior','junior','mid','senior'],
})

# agg — compute one or more aggregations per group
df.groupby('dept')['salary'].agg(['mean', 'max', 'count'])

# Multiple columns in groupby
df.groupby(['dept', 'level'])['salary'].mean()

# transform — returns same-length Series aligned to original df
df['dept_avg'] = df.groupby('dept')['salary'].transform('mean')
# useful for computing "salary relative to dept average"
df['rel_salary'] = df['salary'] / df['dept_avg']

# apply — arbitrary function, most flexible, also slowest
def top_earner(group):
    return group.nlargest(1, 'salary')

top = df.groupby('dept').apply(top_earner, include_groups=False)
```

| Method | Output shape | Speed | Use when |
|---|---|---|---|
| `agg` | one row per group | fast | summary statistics |
| `transform` | same as input | fast | broadcast group stats back |
| `apply` | arbitrary | slow | complex per-group logic |

#### Merge and Join

```python
users   = pd.DataFrame({'id': [1,2,3], 'name': ['Ana','Bob','Cal']})
orders  = pd.DataFrame({'user_id': [1,1,2,4], 'amount': [100,200,150,50]})

# inner join — only matching keys
pd.merge(users, orders, left_on='id', right_on='user_id', how='inner')
# rows: 3 (user 3 dropped — no orders; user 4 dropped — no user record)

# left join — keep all left rows
pd.merge(users, orders, left_on='id', right_on='user_id', how='left')
# user 3 appears with NaN amount

# suffixes for colliding column names
a = pd.DataFrame({'key': [1], 'val': [10]})
b = pd.DataFrame({'key': [1], 'val': [20]})
pd.merge(a, b, on='key', suffixes=('_a', '_b'))
# columns: key, val_a, val_b
```

#### Missing Data

```python
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': ['x', 'y', None]})

df.isna()               # boolean mask
df.isna().sum()         # count per column
df.dropna()             # drop rows with ANY NaN
df.dropna(subset=['A']) # drop only if NaN in col A

# Fill strategies
df['A'].fillna(df['A'].mean())       # mean imputation (numeric)
df['A'].fillna(method='ffill')       # forward fill (time series)
df['B'].fillna('unknown')            # category fill (string)
df['A'].fillna(df.groupby('cat')['A'].transform('median'))  # group median
```

#### apply vs Vectorized Operations

```python
# SLOW — apply with Python function, one row at a time
df['result'] = df['salary'].apply(lambda x: x * 1.1 if x > 70000 else x)

# FAST — vectorized with np.where
df['result'] = np.where(df['salary'] > 70000, df['salary'] * 1.1, df['salary'])

# SLOW — string apply
df['upper'] = df['name'].apply(str.upper)

# FAST — str accessor (vectorized string methods)
df['upper'] = df['name'].str.upper()

# Rule: use apply only when no vectorized equivalent exists
```

#### Memory Optimization

```python
df = pd.read_csv('large_file.csv')
print(df.memory_usage(deep=True).sum() / 1e6, 'MB')

# Downcast numeric
df['age']    = pd.to_numeric(df['age'],    downcast='integer')  # int64 → int8/16
df['score']  = pd.to_numeric(df['score'],  downcast='float')    # float64 → float32

# Category type for low-cardinality strings (crucial for groupby performance)
df['dept'] = df['dept'].astype('category')
# Stores as integer codes + mapping dict — often 10-50× smaller

print(df.memory_usage(deep=True).sum() / 1e6, 'MB')  # compare
```

---

### 21.3 Matplotlib & Seaborn — Visualization for ML

#### Figure/Axes Architecture

Always use `fig, ax = plt.subplots()`. The pyplot state machine (`plt.plot()`) becomes unmanageable with multiple subplots and is unsuitable for functions that return figures.

```python
import matplotlib.pyplot as plt
import numpy as np

# Correct pattern
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1, 2, 3], [4, 5, 6], label='series 1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
fig.savefig('loss.png', dpi=150, bbox_inches='tight')
plt.close(fig)   # free memory — critical in loops

# Multiple subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(...)
axes[1].hist(...)
fig.tight_layout()
fig.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

**Never use `plt.show()`** in scripts or production code — it blocks execution and fails in headless environments. Save to file.

#### ML-Specific Plot Recipes

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Loss curve ---
def plot_loss_curves(train_losses, val_losses, path='loss.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train loss')
    ax.plot(epochs, val_losses,   label='Val loss', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- Confusion matrix heatmap ---
def plot_confusion_matrix(cm, class_names, path='cm.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- ROC curve ---
def plot_roc(fpr, tpr, auc_score, path='roc.png'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    ax.plot([0,1], [0,1], 'k--', label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- Feature importance bar ---
def plot_feature_importance(names, importances, path='feat_imp.png'):
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), importances[idx])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in idx], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- Scatter with class labels ---
def plot_scatter_labels(X, y, path='scatter.png'):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], label=f'class {label}', alpha=0.7, s=20)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

#### Seaborn for ML

```python
import seaborn as sns

# Pairplot — quick multivariate EDA for small datasets (< ~10 cols)
fig = sns.pairplot(df, hue='target', diag_kind='kde')
fig.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.close()

# Heatmap — correlation matrix
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax)
fig.savefig('corr.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Boxplot — distribution per class for outlier detection
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='dept', y='salary', ax=ax)
fig.savefig('box.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

| Plot | Best for |
|---|---|
| `pairplot` | Quick EDA on <10 feature dataset |
| `heatmap` | Correlation or confusion matrix |
| `boxplot` | Distribution comparison across categories |
| `violinplot` | Distribution shape (vs just quartiles) |
| `histplot` | Single feature distribution |

---

### 21.4 Scikit-Learn Patterns

#### Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Simple pipeline — fit/transform in one object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)
```

#### ColumnTransformer

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_cols = ['age', 'salary']
cat_cols = ['dept', 'level']

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

full_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model',      RandomForestClassifier(n_estimators=100, random_state=42)),
])

full_pipe.fit(X_train, y_train)
```

#### Cross-Validation and GridSearchCV

```python
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                      GridSearchCV, cross_validate)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Simple CV score
scores = cross_val_score(full_pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Multiple metrics at once
results = cross_validate(full_pipe, X, y, cv=cv,
                          scoring=['accuracy','roc_auc','f1_weighted'],
                          return_train_score=True, n_jobs=-1)

# Hyperparameter search — nested parameter names use __
param_grid = {
    'model__n_estimators':   [100, 300],
    'model__max_depth':      [None, 5, 10],
    'preprocess__num__scale': [StandardScaler()],
}
gs = GridSearchCV(full_pipe, param_grid, cv=cv,
                  scoring='roc_auc', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

#### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """Log1p transform for right-skewed positive features."""

    def __init__(self, offset=1.0):
        self.offset = offset          # store all __init__ args as attributes

    def fit(self, X, y=None):
        # stateless transform — nothing to learn
        return self                   # always return self

    def transform(self, X):
        return np.log1p(X + self.offset - 1)

    # fit_transform is inherited from TransformerMixin — calls fit then transform

class ClipTransformer(BaseEstimator, TransformerMixin):
    """Clip outliers at fitted percentiles."""

    def __init__(self, low=1, high=99):
        self.low  = low
        self.high = high

    def fit(self, X, y=None):
        self.low_  = np.percentile(X, self.low,  axis=0)
        self.high_ = np.percentile(X, self.high, axis=0)
        # convention: fitted attributes end with _ (sklearn standard)
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)
```

**Data leakage rule**: call `fit` or `fit_transform` only on training data. The scaler's mean/std must come from training data only. Pipelines handle this automatically when you call `pipe.fit(X_train, y_train)` — each step sees only transformed training data.

#### Model Persistence

```python
import joblib

# Save
joblib.dump(full_pipe, 'model.joblib')

# Load
pipe_loaded = joblib.load('model.joblib')
preds = pipe_loaded.predict(X_test)

# Why joblib over pickle for sklearn:
# - joblib uses memory-mapped arrays for numpy arrays inside models
# - ~10× faster for large fitted objects (e.g., RandomForest with many trees)
# - pickle works but is slower and uses more memory during load
```

---

### 21.5 PyTorch Essentials for ML

#### Tensor Creation and Device Management

```python
import torch
import numpy as np

# Creation
x = torch.tensor([1.0, 2.0, 3.0])              # from list, infers float32
x = torch.tensor(np.array([1, 2, 3]), dtype=torch.float32)
x = torch.zeros(3, 4)
x = torch.randn(3, 4)                           # N(0,1)
x = torch.arange(10, dtype=torch.float32)

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
x = x.cuda()     # equivalent if GPU available
x = x.cpu()      # back to CPU (needed before numpy conversion)

# NumPy bridge — shares memory if CPU tensor
arr = x.cpu().numpy()           # tensor → numpy (no copy if contiguous)
x2  = torch.from_numpy(arr)    # numpy → tensor (shares memory)
```

#### Autograd

```python
# requires_grad=True — PyTorch tracks all operations on this tensor
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor(6.0)

# Forward pass — builds computation graph
y_pred = (w * x).sum() + b
loss   = (y_pred - y_true)**2

# Backward — compute gradients via chain rule
loss.backward()
print(w.grad)   # d(loss)/d(w)
print(b.grad)   # d(loss)/d(b)

# Gradient accumulation — grads accumulate by default, must zero before next step
w.grad.zero_()  # in-place zero (trailing _ = in-place in PyTorch convention)

# Disable gradient tracking for inference or non-differentiable code
with torch.no_grad():
    y_pred = (w * x).sum() + b   # no graph built — faster, less memory

# torch.no_grad() is equivalent to:
x.detach()   # returns tensor that shares data but is detached from graph
```

#### nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)   # (batch, out_dim)

model = MLP(128, 256, 10).to(device)

# Introspection
print(sum(p.numel() for p in model.parameters()))          # total params
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# State dict — for saving/loading
state = model.state_dict()              # OrderedDict of tensors
model.load_state_dict(state)            # restore
torch.save(state, 'model.pt')
state = torch.load('model.pt', map_location=device)
model.load_state_dict(state)
```

#### Training Loop Template

```python
import torch.optim as optim

model     = MLP(128, 256, 10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()       # expects logits (raw scores)

for epoch in range(num_epochs):
    # --- Train ---
    model.train()    # enables dropout, batchnorm update
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()          # 1. clear gradients
        logits = model(X_batch)        # 2. forward
        loss   = criterion(logits, y_batch)  # 3. compute loss
        loss.backward()                # 4. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
        optimizer.step()               # 5. update weights
        train_loss += loss.item()

    scheduler.step()

    # --- Eval ---
    model.eval()     # disables dropout, uses running stats for batchnorm
    val_loss = 0.0
    correct  = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits   = model(X_batch)
            val_loss += criterion(logits, y_batch).item()
            correct  += (logits.argmax(1) == y_batch).sum().item()

    print(f"Epoch {epoch+1}: train={train_loss/len(train_loader):.4f}  "
          f"val={val_loss/len(val_loader):.4f}  acc={correct/len(val_ds):.3f}")
```

#### Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader, random_split

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)   # long for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = TabularDataset(X_array, y_array)
train_ds, val_ds = random_split(ds, [0.8, 0.2],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                           num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                           num_workers=4, pin_memory=True)
# num_workers > 0: parallel data loading (CPU)
# pin_memory=True: faster CPU→GPU transfer
```

**Common GPU/CPU mismatch bugs:**

```python
# Bug 1: model on GPU, data on CPU
model = model.to('cuda')
for X, y in loader:
    out = model(X)          # RuntimeError: Expected all tensors to be on the same device
    # Fix: X = X.to(device), y = y.to(device)

# Bug 2: loss is on GPU, converting to Python float requires .item()
loss_val = loss.item()      # correct — pulls scalar to CPU
loss_val = float(loss)      # also works but keeps tensor context longer

# Bug 3: numpy conversion without moving to CPU
arr = tensor.cuda().numpy()  # RuntimeError: can't convert CUDA tensor to numpy
arr = tensor.cpu().numpy()   # correct
```

---

### 21.6 HuggingFace Transformers Essentials

#### AutoTokenizer and AutoModel

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModel.from_pretrained(model_name)
model.eval()

texts = ["The cat sat on the mat.", "Deep learning is powerful."]
```

#### Tokenizer Output

```python
encoding = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt',    # 'pt' for PyTorch, 'np' for NumPy, 'tf' for TensorFlow
)

print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

print(encoding['input_ids'].shape)       # (2, 128)
print(encoding['attention_mask'].shape)  # (2, 128) — 1 for real tokens, 0 for padding

# Decode back to inspect
print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0].tolist()[:10]))
# ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '.', '[SEP]', '[PAD]']
```

#### Extracting Embeddings

```python
with torch.no_grad():
    outputs = model(**encoding)

# outputs.last_hidden_state: (batch, seq_len, hidden_size) — per-token embeddings
token_embeddings = outputs.last_hidden_state   # (2, 128, 768) for BERT-base

# [CLS] embedding — used by BERT for classification tasks
cls_embedding = token_embeddings[:, 0, :]      # (2, 768)

# Mean pooling — often better for sentence similarity
# Must mask out padding tokens
mask  = encoding['attention_mask'].unsqueeze(-1).float()   # (2, 128, 1)
summed = (token_embeddings * mask).sum(dim=1)              # (2, 768)
counts = mask.sum(dim=1).clamp(min=1e-9)                   # (2, 1)
mean_pooled = summed / counts                              # (2, 768)
```

#### Pipeline Shortcut

```python
from transformers import pipeline

# Text classification (sentiment)
clf = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
print(clf(["I love this!", "This is terrible."]))

# Text generation
gen = pipeline('text-generation', model='gpt2', max_new_tokens=50)
print(gen("The future of AI is"))

# Feature extraction (embeddings)
feat = pipeline('feature-extraction', model='bert-base-uncased', return_tensors=True)
embeddings = feat("Hello world")   # list of arrays
```

#### Saving and Loading Fine-tuned Models

```python
# After fine-tuning
output_dir = './my-finetuned-bert'
model.save_pretrained(output_dir)       # saves config.json + pytorch_model.bin (or shards)
tokenizer.save_pretrained(output_dir)   # saves tokenizer files

# Load later — identical API
model     = AutoModel.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# from_pretrained vs from_config
# from_pretrained: loads weights + config (for inference or fine-tuning from checkpoint)
# from_config:     loads architecture only, random weights (for training from scratch)
from transformers import AutoConfig
config = AutoConfig.from_pretrained('bert-base-uncased')
config.hidden_size = 512              # modify architecture
model_scratch = AutoModel.from_config(config)   # random weights, modified arch
```

---

### 21.7 Environment & Dependency Management

#### venv vs conda vs uv

```bash
# venv — stdlib, minimal, fast
python3.14 -m venv .venv
source .venv/bin/activate
pip install numpy pandas torch

# conda — manages Python itself + non-Python deps (e.g., CUDA libs)
conda create -n myenv python=3.11
conda activate myenv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# uv — Rust-based, 10-100× faster than pip, drop-in replacement
uv venv .venv
source .venv/bin/activate
uv pip install numpy pandas torch
```

| Tool | Speed | Python version mgmt | Non-Python deps | Lockfile |
|---|---|---|---|---|
| venv+pip | slow | no | no | pip-tools/pip freeze |
| conda | medium | yes | yes (CUDA, MKL) | conda lock |
| uv | very fast | yes (uv python) | no | uv.lock |

#### requirements.txt vs pyproject.toml

```toml
# pyproject.toml — modern standard (PEP 517/518/621)
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "torch>=2.0",
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
```

```
# requirements.txt — still used for deployment, pip install -r
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
torch>=2.0
```

#### CUDA Version Compatibility

PyTorch wheels are compiled against specific CUDA versions. Installing `pip install torch` gets CPU-only or CUDA-unspecified build.

```bash
# Check CUDA version
nvcc --version
nvidia-smi   # shows driver and max CUDA version supported

# Install PyTorch for specific CUDA (get command from https://pytorch.org/get-started)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio  # CPU-only default

# Verify GPU is accessible
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

#### .env Files for API Keys

```bash
# .env  (never commit this file)
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

```python
# In code
from dotenv import load_dotenv
import os

load_dotenv()   # reads .env into environment variables
api_key = os.getenv('OPENAI_API_KEY')
assert api_key, "OPENAI_API_KEY not set"

# Always add .env to .gitignore
```

```
# .gitignore entries
.env
*.pkl
*.pt
*.pth
__pycache__/
.venv/
```

#### Editable Installs

```bash
# Install your own package in editable mode (changes reflected immediately)
pip install -e .
# or with uv:
uv pip install -e .

# Requires pyproject.toml or setup.py at repo root
# Makes `import my_package` work from anywhere in the venv
```

---

### 21.8 Debugging ML Code

#### pdb and breakpoint()

```python
# Python 3.7+: built-in breakpoint() — no import needed
def train_step(model, batch):
    X, y = batch
    breakpoint()     # drops into pdb here
    # pdb commands: n (next), s (step into), c (continue), p expr (print), q (quit)
    # l (list source), w (where/stack trace), pp tensor.shape

# Programmatic: useful in data loading loops
import pdb
for i, batch in enumerate(loader):
    if batch[0].shape[0] != expected_batch_size:
        pdb.set_trace()    # inspect batch when something is wrong
```

#### Shape Debugging Pattern

```python
# During development, print shape after every operation
def forward(self, x):
    print(f"input:   {x.shape}")          # (batch, seq, d_model)
    x = self.attn(x)
    print(f"after attn: {x.shape}")
    x = self.ff(x)
    print(f"after ff:   {x.shape}")
    return x

# Use asserts to encode your understanding — they catch bugs early
def forward(self, x, mask=None):
    B, T, D = x.shape
    assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"
    q = self.W_q(x)           # (B, T, D)
    assert q.shape == (B, T, self.d_model), f"q shape wrong: {q.shape}"
    return q
```

#### Reproducibility Checklist

```python
import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)             # legacy API — still used by some libs
    rng = np.random.default_rng(seed)  # modern API
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDA determinism — slower but fully reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)

set_seed(42)
```

#### Common ML Bugs

```python
# Bug 1: Wrong loss reduction — mean vs sum gives different gradient scales
loss = nn.CrossEntropyLoss(reduction='mean')  # correct default
loss = nn.CrossEntropyLoss(reduction='sum')   # gradients scale with batch size

# Bug 2: Wrong label dtype for CrossEntropyLoss
y = torch.tensor([0, 1, 2], dtype=torch.long)    # correct
y = torch.tensor([0, 1, 2], dtype=torch.float)   # wrong — raises RuntimeError

# Bug 3: Gradient not zeroed — accumulates across batches
optimizer.zero_grad()    # must call before loss.backward()
# Exception: intentional gradient accumulation for large effective batch sizes:
accumulation_steps = 4
for i, (X, y) in enumerate(loader):
    loss = criterion(model(X), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Bug 4: In-place operation breaks autograd
x = torch.randn(3, requires_grad=True)
y = x.relu_()    # in-place relu — modifies x before grad computation, may error
y = x.relu()     # correct — creates new tensor

# Bug 5: model.eval() not called at inference — dropout active, random outputs
model.eval()
with torch.no_grad():
    preds = model(X_test)
```

---

### 21.9 Performance Patterns

#### Profiling

```python
import cProfile
import pstats

# cProfile — function-level timing
profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)    # top 20 functions by cumulative time

# line_profiler — line-by-line (pip install line_profiler)
# Add @profile decorator, then: kernprof -l -v script.py

# PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    with record_function("model_inference"):
        model(X_batch.to(device))

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### Vectorize First

```python
import numpy as np

data = np.random.default_rng(42).standard_normal(1_000_000)

# SLOW — Python loop
result = [x**2 + 2*x + 1 for x in data]

# FAST — vectorized
result = data**2 + 2*data + 1

# SLOW — loop over DataFrame rows
for i, row in df.iterrows():
    df.at[i, 'new_col'] = row['a'] * row['b']

# FAST — vectorized column operation
df['new_col'] = df['a'] * df['b']
```

#### Generators vs List Comprehensions

```python
# List comprehension — builds entire list in memory
squares = [x**2 for x in range(10_000_000)]   # ~80MB

# Generator expression — lazy, O(1) memory
squares = (x**2 for x in range(10_000_000))   # iterator, values on demand

# Use generators for large datasets, pipelines, when you only need one-pass
def stream_batches(file_path, batch_size=64):
    """Yield batches from large file without loading all into memory."""
    batch = []
    with open(file_path) as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

#### functools.lru_cache

```python
from functools import lru_cache

# Cache tokenization results — tokenizing the same text multiple times is wasteful
@lru_cache(maxsize=10_000)
def tokenize_cached(text: str, max_length: int = 128):
    return tuple(tokenizer.encode(text, max_length=max_length, truncation=True))
    # Note: lru_cache requires hashable arguments — use tuple not list as return

# Cache expensive config parsing
@lru_cache(maxsize=None)  # unbounded cache
def load_config(path: str):
    with open(path) as f:
        return json.load(f)
```

#### Multiprocessing vs Threading for ML

```python
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# GIL (Global Interpreter Lock): only one Python thread runs bytecode at a time.
# Threading does NOT parallelize CPU-bound Python code.

# CPU-bound (data preprocessing, feature computation) → multiprocessing
def preprocess_record(record):
    # heavy computation
    return result

with Pool(processes=8) as pool:
    results = pool.map(preprocess_record, records)

# I/O-bound (downloading files, reading from disk, API calls) → threading
def download_file(url):
    import requests
    return requests.get(url).content

with ThreadPoolExecutor(max_workers=16) as executor:
    futures  = [executor.submit(download_file, url) for url in urls]
    contents = [f.result() for f in futures]

# DataLoader with num_workers uses multiprocessing internally
# → always use num_workers >= 2 for GPU training to avoid CPU bottleneck
```

#### Batch Processing vs Streaming

```python
# Batch — load everything into memory, faster random access
X, y = load_all_data()   # fits in RAM
loader = DataLoader(TensorDataset(X, y), batch_size=64)

# Streaming — generator, constant memory, sequential only
def data_generator(file_path, batch_size=64):
    X_batch, y_batch = [], []
    with open(file_path) as f:
        for line in f:
            record = json.loads(line)
            X_batch.append(record['features'])
            y_batch.append(record['label'])
            if len(X_batch) == batch_size:
                yield np.array(X_batch), np.array(y_batch)
                X_batch, y_batch = [], []

# Use streaming when: dataset > available RAM, data arrives continuously (online ML)
# Use batch when: dataset fits in memory, need multiple epochs, need shuffling
```

---

### 21.10 Interview Code Patterns in ML

These are the most frequently asked "implement from scratch" questions in ML engineering interviews. Know them cold.

#### Numerically Stable Softmax

Naive `exp(x) / sum(exp(x))` overflows for large $x$. Subtract the max first — mathematically identical, numerically stable.

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. Works on last axis."""
    x = x - x.max(axis=-1, keepdims=True)   # shift by max — prevents overflow
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Test
logits = np.array([[1000.0, 1001.0, 1002.0],   # would overflow without shift
                   [1.0, 2.0, 3.0]])
probs = softmax(logits)
print(probs.sum(axis=1))    # [1.0, 1.0] — sums to 1

# Batch version (same function works via axis=-1)
batch_logits = np.random.randn(32, 10)
batch_probs  = softmax(batch_logits)
assert np.allclose(batch_probs.sum(axis=1), 1.0)
```

#### Cosine Similarity from Scratch

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

```python
import numpy as np

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    dot   = np.dot(u, v)
    norm  = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-10:
        return 0.0
    return dot / norm

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarities.
    A: (m, d), B: (n, d) → returns (m, n)
    """
    # Normalize each row to unit norm
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T   # (m, n)

# Test
u = np.array([1.0, 0.0, 0.0])
v = np.array([1.0, 0.0, 0.0])
print(cosine_similarity(u, v))   # 1.0 — identical

w = np.array([0.0, 1.0, 0.0])
print(cosine_similarity(u, w))   # 0.0 — orthogonal
```

#### K-Nearest Neighbors Search with NumPy

```python
import numpy as np

def knn_search(query: np.ndarray, database: np.ndarray, k: int) -> tuple:
    """
    Find k nearest neighbors of query in database using L2 distance.
    query:    (d,)
    database: (N, d)
    returns:  (indices of shape (k,), distances of shape (k,))
    """
    # Squared L2: ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x
    # Vectorized computation avoids loop over N
    diff = database - query           # (N, d) broadcasting
    dists_sq = (diff**2).sum(axis=1)  # (N,)
    idx = np.argpartition(dists_sq, k)[:k]   # O(N) partial sort — faster than full sort
    idx = idx[np.argsort(dists_sq[idx])]     # sort the k candidates
    return idx, np.sqrt(dists_sq[idx])

def knn_batch(queries: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    """
    Batch KNN. queries: (Q, d), database: (N, d) → indices (Q, k)
    Uses cosine-similarity trick for efficiency.
    """
    # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x^T
    q_sq = (queries**2).sum(axis=1, keepdims=True)   # (Q, 1)
    d_sq = (database**2).sum(axis=1, keepdims=True).T  # (1, N)
    cross = queries @ database.T                       # (Q, N)
    dists_sq = q_sq + d_sq - 2 * cross               # (Q, N)
    return np.argpartition(dists_sq, k, axis=1)[:, :k]

# Test
rng = np.random.default_rng(42)
db  = rng.standard_normal((1000, 64))
q   = rng.standard_normal(64)
idx, dists = knn_search(q, db, k=5)
print(idx, dists)
```

#### One-Hot Encoding without sklearn

```python
import numpy as np

def one_hot(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    y: integer array of shape (N,), values in [0, num_classes)
    returns: (N, num_classes) float array
    """
    y = np.asarray(y)
    if num_classes is None:
        num_classes = y.max() + 1
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

# Test
labels = np.array([0, 2, 1, 2, 0])
ohe    = one_hot(labels, num_classes=3)
print(ohe)
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
assert ohe.sum(axis=1).tolist() == [1.0]*5
```

#### Train/Val/Test Split without sklearn

```python
import numpy as np

def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Stratified split: preserves class proportions in each split.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    rng = np.random.default_rng(seed)
    assert len(X) == len(y)

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n   = len(idx)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        X_test.extend(X[idx[:n_test]])
        y_test.extend(y[idx[:n_test]])
        X_val.extend(X[idx[n_test:n_test + n_val]])
        y_val.extend(y[idx[n_test:n_test + n_val]])
        X_train.extend(X[idx[n_test + n_val:]])
        y_train.extend(y[idx[n_test + n_val:]])

    return (np.array(X_train), np.array(X_val), np.array(X_test),
            np.array(y_train), np.array(y_val), np.array(y_test))

# Test
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = np.array([0]*50 + [1]*50)

X_tr, X_v, X_te, y_tr, y_v, y_te = train_val_test_split(X, y)
print(X_tr.shape, X_v.shape, X_te.shape)   # (70, 5) (15, 5) (15, 5)

# Verify class balance
for split, ys in [('train', y_tr), ('val', y_v), ('test', y_te)]:
    ratio = ys.mean()
    print(f"{split}: class-1 ratio = {ratio:.2f}")   # should be ~0.50
```

#### Summary: Implement-from-Scratch Checklist

| Question | Key technique |
|---|---|
| Softmax | Subtract max before exp |
| Cosine similarity | Normalize then dot product |
| KNN search | `np.argpartition` for O(N) partial sort |
| One-hot encoding | Index assignment `out[arange(N), y] = 1` |
| Stratified split | Per-class shuffle + proportional slice |
| Cross-entropy loss | `-(y * log(p+eps)).sum(axis=1).mean()` |
| Batch norm (forward) | `(x - mean) / (std + eps) * gamma + beta` |
| Attention | `softmax(QK^T / sqrt(d_k)) V` |

```python
# Cross-entropy from scratch (bonus)
def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    logits:  (N, C) raw scores
    targets: (N,)  integer class labels
    returns: scalar mean loss
    """
    probs   = softmax(logits)                         # (N, C)
    N       = len(targets)
    log_p   = np.log(probs[np.arange(N), targets] + 1e-10)  # log prob of true class
    return -log_p.mean()

# Manual attention (scaled dot-product)
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, T_q, d_k)  K: (B, T_k, d_k)  V: (B, T_k, d_v)
    returns: (B, T_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # (B, T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)                           # (B, T_q, T_k)
    return weights @ V                                  # (B, T_q, d_v)
```

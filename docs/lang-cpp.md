# C++ — Systems + OOP + Modern Features

---

## Quick Reference

| Feature | Key Facts |
|---------|-----------|
| RAII | Resource acquired in constructor, released in destructor; exception-safe by design |
| Rule of 3 | If you define destructor/copy-ctor/copy-assign, define all three |
| Rule of 5 | + move-ctor and move-assign for value semantics with efficiency |
| Rule of 0 | Use standard containers/smart pointers; define none of the five |
| `unique_ptr` | Sole owner; no overhead vs raw pointer; non-copyable, movable |
| `shared_ptr` | Reference-counted; 2-word overhead (ptr + control block) |
| `weak_ptr` | Non-owning observer of `shared_ptr`; breaks cycles |
| `std::move` | Casts to rvalue reference; does not move anything itself |
| `std::forward` | Perfect forwarding; preserves value category |
| `constexpr` | Evaluated at compile time if operands are constant |
| `consteval` | Must be evaluated at compile time (C++20); error if called at runtime |
| vtable | Per-class pointer array of virtual function addresses; one vptr per object |
| `dynamic_cast` | Runtime type check using RTTI; returns `nullptr` on failure (pointer case) |
| `decltype(auto)` | Preserves reference qualifiers; differs from plain `auto` |
| Structured bindings | `auto [a, b] = pair;` (C++17) |

---

## Core Concepts

### RAII (Resource Acquisition Is Initialization)

The invariant: resource lifetime = object lifetime. Constructors acquire, destructors release. Stack unwinding during exception propagation guarantees destructors run.

```cpp
// Without RAII
FILE *fp = fopen("x", "r");
if (!fp) throw std::runtime_error("open failed");
process(fp);
fclose(fp);  // not called if process() throws
```

```cpp
// With RAII
struct FileGuard {
    FILE *fp;
    explicit FileGuard(const char *path, const char *mode)
        : fp(fopen(path, mode))
    {
        if (!fp) throw std::runtime_error("open failed");
    }
    ~FileGuard() { fclose(fp); }
    FileGuard(const FileGuard&) = delete;
    FileGuard& operator=(const FileGuard&) = delete;
};
```

Destructor runs on scope exit — whether normal, exception, or early return.

### Rule of 3 / 5 / 0

If a class manages a resource (raw pointer, file handle, mutex), the compiler-generated copy/move operations are wrong.

**Rule of 3 (pre-C++11):** Define copy constructor, copy assignment, destructor.

**Rule of 5 (C++11+):** Add move constructor and move assignment to avoid unnecessary deep copies.

```
Operation             Generated behavior          When wrong
-----------           -------------------          ----------
Copy constructor      memberwise copy              shallow-copies raw pointer
Copy assignment       memberwise assign            double-free on self-assign
Destructor            does nothing                 leaks resource
Move constructor      same as copy (pre-C++11)     misses optimization
Move assignment       same as copy assignment      misses optimization
```

**Explicit `= delete`:** Prevents accidental copy of non-copyable resources.  
**Explicit `= default`:** Requests compiler-generated version with correct semantics.

### Move Semantics & Rvalue References

An **lvalue** has an identifiable address. An **rvalue** is a temporary or a value about to expire.

`T&&` binds to rvalues (and to named rvalue references). `std::move(x)` is a cast to `T&&` — it does not actually move; the move constructor/assignment does the work.

```cpp
std::string a = "hello";
std::string b = std::move(a);   // a is valid but unspecified (empty for string)
```

**Move semantics eliminate copies:** Moving a `vector<int>` with 1M elements transfers three words (ptr, size, cap) — $O(1)$ vs $O(n)$ copy.

**Perfect forwarding:** Preserve value category through template functions.

```cpp
template<typename T, typename... Args>
T* make(Args&&... args) {
    return new T(std::forward<Args>(args)...);
}
```

`std::forward<Args>(args)` forwards lvalues as lvalues, rvalues as rvalues.

**Reference collapsing rules:**
| Template param | Arg passed | Deduced type |
|----------------|------------|-------------|
| `T&` | lvalue `U` | `U&` |
| `T&&` | lvalue `U` | `U&` |
| `T&&` | rvalue `U` | `U&&` |

### Smart Pointers

**`unique_ptr<T>`:** Exclusive ownership. Destructor calls `delete`. Zero overhead — no control block, no atomic operations. Non-copyable; movable.

```cpp
auto p = std::make_unique<int>(42);
auto q = std::move(p);   // p is now null
```

**`shared_ptr<T>`:** Shared ownership via reference count stored in control block. Two atomic increments/decrements on copy/destroy. Use `make_shared<T>()` — allocates object and control block in one allocation.

```cpp
auto a = std::make_shared<std::string>("hello");
std::shared_ptr<std::string> b = a;   // ref count = 2
```

**`weak_ptr<T>`:** Observes a `shared_ptr` without owning. Does not increment strong ref count. Must lock before access:

```cpp
std::weak_ptr<T> w = shared;
if (auto sp = w.lock()) {
    // sp is a valid shared_ptr
}
```

Cycle-breaking: `A` owns `shared_ptr<B>`; `B` holds `weak_ptr<A>`. Without weak_ptr, neither destructor would run.

### Templates & Template Specialization

Templates are compile-time code generation parameterized on types or values.

```cpp
template<typename T>
T max(T a, T b) { return (a > b) ? a : b; }
```

**Full specialization:**
```cpp
template<>
const char* max<const char*>(const char *a, const char *b) {
    return strcmp(a, b) > 0 ? a : b;
}
```

**Partial specialization** (classes only):
```cpp
template<typename T>
class Vec<T*> { ... };   // specialization for pointer types
```

**SFINAE (Substitution Failure Is Not An Error):** Failed template substitution removes the overload candidate without error:

```cpp
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
void only_integers(T x);
```

**Variadic templates:**
```cpp
template<typename... Args>
void print(Args&&... args) {
    (std::cout << ... << args) << '\n';  // fold expression (C++17)
}
```

### STL Container Internals

| Container | Internal structure | `operator[]` | Insert (typical) | Search |
|-----------|-------------------|-------------|-----------------|--------|
| `vector` | Contiguous array; doubles capacity | $O(1)$ | $O(1)$ amortized back; $O(n)$ middle | $O(n)$ |
| `deque` | Array of fixed-size chunks | $O(1)$ | $O(1)$ front/back | $O(n)$ |
| `list` | Doubly-linked nodes | — | $O(1)$ | $O(n)$ |
| `map` | Red-black tree (ordered) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| `unordered_map` | Hash table (chaining) | $O(1)$ avg | $O(1)$ avg | $O(1)$ avg |
| `set` | Red-black tree | — | $O(\log n)$ | $O(\log n)$ |
| `priority_queue` | Binary heap over `vector` | — | $O(\log n)$ | $O(1)$ top |

`unordered_map` worst-case is $O(n)$ (all keys hash to same bucket). Rehashing occurs when load factor > threshold (default 1.0); capacity doubles.

`vector` iterator invalidation: any reallocation invalidates all iterators/references. `push_back` may reallocate.

### Lambdas & `std::function`

Lambda syntax: `[captures](params) specifiers -> ret { body }`.

**Capture modes:**
- `[=]` — capture all by value (copy at lambda creation)
- `[&]` — capture all by reference (dangling if lambda outlives scope)
- `[x, &y]` — `x` by value, `y` by reference
- `[this]` — capture `this` pointer (member access)
- `[*this]` (C++17) — capture `*this` by value

```cpp
int factor = 3;
auto mul = [factor](int x) { return x * factor; };
// factor is copied; lambda is a closure object
```

`std::function<R(Args...)>` — type-erased callable wrapper. Has overhead: heap allocation for large closures, virtual dispatch. Use templates/`auto` for local callables to keep zero cost.

`mutable` lambda: allows modifying captured-by-value variables (they are `const` by default in the closure).

### `constexpr` / `consteval`

`constexpr` function: evaluated at compile time when arguments are constant expressions; falls back to runtime otherwise.

```cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
static_assert(factorial(5) == 120);   // computed at compile time
```

`consteval` (C++20): function must be evaluated at compile time. Calling with runtime arguments is a compile error. Used for compile-time validation and code generation.

`constinit` (C++20): variable must be initialized with a constant expression (prevents static initialization order fiasco) but is not `const` — can be modified at runtime.

### Concepts (C++20)

Concepts constrain template parameters — replace SFINAE with readable syntax.

```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T square(T x) { return x * x; }
```

Abbreviated function templates:
```cpp
auto square(Numeric auto x) { return x * x; }
```

Standard library concepts: `std::integral`, `std::floating_point`, `std::same_as<T>`, `std::convertible_to<T>`, `std::invocable<F, Args...>`, `std::ranges::range`.

### Virtual Dispatch & vtable

Every class with virtual functions has a vtable (virtual function table): an array of function pointers, one per virtual function. Each object of a polymorphic class contains a vptr (virtual pointer) as its first hidden member, pointing to its class's vtable.

```
class Animal { virtual void speak(); };
class Dog : Animal { void speak() override; };

Dog d;
// d in memory: [ vptr ] [ other members ]
//                 |
//                 ▼
//             Dog::vtable: [ &Dog::speak ]
```

Virtual call: load vptr → index vtable → indirect call. Cost: one extra memory load + indirect branch (not predicted by branch predictor as reliably as direct calls).

**`override` keyword:** Compile-time check that the function actually overrides a virtual function (catches signature mismatches).

**`final`:** Prevents further derivation (`class Leaf final`) or overriding (`void f() final`). Enables devirtualization by compiler.

### Multiple Inheritance & Diamond Problem

```
class A { int x; };
class B : public A {};
class C : public A {};
class D : public B, public C {};  // D has two copies of A::x
```

Access to `x` through `D` is ambiguous — `D::B::x` vs `D::C::x`.

**Virtual inheritance:** `class B : virtual public A` — a single shared subobject of `A` for all virtual bases. Implemented via an extra pointer (vbptr) or offset. Slight overhead on member access.

```cpp
class B : virtual public A {};
class C : virtual public A {};
class D : public B, public C {};   // single A subobject
```

### Exceptions

`throw expr;` — stack unwinds, destructors called in reverse construction order. Caught by `catch(T& e)` matching by type (exact match, public base, `...`).

**`noexcept`:** Declares function does not throw. If it does throw, `std::terminate()` is called. Enables optimizer to elide exception-handling code. Move constructors should be `noexcept` — `vector` uses move only if move ctor is `noexcept`, otherwise copies.

**Exception safety guarantees:**
| Level | Guarantee |
|-------|-----------|
| No-throw | Operation always succeeds |
| Strong | On failure, state unchanged (commit-or-rollback) |
| Basic | No resource leaks; object in valid but unspecified state |
| None | No guarantees |

### Memory Layout of Objects

```
struct Base {
    virtual void f();     // vptr: 8 bytes
    int x;                // 4 bytes
    // 4 bytes padding
};  // sizeof = 16

struct Derived : Base {
    double y;             // 8 bytes (no extra vptr; reuses Base's)
};  // sizeof = 24

struct MultiBase : B1, B2 {
    // two vptrs (one per base with virtual functions)
};
```

`alignof(T)` — required alignment. `alignas(N)` — enforce alignment on variable/member. `std::aligned_storage` — raw aligned buffer without constructing.

---

## Code Examples

### Custom RAII Wrapper

```c++
#include <cstdio>
#include <stdexcept>
#include <utility>

class FileHandle {
    FILE *fp_;
public:
    explicit FileHandle(const char *path, const char *mode)
        : fp_(std::fopen(path, mode))
    {
        if (!fp_) throw std::runtime_error(std::string("cannot open: ") + path);
    }

    ~FileHandle() { if (fp_) std::fclose(fp_); }

    FileHandle(FileHandle&& other) noexcept : fp_(other.fp_) { other.fp_ = nullptr; }
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (fp_) std::fclose(fp_);
            fp_ = other.fp_;
            other.fp_ = nullptr;
        }
        return *this;
    }

    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FILE* get() const noexcept { return fp_; }
    FILE* release() noexcept { FILE *f = fp_; fp_ = nullptr; return f; }
};

int main() {
    try {
        FileHandle f("lang-cpp.md", "r");
        char buf[64];
        if (std::fgets(buf, sizeof buf, f.get()))
            std::fputs(buf, stdout);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
```

### Variadic Template

```c++
#include <iostream>
#include <type_traits>

template<typename T>
void print_one(const T& val) {
    if constexpr (std::is_same_v<T, bool>)
        std::cout << (val ? "true" : "false");
    else
        std::cout << val;
}

template<typename... Args>
void println(Args&&... args) {
    std::size_t i = 0, n = sizeof...(args);
    ((print_one(args), (++i < n ? std::cout << ", " : std::cout << '\n')), ...);
}

template<typename T, typename... Rest>
T sum(T first, Rest... rest) {
    if constexpr (sizeof...(rest) == 0)
        return first;
    else
        return first + sum(rest...);
}

int main() {
    println(1, 3.14, "hello", true);
    std::cout << "sum = " << sum(1, 2, 3, 4, 5) << '\n';
    return 0;
}
```

### Lambda Captures

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

std::function<int()> make_counter(int start = 0) {
    int count = start;
    return [count]() mutable { return count++; };
}

int main() {
    auto counter = make_counter(10);
    for (int i = 0; i < 5; i++)
        std::cout << counter() << ' ';
    std::cout << '\n';

    std::vector<int> v = {5, 2, 8, 1, 9, 3};
    int threshold = 4;
    auto it = std::partition(v.begin(), v.end(),
                             [threshold](int x) { return x <= threshold; });

    std::cout << "≤" << threshold << ": ";
    for (auto p = v.begin(); p != it; ++p) std::cout << *p << ' ';
    std::cout << "\n>" << threshold << ": ";
    for (auto p = it; p != v.end(); ++p) std::cout << *p << ' ';
    std::cout << '\n';

    return 0;
}
```

### vtable Demonstration

```c++
#include <iostream>
#include <memory>
#include <vector>

class Shape {
public:
    virtual double area() const = 0;
    virtual const char* name() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double r_;
public:
    explicit Circle(double r) : r_(r) {}
    double area() const override { return 3.14159265358979 * r_ * r_; }
    const char* name() const override { return "Circle"; }
};

class Rect : public Shape {
    double w_, h_;
public:
    Rect(double w, double h) : w_(w), h_(h) {}
    double area() const override { return w_ * h_; }
    const char* name() const override { return "Rectangle"; }
};

void print_info(const Shape &s) {
    std::cout << s.name() << " area = " << s.area() << '\n';
}

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(3.0));
    shapes.push_back(std::make_unique<Rect>(4.0, 5.0));
    shapes.push_back(std::make_unique<Circle>(1.5));

    for (const auto &s : shapes)
        print_info(*s);   // virtual dispatch via vtable

    std::cout << "vptr size overhead: " << sizeof(Circle) - sizeof(double)
              << " bytes\n";
    return 0;
}
```

---

## Interview Q&A

**Q1: What is the difference between `std::move` and `std::forward`?**

`std::move(x)` unconditionally casts `x` to `T&&` (rvalue reference), signaling that the value may be moved from. It does not actually move anything. `std::forward<T>(x)` is a conditional cast: if `T` is an lvalue reference type (deduced when an lvalue is passed), `x` is cast to `T&`; if `T` is a non-reference type, `x` is cast to `T&&`. Used in generic code to preserve the value category of arguments passed to a template function — called perfect forwarding. Using `std::move` in a forwarding context would unconditionally move lvalues, which is wrong.

---

**Q2: Why should move constructors be declared `noexcept`, and what happens if they are not?**

`std::vector` must provide the strong exception safety guarantee during `push_back`/`resize`. If moving elements could throw, a partially-moved state would leave the vector corrupted. So the standard requires vector to copy elements unless the move constructor is `noexcept`. If `noexcept` is absent, `vector` falls back to copying even when a move would be possible — $O(n)$ instead of $O(n)$ with a smaller constant, but more importantly, it defeats the purpose of move semantics. Same applies to `std::swap` and algorithm operations.

---

**Q3: Explain the vtable layout and cost of virtual dispatch.**

Each polymorphic class has a static vtable — an array of function pointers, one entry per virtual function in declaration order (plus RTTI pointer). Each polymorphic object starts with a hidden `vptr` pointing to its class's vtable. A virtual call compiles to: load `vptr` from object, load function pointer at fixed offset in vtable, indirect call. This is one additional indirection vs a direct call. The indirect call inhibits inlining and stresses the branch predictor (indirect branch target prediction). On modern CPUs, mispredicted indirect calls cost ~15–20 cycles. Mark classes `final` to allow devirtualization.

---

**Q4: What is the difference between `shared_ptr` and `weak_ptr`, and when would you use a `weak_ptr`?**

`shared_ptr` participates in ownership — its constructor increments the strong reference count, its destructor decrements it; the managed object is destroyed when the strong count reaches zero. `weak_ptr` is a non-owning observer — it does not affect the strong count (only the weak count, which controls control-block lifetime). Use `weak_ptr` to break reference cycles (e.g., parent/child bidirectional graphs where each holds a `shared_ptr` to the other — neither destructor runs). Also use for caches: store `weak_ptr` in a cache map; if the strong-owner has destroyed the object, `w.lock()` returns null, and the cache can refresh.

---

**Q5: What is CRTP and what problem does it solve?**

Curiously Recurring Template Pattern: a base class takes the derived class as a template parameter.

```cpp
template<typename Derived>
class Base {
    void interface() { static_cast<Derived*>(this)->implementation(); }
};
class Child : public Base<Child> {
    void implementation() { ... }
};
```

Achieves static (compile-time) polymorphism without vtable overhead. Useful for mixins that need to call derived-class methods (e.g., `enable_shared_from_this`, `boost::operators`). Limitation: heterogeneous containers require a common runtime base.

---

**Q6: Explain how `std::unordered_map` handles hash collisions and resizing.**

Collision resolution uses separate chaining: each bucket holds a linked list of key-value pairs with the same hash modulo bucket count. On lookup, hash the key, index into bucket array, traverse the list comparing keys. Load factor = elements / bucket count. When load factor exceeds `max_load_factor()` (default 1.0), the table rehashes: allocates roughly double the buckets, reinserts all elements — amortized $O(1)$ per insert but $O(n)$ for the rehash operation. Worst-case for all operations is $O(n)$ (degenerate hash function sends all keys to same bucket). Reserve buckets upfront with `reserve(n)` to avoid rehashing.

---

**Q7: What is the diamond problem in multiple inheritance and how does virtual inheritance solve it?**

When `B` and `C` both inherit from `A`, and `D` inherits from both `B` and `C`, `D` has two copies of `A`'s subobject — one via `B`, one via `C`. Accessing `A`'s members through `D` is ambiguous. Virtual inheritance (`class B : virtual public A`) instructs the compiler to ensure only one shared `A` subobject exists in any diamond. Implemented via a vbptr (virtual base pointer) or offset table that locates the shared base. Cost: one extra indirection on member access through the virtual base; more complex object layout.

---

**Q8: What does `std::enable_if` do, and how do concepts (C++20) improve on it?**

`std::enable_if<Condition, T>::type` is defined as `T` when `Condition` is true and produces a substitution failure otherwise — removing the template from the overload set via SFINAE. This is syntactically verbose and produces cryptic error messages. C++20 concepts replace this with a first-class language feature: `template<std::integral T>` reads like a type constraint, is checked at the call site with a clear diagnostic, and can express complex requirements via `requires` clauses. Concepts also support subsumption: a more constrained overload is preferred over a less constrained one, which SFINAE did not handle cleanly.

---

**Q9: What are the exception safety guarantees and how do you achieve the strong guarantee?**

The strong guarantee (commit-or-rollback) means the operation either succeeds fully or leaves the program state unchanged. The standard idiom: copy-and-swap.

```cpp
T& operator=(T other) {   // other is a copy (may throw in copy-ctor)
    swap(*this, other);   // swap is noexcept
    return *this;         // other (old state) destroyed on exit
}
```

If the copy constructor throws, `*this` is untouched. The swap is `noexcept`, so after a successful copy, the assignment cannot fail. This pattern also handles self-assignment correctly without an explicit check.

---

**Q10: Explain `decltype`, `decltype(auto)`, and how they differ from `auto`.**

`auto` deduces type discarding references and cv-qualifiers (like a template type parameter). `decltype(expr)` yields the declared type of an entity or the type-and-value-category of an expression: lvalue expression → `T&`; xvalue → `T&&`; prvalue → `T`. `decltype(auto)` uses `decltype` rules on the initializer, preserving references. Consequence: `auto x = v[0];` copies the element; `decltype(auto) x = v[0];` holds a reference. Critical in generic return types where the function should return a reference when the underlying call returns a reference — using `auto` would strip the reference.

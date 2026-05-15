# C++ — Zero to Modern Expert

---

## 1. C vs C++: What's Added

| Feature | C | C++ |
|---|---|---|
| Classes / structs with methods | No | Yes |
| Operator overloading | No | Yes |
| Function overloading | No | Yes |
| Templates (generics) | No | Yes |
| Exceptions | No | Yes |
| RAII | No | Yes (via constructors/destructors) |
| References (`int&`) | No | Yes |
| `new` / `delete` | No | Yes (plus smart pointers) |
| Namespaces | No | Yes |
| `bool` type | No (use int) | Yes |
| Standard library (STL) | No (only libc) | Yes |
| `inline` semantics | Limited | Full |
| Default arguments | No | Yes |
| `const` correctness | Weak | Strong (methods, pointers) |
| Constructors / destructors | No | Yes |
| Virtual dispatch / polymorphism | Manual (fn pointers) | Built-in (vtable) |
| Move semantics | No | Yes (C++11+) |
| Lambdas | No | Yes (C++11+) |
| Concepts | No | Yes (C++20) |
| Coroutines | No | Yes (C++20) |

C++ is a strict superset of C for most practical code. Every valid C program can be compiled as C++ with minor exceptions (implicit `void*` conversions, `//` comments in older C standards).

---

## 2. Classes & Objects

### class vs struct

The only difference in C++ is the **default access specifier**:

| Keyword | Default access |
|---|---|
| `class` | `private` |
| `struct` | `public` |

Both support methods, constructors, inheritance, virtual functions. Convention: use `struct` for plain data aggregates, `class` for encapsulated types with invariants.

### Access Specifiers

| Specifier | Accessible from |
|---|---|
| `public` | Anywhere |
| `protected` | Class itself + derived classes |
| `private` | Class itself only (not even derived classes) |

```cpp
class BankAccount {
public:
    BankAccount(double initial) : balance(initial) {}
    void deposit(double amount) { balance += amount; }
    double getBalance() const { return balance; }

protected:
    void audit() {}  // derived classes can call this

private:
    double balance;  // hidden from outside
};
```

### Member Variables and Methods

```cpp
class Rectangle {
public:
    double width, height;  // member variables

    double area() const {       // const method: cannot modify members
        return width * height;
    }

    void scale(double factor) { // non-const: modifies state
        width  *= factor;
        height *= factor;
    }
};
```

`const` after the method signature means the method does not modify any non-`mutable` member. The compiler enforces this.

### Constructor Types

**Default constructor** — no arguments:
```cpp
class Vec2 {
public:
    double x, y;
    Vec2() : x(0.0), y(0.0) {}  // initializer list preferred over body assignment
};
```

**Parameterized constructor**:
```cpp
Vec2(double x, double y) : x(x), y(y) {}
```

**Copy constructor** — initialize from same type:
```cpp
Vec2(const Vec2& other) : x(other.x), y(other.y) {}
```

**Move constructor** — steal resources from rvalue:
```cpp
Vec2(Vec2&& other) noexcept : x(other.x), y(other.y) {
    other.x = 0; other.y = 0;
}
```

**Delegating constructor** (C++11) — one constructor calls another:
```cpp
Vec2() : Vec2(0.0, 0.0) {}
```

**`explicit` constructor** — prevents implicit conversions:
```cpp
explicit Vec2(double scalar) : x(scalar), y(scalar) {}
// Vec2 v = 5.0;  // error: explicit prevents this
// Vec2 v(5.0);   // ok
```

### Initializer List vs Body Assignment

```cpp
// Preferred: initializer list
Foo(int a, std::string s) : a_(a), s_(s) {}

// Wrong: default-constructs then assigns (wasteful for complex types)
Foo(int a, std::string s) { a_ = a; s_ = s; }
```

Members are initialized in **declaration order**, not initializer-list order.

### Destructor

```cpp
class FileHandle {
public:
    FileHandle(const char* path) { fp = fopen(path, "r"); }
    ~FileHandle() {
        if (fp) fclose(fp);  // cleanup guaranteed, even through exceptions
    }
private:
    FILE* fp = nullptr;
};
```

Destructors are called automatically at end of scope, in **reverse construction order**. Destructors of base classes must be `virtual` if you delete through a base pointer.

### `this` Pointer

`this` is an implicit pointer to the current object inside non-static methods:

```cpp
class Builder {
public:
    Builder& setName(std::string n) { name = n; return *this; }
    Builder& setAge(int a)          { age  = a; return *this; }
private:
    std::string name;
    int age = 0;
};

// Method chaining works because of return *this
Builder b;
b.setName("Alice").setAge(30);
```

### Object Lifecycle (ASCII Timeline)

```
Stack object:
  {
    |-- constructor runs (members initialized)
    |
    |   ... use object ...
    |
    `-- destructor runs automatically at }

Heap object (raw):
  new  --> constructor runs
  ...  --> use object
  delete --> destructor runs (you must call this!)

Heap object (unique_ptr):
  make_unique --> constructor runs
  ...         --> use object
  (scope end) --> destructor of unique_ptr calls delete automatically
```

---

## 3. Operator Overloading

### Which Operators Can Be Overloaded

| Can Overload | Cannot Overload |
|---|---|
| `+  -  *  /  %` | `::` (scope resolution) |
| `==  !=  <  >  <=  >=` | `.` (member access) |
| `[]  ()  ->  ->*` | `.*` (pointer-to-member) |
| `<<  >>` (streams) | `?:` (ternary) |
| `=  +=  -=  *=  /=` | `sizeof`, `typeid`, `alignof` |
| `++  --  !  ~  &  *` | |
| `new  delete` | |
| `,` (comma) | |

### Member vs Non-member Overloads

| Scenario | Use |
|---|---|
| Left operand is the class | Member function |
| Left operand is **not** the class (`<<`, `>>`) | Non-member (friend) |
| Symmetric binary ops (`+`, `==`) | Non-member (allows `int + MyType`) |
| Assignment ops (`=`, `+=`) | Must be member |
| `[]`, `()`, `->`, `=` | Must be member |

```cpp
class Vector2 {
public:
    double x, y;

    Vector2(double x, double y) : x(x), y(y) {}

    // Member: left operand must be Vector2
    Vector2& operator+=(const Vector2& rhs) {
        x += rhs.x; y += rhs.y;
        return *this;  // enables v += a += b chaining
    }

    // Unary minus
    Vector2 operator-() const { return {-x, -y}; }

    // Subscript
    double& operator[](int i) { return i == 0 ? x : y; }
    const double& operator[](int i) const { return i == 0 ? x : y; }
};

// Non-member: symmetric, allows double + Vector2 if desired
Vector2 operator+(Vector2 lhs, const Vector2& rhs) {
    lhs += rhs;  // reuse +=
    return lhs;
}

bool operator==(const Vector2& a, const Vector2& b) {
    return a.x == b.x && a.y == b.y;
}

// Stream output: must be non-member (std::ostream is not our class)
std::ostream& operator<<(std::ostream& os, const Vector2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
```

### Spaceship Operator (C++20)

```cpp
#include <compare>

struct Point {
    int x, y;
    auto operator<=>(const Point&) const = default;  // generates all 6 comparison ops
};
```

### Gotchas

- Always return `*this` from `operator=`, `operator+=`, etc. to enable chaining.
- `operator[]` should have both `const` and non-`const` overloads.
- `operator==` should be symmetric: define as non-member.
- For `operator++`, distinguish prefix (`++x`) from postfix (`x++`) via dummy `int` parameter.

```cpp
Vec& operator++() { ++val; return *this; }         // prefix
Vec  operator++(int) { Vec t=*this; ++val; return t; } // postfix
```

---

## 4. Inheritance

### Types

| Type | Syntax | Description |
|---|---|---|
| Single | `class B : public A` | B inherits from A |
| Multiple | `class C : public A, public B` | C inherits from both |
| Multilevel | `class C : public B` where B inherits A | Chain |
| Hierarchical | Multiple classes inherit same base | B,C,D all from A |

### Inheritance Access

| Base member | `public` inheritance | `protected` inheritance | `private` inheritance |
|---|---|---|---|
| `public` | `public` in derived | `protected` in derived | `private` in derived |
| `protected` | `protected` in derived | `protected` in derived | `private` in derived |
| `private` | Inaccessible | Inaccessible | Inaccessible |

`public` inheritance models "is-a". `private` inheritance models "implemented-in-terms-of" (prefer composition instead).

### Virtual Methods

Without `virtual`, method calls are resolved at **compile time** (static dispatch):

```cpp
struct Animal {
    void speak() { std::cout << "...\n"; }  // non-virtual
};
struct Dog : Animal {
    void speak() { std::cout << "Woof\n"; }
};

Animal* a = new Dog();
a->speak();  // prints "..." — calls Animal::speak, NOT Dog::speak
```

With `virtual`, calls are resolved at **runtime** (dynamic dispatch):

```cpp
struct Animal {
    virtual void speak() { std::cout << "...\n"; }
    virtual ~Animal() {}  // always virtual destructor in polymorphic base
};
struct Dog : Animal {
    void speak() override { std::cout << "Woof\n"; }
};

Animal* a = new Dog();
a->speak();  // prints "Woof" — correct
```

### vtable and vptr Memory Layout

Every class with virtual functions gets a **vtable** (per-class, static). Every object gets a **vptr** (per-object, hidden first member).

```
Dog object in memory:
┌─────────────────────┐
│ vptr ────────────────┼──► Dog's vtable
│ (members of Animal) │    ┌──────────────────────┐
│ (members of Dog)    │    │ &Dog::speak          │
└─────────────────────┘    │ &Animal::~Animal     │
                           │ (other virtual fns)  │
                           └──────────────────────┘

sizeof(Dog) includes the hidden vptr (typically 8 bytes on 64-bit).
```

Virtual call cost: one pointer dereference (vptr) + one indexed load (vtable entry) + indirect call. Inlining is impossible.

### Pure Virtual Functions and Abstract Classes

```cpp
struct Shape {
    virtual double area() const = 0;    // pure virtual
    virtual double perimeter() const = 0;
    virtual ~Shape() = default;
};

// Shape s;  // error: cannot instantiate abstract class

struct Circle : Shape {
    double r;
    Circle(double r) : r(r) {}
    double area()      const override { return 3.14159 * r * r; }
    double perimeter() const override { return 2 * 3.14159 * r; }
};
```

### `override` and `final` (C++11)

```cpp
struct Base {
    virtual void foo(int);
    virtual void bar();
};

struct Derived : Base {
    void foo(int) override;    // compiler checks this really overrides
    // void foo(float) override; // error: no matching virtual in Base
    void bar() final;          // no further class can override bar
};

struct Leaf final : Derived {  // final on class: cannot be inherited
    // ...
};
```

### Diamond Problem and Virtual Inheritance

```
    A
   / \
  B   C
   \ /
    D
```

Without virtual inheritance, D contains two copies of A. With `virtual`:

```cpp
struct A { int val = 0; };

struct B : virtual A {};
struct C : virtual A {};

struct D : B, C {
    void set(int v) { val = v; }  // unambiguous: one shared A
};
```

`virtual` inheritance adds an extra pointer in B and C to the shared A subobject.

### Object Slicing — Hidden Bug

```cpp
void process(Animal a) {  // takes by value
    a.speak();
}

Dog d;
process(d);  // Dog's extra members are sliced off! Copies only Animal part.
```

Fix: always pass polymorphic objects by pointer or reference.

```cpp
void process(Animal& a) { a.speak(); }  // no slicing
void process(Animal* a) { a->speak(); } // no slicing
```

---

## 5. RAII & Resource Management

### The Core Idea

**Resource Acquisition Is Initialization**: tie the lifetime of a resource (memory, file, mutex, socket) to the lifetime of an object. Constructor acquires, destructor releases. The C++ runtime guarantees destructors run when objects go out of scope — even through exceptions.

```
Without RAII:                With RAII:
  acquire resource             {
  ...                            Resource r(args); // acquires
  if (error) {                   ...
    release resource  // easy    // exception? still releases!
    return;                    } // destructor: releases
  }                         
  ...
  release resource  // must not forget
```

### Full File Handle Example

```cpp
#include <cstdio>
#include <stdexcept>
#include <string>

class File {
public:
    explicit File(const std::string& path, const char* mode = "r") {
        fp_ = fopen(path.c_str(), mode);
        if (!fp_) throw std::runtime_error("Cannot open: " + path);
    }

    ~File() {
        if (fp_) fclose(fp_);
    }

    // Prevent copying (you cannot copy a file handle)
    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    // Allow moving
    File(File&& other) noexcept : fp_(other.fp_) { other.fp_ = nullptr; }
    File& operator=(File&& other) noexcept {
        if (this != &other) { fclose(fp_); fp_ = other.fp_; other.fp_ = nullptr; }
        return *this;
    }

    std::string readLine() {
        char buf[1024];
        if (!fgets(buf, sizeof(buf), fp_)) return {};
        return buf;
    }

private:
    FILE* fp_ = nullptr;
};

void example() {
    File f("data.txt");
    std::string line = f.readLine();
    // ... process ...
    // f's destructor closes the file automatically here
}
```

### Why Exceptions Make Manual Cleanup Unreliable

```cpp
void bad() {
    FILE* fp = fopen("x.txt", "r");
    process(fp);  // throws? fp leaks!
    fclose(fp);   // never reached
}

void good() {
    File fp("x.txt");  // RAII
    process(fp);       // throws? destructor closes it
}
```

---

## 6. Rule of 3 / 5 / 0

### Summary Table

| Rule | Special Members | When It Applies |
|---|---|---|
| Rule of 3 | Copy ctor, Copy assignment, Destructor | Class manually manages a resource (C++98/03) |
| Rule of 5 | + Move ctor, Move assignment | C++11+: enable efficient moves |
| Rule of 0 | None (let compiler generate) | Use RAII types (smart ptrs, containers) |

The compiler generates default versions of all five only when you don't define any. Defining one suppresses generation of others (partially — see table).

| You define | Compiler generates |
|---|---|
| Destructor | Copy ctor and copy assign (deprecated behavior), no move |
| Copy ctor | Copy assign; destructor; no move |
| Move ctor | Nothing else |
| Move assign | Nothing else |

### Full Example: Rule of 5

```cpp
#include <cstring>
#include <utility>
#include <iostream>

class Buffer {
public:
    explicit Buffer(std::size_t size)
        : data_(new char[size]), size_(size) {
        std::memset(data_, 0, size_);
        std::cout << "Construct\n";
    }

    // 1. Destructor
    ~Buffer() {
        delete[] data_;
        std::cout << "Destruct\n";
    }

    // 2. Copy constructor
    Buffer(const Buffer& other)
        : data_(new char[other.size_]), size_(other.size_) {
        std::memcpy(data_, other.data_, size_);
        std::cout << "Copy construct\n";
    }

    // 3. Copy assignment
    Buffer& operator=(const Buffer& other) {
        if (this == &other) return *this;  // self-assignment guard
        char* newData = new char[other.size_];  // allocate first (exception safety)
        std::memcpy(newData, other.data_, other.size_);
        delete[] data_;
        data_ = newData;
        size_ = other.size_;
        std::cout << "Copy assign\n";
        return *this;
    }

    // 4. Move constructor
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "Move construct\n";
    }

    // 5. Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "Move assign\n";
        return *this;
    }

    std::size_t size() const { return size_; }

private:
    char* data_;
    std::size_t size_;
};
```

### Rule of 0 — Preferred

```cpp
#include <vector>
#include <string>

class Person {
public:
    Person(std::string name, std::vector<int> scores)
        : name_(std::move(name)), scores_(std::move(scores)) {}
    // No destructor, no copy/move — compiler generates correct versions
    // because string and vector handle their own memory
private:
    std::string name_;
    std::vector<int> scores_;
};
```

---

## 7. Move Semantics & Rvalue References

### lvalue vs rvalue

| Category | Definition | Examples |
|---|---|---|
| lvalue | Has identity, can take its address | `x`, `obj.member`, `*ptr`, `arr[i]` |
| rvalue | Temporary, no persistent address | `42`, `x + y`, `std::string("hi")`, function return |
| xvalue | "expiring" — rvalue ref to named object | `std::move(x)` |

```cpp
int x = 5;
int* p = &x;     // ok: x is lvalue
int* q = &42;    // error: 42 is rvalue, no address
```

### Rvalue References (`&&`)

```cpp
void foo(int& x)  { }  // binds only to lvalues
void foo(int&& x) { }  // binds only to rvalues

int a = 5;
foo(a);           // calls foo(int&)
foo(10);          // calls foo(int&&)
foo(std::move(a)); // casts a to rvalue, calls foo(int&&)
```

### `std::move` — Just a Cast

`std::move` does **not** move anything. It is an unconditional cast to an rvalue reference, signaling "you may steal my resources":

```cpp
template<typename T>
typename std::remove_reference<T>::type&&
move(T&& t) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(t);
}
```

After `std::move(obj)`, `obj` is in a **valid but unspecified state** — you can reassign it but not rely on its value.

### Move Constructor and Move Assignment

```cpp
class MyVec {
public:
    MyVec(MyVec&& other) noexcept
        : data_(other.data_), size_(other.size_), cap_(other.cap_) {
        other.data_ = nullptr;  // leave other in valid state
        other.size_ = 0;
        other.cap_  = 0;
    }

    MyVec& operator=(MyVec&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_; other.data_ = nullptr;
            size_ = other.size_; other.size_ = 0;
            cap_  = other.cap_;  other.cap_  = 0;
        }
        return *this;
    }
private:
    int* data_ = nullptr;
    std::size_t size_ = 0, cap_ = 0;
};
```

Declare move operations `noexcept` — STL containers only use moves (instead of copies) during reallocation if the move is `noexcept`.

### Perfect Forwarding with `std::forward`

Preserves the value category (lvalue/rvalue) of arguments passed to a template:

```cpp
template<typename T, typename... Args>
T create(Args&&... args) {
    return T(std::forward<Args>(args)...);
}

std::string s = "hello";
auto a = create<std::string>(s);            // copy (lvalue)
auto b = create<std::string>(std::move(s)); // move (rvalue)
```

Without `std::forward`, `args` inside the template is always an lvalue (named parameter = lvalue), so you'd always copy.

### Return Value Optimization (RVO / NRVO)

The compiler can construct the return value directly in the caller's memory, eliminating the copy/move:

```cpp
std::vector<int> makeVec() {
    std::vector<int> v = {1, 2, 3};  // NRVO: v constructed in-place at call site
    return v;
    // Do NOT write return std::move(v); — defeats NRVO
}
```

RVO (unnamed) is mandatory in C++17. NRVO (named) is permitted but not required.

---

## 8. Smart Pointers

All smart pointers are in `<memory>`. Never use raw `new`/`delete` in modern C++.

### `unique_ptr` — Sole Ownership

```cpp
#include <memory>

struct Node {
    int val;
    std::unique_ptr<Node> next;
    Node(int v) : val(v) {}
};

auto n1 = std::make_unique<Node>(1);
auto n2 = std::make_unique<Node>(2);

n1->next = std::move(n2);  // transfer ownership
// n2 is now null

// n1 goes out of scope: deletes Node(1), which deletes Node(2) via its destructor
```

`unique_ptr` has **zero overhead** vs a raw pointer in most ABIs.

### `shared_ptr` — Reference Counting

```cpp
auto s1 = std::make_shared<int>(42);
auto s2 = s1;  // s1 and s2 both own the int; ref count = 2

s1.reset();    // ref count drops to 1
s2.reset();    // ref count drops to 0 → int is deleted
```

Control block layout in memory:

```
shared_ptr<T> object:
┌──────────┐    ┌─────────────────────────────────────┐
│ ptr ─────┼───►│  T object (the managed data)        │
│ ctrl_blk ┼──┐ └─────────────────────────────────────┘
└──────────┘  │
              │  Control Block:
              └►┌─────────────────────────────────────┐
                │ strong ref count (shared_ptr count)  │
                │ weak ref count   (weak_ptr count)    │
                │ deleter                              │
                │ allocator                            │
                └─────────────────────────────────────┘

make_shared allocates T and control block in ONE allocation.
shared_ptr(new T) requires TWO allocations.
```

### `weak_ptr` — Non-owning Observer

```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node>   prev;  // weak: no ownership, breaks cycles
};

std::weak_ptr<int> wp;
{
    auto sp = std::make_shared<int>(10);
    wp = sp;  // weak reference
}   // sp destroyed, int deleted (weak_ptr doesn't prevent deletion)

if (auto sp = wp.lock()) {  // try to get a shared_ptr
    std::cout << *sp;
} else {
    std::cout << "expired\n";
}
```

### Decision Table

| Scenario | Use |
|---|---|
| Single, clear owner | `unique_ptr` |
| Shared ownership (multiple owners) | `shared_ptr` |
| Observe without owning (break cycles) | `weak_ptr` |
| Non-owning reference, lifetime guaranteed | Raw pointer `T*` (no ownership) |
| Stack/value semantics | Plain object (no pointer) |

### `make_unique` and `make_shared` vs `new`

Prefer `make_*` in all cases:

| Issue | `shared_ptr<T>(new T(...))` | `make_shared<T>(...)` |
|---|---|---|
| Allocations | 2 (T + control block) | 1 (combined) |
| Exception safety | Risk of leak in `f(shared_ptr<T>(new T), g())` | Safe |
| Custom deleter | Supported | Not supported |

```cpp
// Correct
auto p = std::make_unique<Foo>(arg1, arg2);
auto s = std::make_shared<Bar>(arg1, arg2);

// Avoid
auto p = std::unique_ptr<Foo>(new Foo(arg1, arg2));
```

---

## 9. Templates

### Function Templates

```cpp
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

max(3, 5);          // T deduced as int
max(3.0, 5.0);      // T deduced as double
max<int>(3, 5);     // explicit instantiation
```

### Class Templates

```cpp
template<typename T, std::size_t N>
class StaticArray {
public:
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }
    std::size_t size() const { return N; }
private:
    T data_[N];
};

StaticArray<int, 5> arr;
arr[0] = 42;
```

### Template Specialization

**Full specialization** — specific type:

```cpp
template<typename T>
bool isNull(T* p) { return p == nullptr; }

template<>  // full specialization for bool*
bool isNull(bool* p) { return p == nullptr || !*p; }
```

**Partial specialization** — for class templates only:

```cpp
template<typename T>
struct Wrapper { T val; };

template<typename T>
struct Wrapper<T*> {   // partial spec: pointer types
    T* val;
    T deref() { return *val; }
};
```

### Variadic Templates and Fold Expressions (C++17)

```cpp
// C++11 variadic: recursive expansion
template<typename T>
T sum(T t) { return t; }

template<typename T, typename... Rest>
T sum(T first, Rest... rest) { return first + sum(rest...); }

// C++17 fold expression: cleaner
template<typename... Args>
auto sum(Args... args) { return (args + ...); }  // unary right fold

sum(1, 2, 3, 4);  // 10
```

Fold syntax:
- `(args op ...)` — right fold: `a1 op (a2 op (a3 op init))`
- `(... op args)` — left fold
- `(args op ... op init)` — right fold with init

### `typename` vs `class` in Template Parameters

They are **identical** for type parameters. Convention: use `typename` for clarity (`class` can confuse beginners into thinking only classes are accepted). Both accept any type including primitives.

```cpp
template<typename T>  // same as template<class T>
void foo(T t);
```

### SFINAE: `enable_if`

SFINAE = Substitution Failure Is Not An Error. Template substitution failure removes the candidate silently instead of erroring.

```cpp
#include <type_traits>

// Only enabled when T is integral
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
halve(T x) { return x / 2; }

// Only enabled when T is floating point
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
halve(T x) { return x * 0.5; }

halve(10);    // selects integral version
halve(3.14);  // selects float version
```

C++14 helper: `std::enable_if_t<cond, T>` instead of `typename std::enable_if<cond, T>::type`.

### Concepts (C++20)

Replace SFINAE with readable constraints:

```cpp
#include <concepts>

// Concept definition
template<typename T>
concept Numeric = std::is_integral_v<T> || std::is_floating_point_v<T>;

// Usage
template<Numeric T>
T square(T x) { return x * x; }

// Requires clause
template<typename T>
requires std::totally_ordered<T>
T clamp(T val, T lo, T hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

// Abbreviated function template (C++20)
auto add(std::integral auto a, std::integral auto b) { return a + b; }
```

---

## 10. STL Containers

### Container Comparison Table

| Container | Internal Structure | Access | Insert/Delete | Search | When to use |
|---|---|---|---|---|---|
| `vector<T>` | Dynamic array | O(1) random | O(1) amortized back; O(n) middle | O(n) | Default sequence; cache-friendly |
| `deque<T>` | Array of fixed-size chunks | O(1) random | O(1) front and back; O(n) middle | O(n) | Queue with random access |
| `list<T>` | Doubly-linked list | O(n) | O(1) at iterator | O(n) | Frequent insert/delete in middle |
| `forward_list<T>` | Singly-linked list | O(n) | O(1) after iterator | O(n) | Memory-constrained linked list |
| `array<T,N>` | Fixed-size C array | O(1) | Not applicable | O(n) | Fixed size known at compile time |
| `map<K,V>` | Red-black tree | O(log n) | O(log n) | O(log n) | Ordered key-value store |
| `multimap<K,V>` | Red-black tree | O(log n) | O(log n) | O(log n) | Multiple values per key, ordered |
| `set<T>` | Red-black tree | O(log n) | O(log n) | O(log n) | Ordered unique elements |
| `multiset<T>` | Red-black tree | O(log n) | O(log n) | O(log n) | Ordered with duplicates |
| `unordered_map<K,V>` | Hash table | O(1) avg | O(1) avg | O(1) avg | Fast key-value; no ordering needed |
| `unordered_set<T>` | Hash table | O(1) avg | O(1) avg | O(1) avg | Fast membership test |
| `stack<T>` | deque adapter | Top: O(1) | Push/pop: O(1) | Not applicable | LIFO |
| `queue<T>` | deque adapter | Front/back: O(1) | Push/pop: O(1) | Not applicable | FIFO |
| `priority_queue<T>` | Binary heap (in vector) | Top: O(1) | Push/pop: O(log n) | Not applicable | Max element always at top |
| `string` | Dynamic char array | O(1) | O(1) append; O(n) insert | O(n) / O(nm) | Text; SSO on most implementations |

### Red-Black Tree Internals (map, set)

- Self-balancing BST: guaranteed O(log n) for insert, delete, search.
- Each node stores: key, value, color (red/black), left, right, parent pointers.
- Tree height ≤ 2·log₂(n+1).
- Iteration in order (inorder traversal) → sorted output.
- `lower_bound`, `upper_bound`, `equal_range` all O(log n).

### Hash Table Internals (unordered_map, unordered_set)

- Array of buckets; each bucket is a linked list (open chaining in most STL implementations).
- **Load factor** = size / bucket_count. When load_factor > max_load_factor (default 1.0), **rehash** doubles bucket count.
- Rehashing: O(n) amortized across all insertions.
- Custom hash: specialize `std::hash<T>` or provide hash functor.

```cpp
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};
std::unordered_map<std::pair<int,int>, int, PairHash> mp;
```

### Key API Examples

```cpp
#include <vector>
#include <map>
#include <unordered_map>
#include <set>

// vector
std::vector<int> v = {3, 1, 4};
v.push_back(5);
v.emplace_back(9);   // construct in-place (no copy)
v.reserve(100);      // preallocate capacity without changing size
v.erase(v.begin() + 1);

// map
std::map<std::string, int> freq;
freq["hello"]++;
freq.count("world");          // 0 or 1
auto it = freq.find("hello"); // O(log n)
freq.emplace("cpp", 42);

// unordered_map
std::unordered_map<int, int> cache;
cache[5] = 25;
cache.reserve(1000);      // reserve buckets, reduce rehashes
cache.max_load_factor(0.7); // rehash earlier

// set
std::set<int> s = {5, 1, 3};
auto lb = s.lower_bound(3); // first element >= 3
auto ub = s.upper_bound(3); // first element > 3
```

---

## 11. STL Algorithms & Iterators

### Iterator Categories

| Category | Operations | Examples |
|---|---|---|
| Input | `++`, `*` (read), `==`, `!=` | `istream_iterator` |
| Output | `++`, `*` (write) | `ostream_iterator`, `back_inserter` |
| Forward | Input + multi-pass | `forward_list::iterator` |
| Bidirectional | Forward + `--` | `list::iterator`, `map::iterator` |
| Random Access | Bidirectional + `+`, `-`, `[]`, `<` | `vector::iterator`, raw pointer |
| Contiguous (C++20) | Random Access + guaranteed contiguous memory | `vector::iterator` |

### Key Algorithms

```cpp
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>

std::vector<int> v = {5, 1, 3, 2, 4};

// Sort
std::sort(v.begin(), v.end());                    // ascending O(n log n)
std::sort(v.begin(), v.end(), std::greater<int>()); // descending
std::stable_sort(v.begin(), v.end());              // preserves relative order

// Search
auto it = std::find(v.begin(), v.end(), 3);
auto it2 = std::find_if(v.begin(), v.end(), [](int x){ return x > 3; });
int cnt = std::count_if(v.begin(), v.end(), [](int x){ return x % 2 == 0; });

// Binary search (requires sorted range)
auto lb = std::lower_bound(v.begin(), v.end(), 3); // first >= 3
auto ub = std::upper_bound(v.begin(), v.end(), 3); // first > 3
bool found = std::binary_search(v.begin(), v.end(), 3);

// Transform
std::vector<int> squares(v.size());
std::transform(v.begin(), v.end(), squares.begin(), [](int x){ return x*x; });

// Accumulate / reduce
int total = std::accumulate(v.begin(), v.end(), 0);
int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());

// C++17 reduce (parallelizable)
int total2 = std::reduce(v.begin(), v.end());

// Partition
auto pivot = std::partition(v.begin(), v.end(), [](int x){ return x < 3; });
// [begin, pivot) = elements < 3; [pivot, end) = elements >= 3

// Min/max
auto [mn, mx] = std::minmax_element(v.begin(), v.end());

// Copy and fill
std::vector<int> dest(5);
std::copy(v.begin(), v.end(), dest.begin());
std::fill(dest.begin(), dest.end(), 0);
std::iota(dest.begin(), dest.end(), 1); // {1, 2, 3, 4, 5}

// Remove-erase idiom
v.erase(std::remove(v.begin(), v.end(), 3), v.end()); // remove all 3s
```

### Ranges (C++20)

```cpp
#include <ranges>
#include <algorithm>

std::vector<int> v = {1, 2, 3, 4, 5, 6};

// Composable pipelines with | operator
auto result = v
    | std::views::filter([](int x){ return x % 2 == 0; })
    | std::views::transform([](int x){ return x * x; });
// result is lazy: {4, 16, 36}

// Range algorithms (don't need begin/end)
std::ranges::sort(v);
auto it = std::ranges::find(v, 3);
```

### Custom Comparators

```cpp
struct Person { std::string name; int age; };

std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}};
std::sort(people.begin(), people.end(),
    [](const Person& a, const Person& b){ return a.age < b.age; });

// With map: custom comparison
auto cmp = [](const std::string& a, const std::string& b){
    return a.size() < b.size();
};
std::map<std::string, int, decltype(cmp)> m(cmp);
```

---

## 12. Lambdas

### Syntax Breakdown

```cpp
[capture](parameters) mutable -> return_type { body }
//  1        2          3           4            5

// 1. Capture: what from enclosing scope to capture
// 2. Parameters: like function params
// 3. mutable: allows modifying captured-by-value vars
// 4. Return type: usually deduced, explicit when needed
// 5. Body: the function body
```

### Capture Modes

| Capture | Meaning |
|---|---|
| `[]` | Capture nothing |
| `[=]` | Copy all used local variables |
| `[&]` | Reference all used local variables |
| `[x]` | Copy `x` only |
| `[&x]` | Reference `x` only |
| `[=, &x]` | Copy all, but reference `x` |
| `[&, x]` | Reference all, but copy `x` |
| `[this]` | Capture `this` pointer (access class members) |
| `[*this]` | Copy the entire object (C++17) |

```cpp
int a = 1, b = 2;

auto add   = [=]() { return a + b; };         // copies a, b
auto addR  = [&]() { return a + b; };         // references a, b
auto addX  = [a, &b]() { return a + b; };     // copy a, ref b

auto inc   = [a]() mutable { return ++a; };   // mutable copy
// a in enclosing scope unchanged after inc()
```

### Mutable Lambdas

By default, captured-by-value vars are `const` inside the lambda. `mutable` removes this:

```cpp
int counter = 0;
auto tick = [counter]() mutable { return ++counter; }; // ok
auto read = [counter]()         { return counter; };   // ok (read-only)
auto fail = [counter]()         { return ++counter; }; // error
```

### `std::function` vs `auto`

```cpp
// auto: zero overhead (compiler knows exact type)
auto lambda = [](int x){ return x * 2; };

// std::function: type-erased, heap allocation possible, slower
std::function<int(int)> f = [](int x){ return x * 2; };
```

Prefer `auto`. Use `std::function` only when you need:
- Store lambdas of different types in a container
- Callback stored as a class member
- Pass callback through a non-template API

### Generic Lambdas (C++14)

```cpp
auto identity = [](auto x){ return x; }; // works with any type
identity(5);
identity(3.14);
identity(std::string("hi"));

// Variadic generic lambda
auto print_all = [](auto... args){ ((std::cout << args << ' '), ...); };
print_all(1, 2.0, "three");
```

---

## 13. Concurrency

### `std::thread`

```cpp
#include <thread>
#include <iostream>

void worker(int id) {
    std::cout << "Thread " << id << "\n";
}

int main() {
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);
    t1.join();   // wait for t1 to finish
    t2.join();
    // join or detach MUST be called before thread destructor
    // detach(): let it run independently (dangerous: may outlive data)
}
```

### Data Races and UB

A **data race** occurs when two threads access the same memory location concurrently, at least one writes, and no synchronization exists. Data races are **undefined behavior** in C++ — the program is incorrect, not just possibly slow.

```cpp
int counter = 0;
void bad() { ++counter; }  // read-modify-write: 3 steps, not atomic

// Two threads calling bad() concurrently = data race = UB
```

### Mutex and Lock Guards

```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void safe_increment() {
    std::lock_guard<std::mutex> lock(mtx); // RAII: locks on construct, unlocks on destruct
    ++counter;
}

// unique_lock: can unlock early, needed for condition_variable
void with_unique_lock() {
    std::unique_lock<std::mutex> lock(mtx);
    ++counter;
    lock.unlock();   // release early
    // ... do something without holding lock ...
}
```

### `std::atomic`

For simple types, atomic operations avoid mutex overhead:

```cpp
#include <atomic>

std::atomic<int> counter{0};

void atomic_increment() {
    ++counter;        // atomic: no data race, no mutex needed
    counter.fetch_add(1, std::memory_order_relaxed);  // explicit ordering
}
```

Memory ordering (simplified):

| Ordering | Guarantee |
|---|---|
| `memory_order_relaxed` | Only atomicity; no ordering with other ops |
| `memory_order_acquire` | No reads/writes before this load can move after it |
| `memory_order_release` | No reads/writes after this store can move before it |
| `memory_order_seq_cst` | Full sequential consistency (default) |

### `std::async` and `std::future`

```cpp
#include <future>
#include <numeric>
#include <vector>

int sum_range(const std::vector<int>& v, int lo, int hi) {
    return std::accumulate(v.begin()+lo, v.begin()+hi, 0);
}

int main() {
    std::vector<int> v(1000, 1);
    auto fut = std::async(std::launch::async, sum_range, std::cref(v), 0, 500);
    int second_half = sum_range(v, 500, 1000);
    int first_half  = fut.get();   // blocks until async completes
    // first_half + second_half = 1000
}
```

### Deadlock

Deadlock occurs when two threads each hold a lock the other needs:

```
Thread A: lock(mtx1), wait for mtx2
Thread B: lock(mtx2), wait for mtx1
→ Both wait forever
```

Prevention strategies:
1. **Lock ordering**: always acquire multiple mutexes in the same order globally.
2. `std::lock(m1, m2)`: locks both atomically, avoiding deadlock.
3. `std::scoped_lock<M1,M2>(m1, m2)` (C++17): RAII wrapper for multiple mutexes.
4. `try_lock`: non-blocking attempt; retry or back off on failure.

```cpp
std::mutex m1, m2;

void safe() {
    std::scoped_lock lock(m1, m2);  // C++17: deadlock-free
    // use both resources
}
```

---

## 14. Exception Handling

### try / catch / throw

```cpp
#include <stdexcept>
#include <string>

double divide(double a, double b) {
    if (b == 0.0) throw std::invalid_argument("Division by zero");
    return a / b;
}

int main() {
    try {
        double r = divide(10.0, 0.0);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Unknown std error: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown error\n";
    }
}
```

### Exception Safety Levels

| Level | Guarantee | Description |
|---|---|---|
| No-throw | `noexcept` | Function never throws; operation always succeeds |
| Strong | "commit or rollback" | If exception thrown, state is unchanged (as if call never happened) |
| Basic | Valid state | If exception thrown, object is in some valid state; invariants preserved |
| None | No guarantee | Object may be in broken state after exception |

### `noexcept`

```cpp
void never_throws() noexcept { }     // promises no exception
void maybe_throws() noexcept(false) { } // may throw (default)

// noexcept(expr): conditional
template<typename T>
void swap(T& a, T& b) noexcept(noexcept(T(std::move(a)))) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}
```

Mark destructors, move operations, and swap `noexcept` whenever possible. STL uses this for optimization (e.g., vector reallocation uses move only if `noexcept`).

### `std::exception` Hierarchy

```
std::exception
├── std::logic_error
│   ├── std::invalid_argument
│   ├── std::domain_error
│   ├── std::length_error
│   └── std::out_of_range
└── std::runtime_error
    ├── std::range_error
    ├── std::overflow_error
    └── std::underflow_error
```

Custom exceptions:

```cpp
class AppError : public std::runtime_error {
public:
    int code;
    AppError(const std::string& msg, int code)
        : std::runtime_error(msg), code(code) {}
};
```

### RAII + Exceptions

RAII guarantees cleanup even when exceptions are thrown:

```cpp
void process(const std::string& path) {
    File f(path);          // constructor opens file
    parse(f);              // throws? f's destructor still closes the file
    transform(f);
}   // f.~File() called here regardless of exceptions
```

This is why C++ does not need `finally` blocks — RAII + destructors provide the same guarantee more robustly.

---

## 15. Modern C++ Features (C++11 through C++20)

| Feature | Version | Replaces / Improves | Example |
|---|---|---|---|
| `auto` type deduction | C++11 | Verbose type names | `auto it = v.begin();` |
| Range-based for | C++11 | Index-based for | `for (auto& x : v)` |
| `nullptr` | C++11 | `NULL` / `0` | `int* p = nullptr;` |
| `override` / `final` | C++11 | Error-prone virtual override | `void foo() override;` |
| Move semantics / `&&` | C++11 | Copy everything | `T(T&& o) noexcept` |
| `std::move` | C++11 | Manual cast | `vec.push_back(std::move(s));` |
| Lambdas | C++11 | Function objects / functors | `[x](int y){ return x+y; }` |
| `std::unique_ptr` | C++11 | Raw `new`/`delete` | `make_unique<T>(args)` |
| `std::shared_ptr` | C++11 | Manual ref counting | `make_shared<T>(args)` |
| `constexpr` | C++11 | Runtime only | `constexpr int SZ = 1024;` |
| Initializer lists | C++11 | Manual element insertion | `vector<int> v = {1,2,3};` |
| Variadic templates | C++11 | va_args | `template<typename...Args>` |
| `static_assert` | C++11 | Runtime assert | `static_assert(sizeof(int)==4)` |
| `std::thread` | C++11 | Platform threads (pthreads) | `std::thread t(fn);` |
| `std::atomic` | C++11 | Volatile hacks | `atomic<int> counter{0};` |
| `[[deprecated]]` | C++14 | Comments | `[[deprecated("use bar")]] void foo()` |
| Generic lambdas | C++14 | Template functors | `[](auto x){ return x; }` |
| `make_unique` | C++14 | `unique_ptr<T>(new T)` | `make_unique<Foo>(args)` |
| Structured bindings | C++17 | `std::tie` | `auto [k,v] = *map_iter;` |
| `if constexpr` | C++17 | SFINAE / enable_if hacks | `if constexpr (is_integral_v<T>)` |
| `std::optional` | C++17 | Pointer-as-optional / sentinel | `optional<int> find(...)` |
| `std::variant` | C++17 | Tagged unions | `variant<int,string> v;` |
| `std::string_view` | C++17 | `const string&` / `const char*` | `void fn(string_view sv)` |
| Parallel algorithms | C++17 | OpenMP pragmas | `sort(execution::par, v.begin(),v.end())` |
| Fold expressions | C++17 | Recursive variadic unpacking | `(args + ...)` |
| Concepts | C++20 | SFINAE / enable_if | `template<Integral T>` |
| Ranges | C++20 | begin/end iterators | `v \| views::filter(f) \| views::transform(g)` |
| Coroutines | C++20 | Callback chains / manual state machines | `co_await`, `co_yield`, `co_return` |
| `std::format` | C++20 | printf / stringstream | `format("x={}", x)` |
| `consteval` | C++20 | `constexpr` (not always compile-time) | `consteval int sq(int x)` |
| `std::span` | C++20 | Passing array+size separately | `void fn(span<int> data)` |
| Three-way comparison | C++20 | 6 separate comparison operators | `auto operator<=>(const T&) const = default;` |

---

## 16. Memory Model & Undefined Behavior

### Sequence Points and UB

Undefined behavior means the compiler is allowed to do **anything** — including generating code that looks correct, crashes, or produces garbage only in release mode.

Common UB in C++:

| UB | Example |
|---|---|
| Signed integer overflow | `INT_MAX + 1` |
| Null pointer dereference | `int* p = nullptr; *p = 1;` |
| Array out of bounds | `int a[5]; a[10] = 0;` |
| Use after free | `delete p; *p = 1;` |
| Data race | Two threads write same var without sync |
| Uninitialized read | `int x; return x;` |
| Invalid pointer arithmetic | Pointer past one-past-end |
| `reinterpret_cast` type punning (usually) | Read float bits as int |

Detect with `-fsanitize=address,undefined` (ASan, UBSan).

### `const` Correctness

```cpp
const int x = 5;           // x cannot be modified
const int* p = &x;         // pointer to const int (value immutable)
int* const q = &y;         // const pointer to int (pointer immutable)
const int* const r = &x;   // both immutable

void display(const std::vector<int>& v) { // v not modified
    for (int x : v) std::cout << x;
}
```

`const` member functions:

```cpp
class Foo {
    int val;
    mutable int cache;  // can be modified even in const methods

    int get() const {       // const: can be called on const Foo
        cache = val * 2;    // ok: mutable
        // val = 1;         // error: val is not mutable
        return val;
    }
};
```

### `constexpr` and `consteval`

```cpp
constexpr int factorial(int n) {   // evaluated at compile time if args are constexpr
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr int f5 = factorial(5);   // compile-time: 120
int n = 6;
int f6 = factorial(n);             // runtime: ok for constexpr

consteval int sq(int n) {          // C++20: MUST be compile-time
    return n * n;
}
// sq(n);  // error if n is not constexpr
constexpr int s = sq(7);           // ok: 49 at compile time
```

### Attributes

| Attribute | Meaning | Example |
|---|---|---|
| `[[nodiscard]]` | Warn if return value is discarded | `[[nodiscard]] int errorCode()` |
| `[[maybe_unused]]` | Suppress unused warning | `[[maybe_unused]] int debug_val = 0;` |
| `[[deprecated("msg")]]` | Warn when used | `[[deprecated]] void old_api()` |
| `[[likely]]` / `[[unlikely]]` | Branch prediction hint | `if (x > 0) [[likely]] { ... }` |
| `[[noreturn]]` | Function never returns | `[[noreturn]] void terminate()` |
| `[[fallthrough]]` | Intentional switch fallthrough | `case 1: [[fallthrough]];` |

---

## 17. Interview Q&A — 20 Questions

**Q1. What is the difference between a pointer and a reference?**

A reference is an alias for an existing object: it must be initialized, cannot be null, cannot be rebound to another object. A pointer can be null, can be rebound, and requires `*` to dereference. Prefer references for function parameters (no null, no ownership transfer); use pointers when null is meaningful or ownership semantics are needed.

Follow-up: Can a reference be used to change the value it refers to? Yes — unless it's `const`.

---

**Q2. What is RAII?**

Resource Acquisition Is Initialization: a resource (memory, file, mutex) is acquired in a constructor and released in the destructor. Because destructors run when objects go out of scope — including during stack unwinding due to exceptions — resources are always released. Smart pointers, `lock_guard`, `fstream` all implement RAII. Manual `try/catch`+`finally` patterns from other languages are replaced by RAII in C++.

Follow-up: What happens if a constructor throws? All fully constructed member subobjects have their destructors called. The object's own destructor is NOT called (the object was never fully constructed).

---

**Q3. What is the Rule of 5?**

If a class manages a resource and you define any of: destructor, copy constructor, copy assignment, move constructor, move assignment — you should define all five. The reason: defining a destructor implies manual resource management, so the compiler-generated copy/move may do the wrong thing (shallow copy of a pointer, no null-out on move). The Rule of 0 is preferred: use RAII members so the compiler generates correct versions for free.

Follow-up: What does the compiler generate for copy constructor by default? Memberwise copy — copies each member. For a raw pointer, that means two objects point to the same memory (double-free on destruction).

---

**Q4. What is the difference between `delete` and `delete[]`?**

`delete` calls the destructor of one object and frees the memory. `delete[]` calls the destructor on each element of the array, then frees the memory. Mismatching (using `delete` on `new[]` or vice versa) is undefined behavior. Smart pointers handle this automatically: `unique_ptr<T[]>` uses `delete[]`.

Follow-up: Why does `delete[]` need to know the array size? It stores it in the heap metadata adjacent to the allocation — implementation-defined location.

---

**Q5. What is virtual dispatch and how does it work?**

Each class with virtual functions has a **vtable**: a static array of function pointers for each virtual method. Each object with virtual functions has a hidden **vptr** (usually at offset 0) pointing to the class's vtable. A virtual call dereferences the vptr, indexes into the vtable, and calls the function pointer. Cost: one extra indirection. Benefit: runtime polymorphism — the correct derived-class method is called through a base-class pointer.

Follow-up: Why should a polymorphic base class have a virtual destructor? Without it, `delete base_ptr` only calls the base destructor, not the derived destructor — resource leak and UB.

---

**Q6. What is the difference between `new` and `malloc`?**

| | `new` | `malloc` |
|---|---|---|
| Type-safe | Yes | No (returns `void*`) |
| Calls constructor | Yes | No |
| Throws on failure | `std::bad_alloc` | Returns `nullptr` |
| Counterpart | `delete` | `free` |
| Size calculation | Automatic | Manual `sizeof` |

Never mix them: `free(new T)` and `delete malloc(n)` are both UB.

Follow-up: What is placement new? `new(ptr) T(args)` — constructs a T at an already-allocated address. Useful for custom allocators. Corresponding destruction: call `ptr->~T()` explicitly.

---

**Q7. Explain `std::move` and `std::forward`.**

`std::move` is an unconditional cast to an rvalue reference: `static_cast<T&&>(t)`. It does not move data; it signals that the object's resources may be stolen. The actual transfer happens in the move constructor or move assignment operator.

`std::forward<T>(t)` is a conditional cast: if `T` is an lvalue reference type, it returns an lvalue reference; if `T` is a non-reference type (rvalue), it returns an rvalue reference. This is "perfect forwarding": a template function can pass arguments to another function preserving their value category.

Follow-up: What happens after `std::move(v)` where v is a vector? v is in a valid but unspecified state (likely empty). You can reassign it but should not read from it.

---

**Q8. What is `shared_ptr` reference counting and when can it leak?**

`shared_ptr` maintains a control block with a strong ref count (number of `shared_ptr` owners) and a weak ref count (number of `weak_ptr` observers). The managed object is destroyed when the strong count reaches 0. The control block is destroyed when both counts reach 0.

Cycles cause leaks: if A holds a `shared_ptr<B>` and B holds a `shared_ptr<A>`, neither count reaches 0. Solution: break the cycle with `weak_ptr` for one direction.

Follow-up: Is `shared_ptr` thread-safe? The control block reference count operations are atomic (thread-safe). The managed object itself is NOT protected by `shared_ptr` — concurrent writes to the object still require external synchronization.

---

**Q9. What is template metaprogramming (TMP)?**

Using the C++ template system to perform computations at compile time. Templates are Turing-complete (proven). Modern C++ prefers `constexpr`/`consteval` over TMP for readability. TMP is still used for type manipulation: `std::tuple`, `std::variant`, `std::enable_if`, type traits (`std::is_integral`, `std::remove_reference`).

Follow-up: What is `std::declval`? A way to get a value of type `T` without constructing it, for use in `decltype` expressions: `decltype(std::declval<T>().method())`.

---

**Q10. What is SFINAE?**

Substitution Failure Is Not An Error: when the compiler substitutes template arguments and encounters an error in the function signature (not the body), it removes that overload from consideration instead of issuing an error. This allows `enable_if`-based conditional overloads. In C++20, Concepts replace most SFINAE patterns with more readable code.

Follow-up: Where exactly does SFINAE apply? Only in the **immediate context** of template argument substitution — not deep inside function bodies. An error inside the body is a hard error.

---

**Q11. What is the difference between `static_cast`, `dynamic_cast`, `reinterpret_cast`, and `const_cast`?**

| Cast | Use | Safety |
|---|---|---|
| `static_cast` | Known, checked at compile time (numeric, up/downcasting) | Compile-time check; no runtime overhead |
| `dynamic_cast` | Polymorphic downcasting | Runtime RTTI check; returns `nullptr` or throws `bad_cast` |
| `reinterpret_cast` | Bit-level reinterpretation | Almost always dangerous; UB in many cases |
| `const_cast` | Remove const qualification | Safe only if underlying object is non-const |

Follow-up: When does `dynamic_cast` fail? If the pointer does not actually point to the target type (or a derived type). Requires at least one virtual function in the hierarchy.

---

**Q12. What are the storage duration classes in C++?**

| Storage | Keyword | Lifetime |
|---|---|---|
| Automatic | (local variable) | Block scope: construct at definition, destruct at `}` |
| Static | `static` / global | Program lifetime: constructed before main, destroyed after |
| Thread-local | `thread_local` | Thread lifetime |
| Dynamic | `new` / smart ptrs | Controlled by programmer |

Follow-up: What is the static initialization order fiasco? Globals in different translation units have unspecified initialization order. If one global depends on another, the dependency may not be initialized yet. Fix: wrap in a function-local `static`.

---

**Q13. What is the difference between `emplace_back` and `push_back`?**

`push_back(value)` copies or moves an already-constructed object into the container. `emplace_back(args...)` constructs the object in-place using the provided arguments, potentially avoiding a temporary. For trivially copyable types the difference is negligible; for types with expensive constructors or explicit constructors, `emplace_back` is preferable.

Follow-up: Is `emplace_back` always faster? Not necessarily. For small types it may be identical. `push_back` is safer in some generic contexts because it does not accidentally call explicit constructors.

---

**Q14. Explain `std::optional`, `std::variant`, and `std::any`.**

| Type | Holds | Use case |
|---|---|---|
| `optional<T>` | T or nothing | Function that may not return a value |
| `variant<T1,T2,...>` | One of the listed types | Type-safe tagged union |
| `any` | Any type (type-erased) | When the type is truly unknown at compile time |

```cpp
std::optional<int> parse(const std::string& s) {
    try { return std::stoi(s); } catch(...) { return std::nullopt; }
}
if (auto v = parse("42")) std::cout << *v;

std::variant<int, std::string> v = "hello";
std::visit([](auto&& val){ std::cout << val; }, v);
```

Follow-up: How does `std::variant` avoid heap allocation? It stores all possible types in a union-like buffer of size `max(sizeof(Ti))`, plus a discriminant tag.

---

**Q15. What is a vtable and when is it NOT used?**

The vtable is used only for `virtual` function calls through a pointer or reference to a base class. It is NOT used for: non-virtual functions (resolved at compile time), final classes (compiler can devirtualize), static methods, calls on a known concrete type (compiler sees exact type). Modern compilers (GCC, Clang) perform **devirtualization** when the concrete type can be proven at compile time.

Follow-up: How much memory does the vptr add? Typically 8 bytes on a 64-bit system, per object. Only one vptr per object regardless of how many virtual functions exist.

---

**Q16. What is memory alignment and why does it matter?**

Hardware requires certain types to be at addresses that are multiples of their size (e.g., `double` at 8-byte boundaries). Misaligned access is UB in C++ and may fault on some architectures. The compiler inserts padding in structs to satisfy alignment. `alignas(N)` forces alignment; `alignof(T)` queries it.

```cpp
struct A { char c; int i; };  // sizeof = 8 (3 bytes padding after c)
struct B { int i; char c; };  // sizeof = 8 (3 bytes padding after c)
```

Follow-up: How do you minimize struct size? Sort members largest to smallest — compiler adds less padding.

---

**Q17. Explain lock-free programming and when to use it.**

Lock-free programming uses atomic operations (compare-and-swap, fetch-add) instead of mutexes. Guarantees: at least one thread makes progress at any time (no deadlock). More complex to write correctly. Use `std::atomic` for simple counters, flags, pointers. Use lock-free data structures only when profiling shows mutex contention is the actual bottleneck. Mutexes are easier to reason about and fast enough for most cases.

Follow-up: Is `std::atomic<T>` always lock-free? Only if `T` fits in a CPU word and is trivially copyable. Check with `atomic<T>::is_always_lock_free` or `atomic<T>::is_lock_free()`.

---

**Q18. What is copy elision and RVO?**

Copy elision: compiler omits a copy/move constructor call, constructing the object directly in the destination. Return Value Optimization (RVO): when a function returns an unnamed temporary, the compiler constructs it directly in the caller's return slot. Named RVO (NRVO): same for named local variables. In C++17, RVO is mandatory for unnamed temporaries (guaranteed copy elision). NRVO is an optional optimization.

Follow-up: How do you prevent RVO? Return `std::move(localVar)` — this explicitly creates an xvalue, which prevents the compiler from applying NRVO (because you changed the return expression). Don't do this.

---

**Q19. What is the difference between `struct` and `class` in C++?**

Only the default access specifier: `struct` defaults to `public`, `class` defaults to `private`. Both can have constructors, destructors, virtual functions, inheritance, templates. Convention: `struct` for plain data (POD-like); `class` for types with invariants and encapsulation. Inheritance default also differs: `struct B : A` is `public` inheritance; `class B : A` is `private` inheritance.

Follow-up: What is a POD type? Plain Old Data — trivially copyable and standard-layout. Can be memcpy'd, used in C-compatible APIs. `std::is_pod<T>` (deprecated in C++20; use `std::is_trivial` and `std::is_standard_layout`).

---

**Q20. What are coroutines (C++20)?**

Functions that can suspend execution and resume later. Use `co_await`, `co_yield`, `co_return`. Suspend points allow the function to return control to the caller without losing state. Applications: async I/O (suspend while waiting for data), generators (lazy sequences), cooperative multitasking. The compiler transforms the coroutine body into a state machine. Requires a promise type and a coroutine handle type.

```cpp
#include <coroutine>
#include <generator>  // C++23 / custom

std::generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i)
        co_yield i;
}
```

Follow-up: How are coroutines different from threads? Coroutines are cooperative (explicit suspend), single-threaded by default, and have near-zero overhead vs threads (no OS context switch). Threads are preemptive and require synchronization.

---

## 18. Solved Practice Problems

---

### Problem 1: Generic Stack Using Templates

**Problem**: Implement a generic `Stack<T>` class with `push`, `pop`, `top`, `empty`, `size`.

**Approach**: Wrap `std::vector<T>` — internal storage provides dynamic sizing and correct copy/move semantics automatically.

```cpp
#include <vector>
#include <stdexcept>
#include <iostream>

template<typename T>
class Stack {
public:
    void push(const T& val) { data_.push_back(val); }
    void push(T&& val)      { data_.push_back(std::move(val)); }

    template<typename... Args>
    void emplace(Args&&... args) { data_.emplace_back(std::forward<Args>(args)...); }

    void pop() {
        if (data_.empty()) throw std::underflow_error("Stack is empty");
        data_.pop_back();
    }

    T& top() {
        if (data_.empty()) throw std::underflow_error("Stack is empty");
        return data_.back();
    }
    const T& top() const {
        if (data_.empty()) throw std::underflow_error("Stack is empty");
        return data_.back();
    }

    bool empty() const { return data_.empty(); }
    std::size_t size() const { return data_.size(); }

private:
    std::vector<T> data_;
};

int main() {
    Stack<int> s;
    s.push(1); s.push(2); s.push(3);
    while (!s.empty()) {
        std::cout << s.top() << "\n";
        s.pop();
    }
}
```

**Complexity**: push O(1) amortized, pop O(1), top O(1).

---

### Problem 2: Simplified `unique_ptr` from Scratch

**Problem**: Implement a `UniquePtr<T>` with sole ownership, move semantics, no copy.

**Approach**: Store raw pointer. Delete copy operations. Transfer pointer in move operations. Call destructor in destructor.

```cpp
#include <utility>
#include <iostream>

template<typename T>
class UniquePtr {
public:
    explicit UniquePtr(T* p = nullptr) : ptr_(p) {}

    ~UniquePtr() { delete ptr_; }

    UniquePtr(const UniquePtr&)            = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T& operator*()  const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get()        const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    T* release() { T* p = ptr_; ptr_ = nullptr; return p; }
    void reset(T* p = nullptr) { delete ptr_; ptr_ = p; }

private:
    T* ptr_;
};

template<typename T, typename... Args>
UniquePtr<T> makeUnique(Args&&... args) {
    return UniquePtr<T>(new T(std::forward<Args>(args)...));
}

struct Foo { int x; Foo(int x) : x(x) { std::cout << "Foo(" << x << ")\n"; } ~Foo() { std::cout << "~Foo\n"; } };

int main() {
    auto p = makeUnique<Foo>(42);
    std::cout << p->x << "\n";
    auto q = std::move(p);
    // p is null now; q owns Foo
}
```

**Complexity**: All operations O(1). No heap overhead beyond the object itself.

---

### Problem 3: Linked List with RAII

**Problem**: Implement a singly-linked list class with proper RAII — no memory leak through exceptions or normal destruction.

```cpp
#include <iostream>
#include <memory>

template<typename T>
class LinkedList {
    struct Node {
        T val;
        std::unique_ptr<Node> next;
        Node(T v) : val(std::move(v)) {}
    };

    std::unique_ptr<Node> head_;
    std::size_t size_ = 0;

public:
    void pushFront(T val) {
        auto node = std::make_unique<Node>(std::move(val));
        node->next = std::move(head_);
        head_ = std::move(node);
        ++size_;
    }

    void reverse() {
        std::unique_ptr<Node> prev;
        auto curr = std::move(head_);
        while (curr) {
            auto next = std::move(curr->next);
            curr->next = std::move(prev);
            prev = std::move(curr);
            curr = std::move(next);
        }
        head_ = std::move(prev);
    }

    void print() const {
        for (Node* n = head_.get(); n; n = n->next.get())
            std::cout << n->val << " -> ";
        std::cout << "null\n";
    }

    std::size_t size() const { return size_; }
};
// Destructor: unique_ptr chain unwinds, each node deleted automatically

int main() {
    LinkedList<int> list;
    list.pushFront(3); list.pushFront(2); list.pushFront(1);
    list.print();   // 1 -> 2 -> 3 -> null
    list.reverse();
    list.print();   // 3 -> 2 -> 1 -> null
}
```

**Complexity**: pushFront O(1), reverse O(n).

---

### Problem 4: `std::vector` from Scratch

**Problem**: Implement a dynamic array with `push_back`, `operator[]`, `size`, `capacity`, `reserve`.

```cpp
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <utility>

template<typename T>
class Vector {
public:
    Vector() = default;

    ~Vector() {
        for (std::size_t i = 0; i < size_; ++i) data_[i].~T();
        ::operator delete(data_);
    }

    Vector(const Vector& other) {
        reserve(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i)
            new (data_ + i) T(other.data_[i]);
        size_ = other.size_;
    }

    Vector(Vector&& other) noexcept
        : data_(other.data_), size_(other.size_), cap_(other.cap_) {
        other.data_ = nullptr; other.size_ = 0; other.cap_ = 0;
    }

    void push_back(const T& val) { emplace_back(val); }
    void push_back(T&& val)      { emplace_back(std::move(val)); }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ == cap_) grow();
        new (data_ + size_) T(std::forward<Args>(args)...);
        ++size_;
    }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("index out of range");
        return data_[i];
    }

    std::size_t size()     const { return size_; }
    std::size_t capacity() const { return cap_; }
    bool empty()           const { return size_ == 0; }

    void reserve(std::size_t newCap) {
        if (newCap <= cap_) return;
        T* newData = static_cast<T*>(::operator new(newCap * sizeof(T)));
        for (std::size_t i = 0; i < size_; ++i) {
            new (newData + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        ::operator delete(data_);
        data_ = newData;
        cap_  = newCap;
    }

    T* begin() { return data_; }
    T* end()   { return data_ + size_; }

private:
    void grow() { reserve(cap_ == 0 ? 1 : cap_ * 2); }

    T* data_         = nullptr;
    std::size_t size_ = 0;
    std::size_t cap_  = 0;
};

int main() {
    Vector<int> v;
    for (int i = 0; i < 10; ++i) v.push_back(i);
    for (std::size_t i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
    std::cout << "\n";
}
```

**Complexity**: `push_back` O(1) amortized, `operator[]` O(1), `reserve` O(n).

---

### Problem 5: Thread-safe Singleton with `std::call_once`

**Problem**: Implement a singleton that is thread-safe and initialized exactly once.

```cpp
#include <mutex>
#include <memory>
#include <iostream>

class Config {
public:
    static Config& instance() {
        std::call_once(initFlag_, [](){
            instance_.reset(new Config());
        });
        return *instance_;
    }

    void set(const std::string& k, int v) {
        std::lock_guard<std::mutex> lk(mtx_);
        data_[k] = v;
    }

    int get(const std::string& k) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = data_.find(k);
        return it != data_.end() ? it->second : 0;
    }

private:
    Config() = default;

    static std::unique_ptr<Config> instance_;
    static std::once_flag initFlag_;
    mutable std::mutex mtx_;
    std::map<std::string, int> data_;
};

std::unique_ptr<Config> Config::instance_;
std::once_flag Config::initFlag_;

int main() {
    Config::instance().set("threads", 8);
    std::cout << Config::instance().get("threads") << "\n";
}
```

**Notes**: `std::call_once` guarantees the lambda runs exactly once even with concurrent callers. A simpler alternative: function-local `static` (guaranteed thread-safe initialization in C++11).

```cpp
static Config& instance() {
    static Config inst;  // initialized exactly once, thread-safe (C++11)
    return inst;
}
```

---

### Problem 6: Observer Pattern

**Problem**: Implement an event system where multiple observers subscribe to events fired by a subject.

```cpp
#include <functional>
#include <vector>
#include <string>
#include <iostream>

template<typename... Args>
class Event {
public:
    using Handler = std::function<void(Args...)>;

    void subscribe(Handler h) {
        handlers_.push_back(std::move(h));
    }

    void fire(Args... args) const {
        for (auto& h : handlers_) h(args...);
    }

private:
    std::vector<Handler> handlers_;
};

struct Button {
    Event<const std::string&> onClick;
    Event<int, int>           onResize;

    void click(const std::string& label) { onClick.fire(label); }
    void resize(int w, int h)            { onResize.fire(w, h); }
};

int main() {
    Button btn;

    btn.onClick.subscribe([](const std::string& lbl){
        std::cout << "Clicked: " << lbl << "\n";
    });
    btn.onResize.subscribe([](int w, int h){
        std::cout << "Resized to " << w << "x" << h << "\n";
    });

    btn.click("Submit");
    btn.resize(800, 600);
}
```

**Complexity**: subscribe O(1), fire O(n handlers).

---

### Problem 7: LRU Cache

**Problem**: Implement an LRU (Least Recently Used) cache with O(1) get and put.

**Approach**: `unordered_map` from key → list iterator for O(1) lookup. `list` maintains access order (most recent at front). On access, move node to front. On overflow, remove from back.

```cpp
#include <list>
#include <unordered_map>
#include <optional>
#include <iostream>

template<typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(std::size_t capacity) : cap_(capacity) {}

    std::optional<V> get(const K& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return std::nullopt;
        // Move to front (most recently used)
        order_.splice(order_.begin(), order_, it->second);
        return it->second->second;
    }

    void put(const K& key, V value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second->second = std::move(value);
            order_.splice(order_.begin(), order_, it->second);
            return;
        }
        if (order_.size() == cap_) {
            map_.erase(order_.back().first);
            order_.pop_back();
        }
        order_.emplace_front(key, std::move(value));
        map_[key] = order_.begin();
    }

    std::size_t size() const { return order_.size(); }

private:
    std::size_t cap_;
    std::list<std::pair<K, V>> order_;
    std::unordered_map<K, typename std::list<std::pair<K,V>>::iterator> map_;
};

int main() {
    LRUCache<int, std::string> cache(3);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");
    cache.get(1);             // 1 is now most recent
    cache.put(4, "four");     // evicts 2 (least recently used)
    std::cout << (cache.get(2) ? "found" : "evicted") << "\n"; // evicted
    std::cout << *cache.get(1) << "\n";                         // one
}
```

**Complexity**: get O(1), put O(1).

---

### Problem 8: Binary Search Tree

**Problem**: BST with insert, search, inorder traversal.

```cpp
#include <memory>
#include <functional>
#include <iostream>

template<typename T>
class BST {
    struct Node {
        T val;
        std::unique_ptr<Node> left, right;
        Node(T v) : val(std::move(v)) {}
    };

    std::unique_ptr<Node> root_;

    void insert_(std::unique_ptr<Node>& node, T val) {
        if (!node) { node = std::make_unique<Node>(std::move(val)); return; }
        if (val < node->val)      insert_(node->left,  std::move(val));
        else if (val > node->val) insert_(node->right, std::move(val));
        // equal: ignore duplicates
    }

    bool search_(const Node* node, const T& val) const {
        if (!node) return false;
        if (val == node->val) return true;
        return val < node->val ? search_(node->left.get(), val)
                               : search_(node->right.get(), val);
    }

    void inorder_(const Node* node, std::function<void(const T&)> fn) const {
        if (!node) return;
        inorder_(node->left.get(), fn);
        fn(node->val);
        inorder_(node->right.get(), fn);
    }

public:
    void insert(T val) { insert_(root_, std::move(val)); }
    bool search(const T& val) const { return search_(root_.get(), val); }

    void inorder(std::function<void(const T&)> fn) const {
        inorder_(root_.get(), fn);
    }
};

int main() {
    BST<int> tree;
    for (int x : {5, 3, 7, 1, 4, 6, 8}) tree.insert(x);
    tree.inorder([](int x){ std::cout << x << " "; });
    std::cout << "\n";
    std::cout << tree.search(4) << "\n";  // 1
    std::cout << tree.search(9) << "\n";  // 0
}
```

**Complexity**: insert O(h), search O(h), inorder O(n). h = O(log n) balanced, O(n) worst case.

---

### Problem 9: `shared_ptr` Reference Counting from Scratch

**Problem**: Implement `SharedPtr<T>` with reference counting, correct copy/move, destruction when count hits zero.

```cpp
#include <cstddef>
#include <iostream>
#include <utility>

template<typename T>
class SharedPtr {
    struct ControlBlock {
        T* ptr;
        std::size_t refCount;
        ControlBlock(T* p) : ptr(p), refCount(1) {}
    };

    ControlBlock* cb_ = nullptr;

    void release() {
        if (!cb_) return;
        if (--cb_->refCount == 0) {
            delete cb_->ptr;
            delete cb_;
        }
        cb_ = nullptr;
    }

public:
    explicit SharedPtr(T* p = nullptr) {
        if (p) cb_ = new ControlBlock(p);
    }

    SharedPtr(const SharedPtr& other) : cb_(other.cb_) {
        if (cb_) ++cb_->refCount;
    }

    SharedPtr(SharedPtr&& other) noexcept : cb_(other.cb_) {
        other.cb_ = nullptr;
    }

    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {
            release();
            cb_ = other.cb_;
            if (cb_) ++cb_->refCount;
        }
        return *this;
    }

    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {
            release();
            cb_ = other.cb_;
            other.cb_ = nullptr;
        }
        return *this;
    }

    ~SharedPtr() { release(); }

    T& operator*()  const { return *cb_->ptr; }
    T* operator->() const { return cb_->ptr; }
    T* get()        const { return cb_ ? cb_->ptr : nullptr; }
    std::size_t useCount() const { return cb_ ? cb_->refCount : 0; }
    explicit operator bool() const { return cb_ != nullptr; }
};

template<typename T, typename... Args>
SharedPtr<T> makeShared(Args&&... args) {
    return SharedPtr<T>(new T(std::forward<Args>(args)...));
}

struct Obj { int x; Obj(int x) : x(x) { std::cout << "Obj\n"; } ~Obj() { std::cout << "~Obj\n"; } };

int main() {
    auto a = makeShared<Obj>(10);
    {
        auto b = a;  // refCount = 2
        std::cout << b->x << " refs=" << a.useCount() << "\n";
    }  // b destroyed, refCount = 1
    std::cout << "refs=" << a.useCount() << "\n";
}   // a destroyed, refCount = 0 → ~Obj
```

**Complexity**: copy O(1) (atomic increment in real impl), dereference O(1).

---

### Problem 10: Matrix Multiplication with Templates and Operator Overloading

**Problem**: Implement a `Matrix<T, Rows, Cols>` class with `operator*`, `operator[]`, and streaming output.

```cpp
#include <array>
#include <stdexcept>
#include <iostream>
#include <iomanip>

template<typename T, std::size_t R, std::size_t C>
class Matrix {
public:
    Matrix() { for (auto& row : data_) row.fill(T{}); }

    std::array<T, C>& operator[](std::size_t r) { return data_[r]; }
    const std::array<T, C>& operator[](std::size_t r) const { return data_[r]; }

    T& at(std::size_t r, std::size_t c) {
        if (r >= R || c >= C) throw std::out_of_range("Matrix index");
        return data_[r][c];
    }

    static constexpr std::size_t rows() { return R; }
    static constexpr std::size_t cols() { return C; }

    Matrix operator+(const Matrix& rhs) const {
        Matrix result;
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j)
                result[i][j] = data_[i][j] + rhs[i][j];
        return result;
    }

    // Multiply: (R x C) * (C x K) = (R x K)
    template<std::size_t K>
    Matrix<T, R, K> operator*(const Matrix<T, C, K>& rhs) const {
        Matrix<T, R, K> result;
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t k = 0; k < K; ++k)
                for (std::size_t j = 0; j < C; ++j)
                    result[i][k] += data_[i][j] * rhs[j][k];
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j)
                os << std::setw(6) << m[i][j];
            os << "\n";
        }
        return os;
    }

private:
    std::array<std::array<T, C>, R> data_;
};

int main() {
    Matrix<int, 2, 3> A;
    Matrix<int, 3, 2> B;

    // Fill A
    int v = 1;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            A[i][j] = v++;

    // Fill B
    v = 1;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            B[i][j] = v++;

    auto C = A * B;  // 2x3 * 3x2 = 2x2
    std::cout << "A:\n" << A;
    std::cout << "B:\n" << B;
    std::cout << "A*B:\n" << C;
}
```

**Complexity**: multiplication O(R·C·K) — standard cubic. Template parameters enforce size correctness at compile time (wrong sizes = compile error).

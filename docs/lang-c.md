# C Programming — Zero to Systems Expert

---

## 1. Why C Exists

C was created at Bell Labs (1972) by Dennis Ritchie to rewrite Unix in a portable, high-level language that compiles to efficient machine code with no runtime overhead. It sits one thin layer above assembly: every operation maps predictably to CPU instructions, memory is managed manually, and there is no garbage collector, virtual machine, or hidden indirection. This directness makes C the lingua franca of operating systems, embedded firmware, compilers, databases, and any domain where you must control hardware, latency, or binary layout with precision. Every systems programmer must know C because nearly all systems software is either written in C or exposes a C-compatible interface.

---

## 2. Your First C Program

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

**Line-by-line breakdown:**

| Token | Meaning |
|---|---|
| `#include` | Preprocessor directive — pastes another file's contents here |
| `<stdio.h>` | Standard I/O header; declares `printf`, `scanf`, `fopen`, etc. |
| `int main(void)` | Entry point; `int` = return type; `void` = no command-line args |
| `printf(...)` | Writes formatted text to stdout |
| `"\n"` | Newline escape sequence (ASCII 10) |
| `return 0;` | Signals success to the OS (non-zero = error) |

`#include <stdio.h>` does not link code — it only inserts declarations so the compiler knows `printf` exists. The actual implementation lives in `libc`, linked at the end.

`main` must return `int`. The OS reads the exit code; shell scripts check `$?`. Returning `void` from `main` is undefined behavior in standard C.

### Compilation

```bash
gcc -o hello hello.c && ./hello
```

`-o hello` names the output binary. Without it, the default is `a.out`.

### Compilation Pipeline

```
hello.c
   │
   ▼  [Preprocessor — cpp]
   │  Expands #include, #define, #ifdef
   │  Output: hello.i  (pure C text, no directives)
   │
   ▼  [Compiler — cc1]
   │  Parses C, type-checks, generates intermediate representation
   │  Output: hello.s  (assembly text)
   │
   ▼  [Assembler — as]
   │  Converts assembly mnemonics to binary machine code
   │  Output: hello.o  (ELF object file, has unresolved symbols)
   │
   ▼  [Linker — ld]
   │  Combines hello.o + libc.a/libc.so, resolves printf symbol
   │  Output: hello    (executable ELF binary)
```

Inspect intermediate files:

```bash
gcc -E hello.c -o hello.i     # stop after preprocessing
gcc -S hello.c -o hello.s     # stop after compilation (assembly)
gcc -c hello.c -o hello.o     # stop after assembly (object file)
objdump -d hello.o             # disassemble object file
```

---

## 3. Data Types & Variables

### Primitive Type Sizes (x86-64 Linux, LP64 model)

| Type | Size (bytes) | Range (signed) | Format specifier |
|---|---|---|---|
| `char` | 1 | −128 … 127 | `%c` (char), `%d` (int val) |
| `unsigned char` | 1 | 0 … 255 | `%u` |
| `short` | 2 | −32 768 … 32 767 | `%hd` |
| `unsigned short` | 2 | 0 … 65 535 | `%hu` |
| `int` | 4 | −2 147 483 648 … 2 147 483 647 | `%d` |
| `unsigned int` | 4 | 0 … 4 294 967 295 | `%u` |
| `long` | 8 | −9.2 × 10¹⁸ … 9.2 × 10¹⁸ | `%ld` |
| `unsigned long` | 8 | 0 … 1.8 × 10¹⁹ | `%lu` |
| `long long` | 8 | same as long on LP64 | `%lld` |
| `float` | 4 | ≈ ±3.4 × 10³⁸, 7 sig digits | `%f` |
| `double` | 8 | ≈ ±1.8 × 10³⁰⁸, 15 sig digits | `%lf` |
| `long double` | 16 | 80-bit extended (x87) | `%Lf` |
| pointer (`void*`) | 8 | 0 … 2⁶⁴−1 (virtual address) | `%p` |

Sizes are guaranteed minimums by the standard; `<stdint.h>` gives exact-width types: `int8_t`, `uint32_t`, `int64_t`, etc.

### Signed vs Unsigned

Signed integers use two's complement. The MSB is the sign bit. Unsigned integers use all bits for magnitude. Mixing signed and unsigned in expressions triggers implicit conversion — unsigned "wins", which silently converts negative values to large positives.

```c
int  a = -1;
unsigned int b = 1;
if (a < b)           /* FALSE — a promotes to unsigned: 4294967295 */
    printf("negative");
```

### sizeof Operator

`sizeof` is evaluated at compile time (for fixed types) and returns `size_t` (an unsigned integer type, use `%zu`).

```c
printf("%zu\n", sizeof(int));       /* 4 */
printf("%zu\n", sizeof(double));    /* 8 */
printf("%zu\n", sizeof(int[10]));   /* 40 */
```

### Integer Overflow

Signed integer overflow is **undefined behavior** (UB). The compiler assumes it never happens and may optimize based on that assumption. Unsigned overflow is well-defined: it wraps modulo $2^n$.

```c
int x = 2147483647;
x = x + 1;     /* UB — signed overflow; likely becomes -2147483648 on x86 */

unsigned int u = 4294967295u;
u = u + 1;     /* Well-defined: wraps to 0 */
```

Detect with `-fsanitize=undefined` (UBSan).

### Type Casting

**Implicit:** compiler inserts conversion automatically; may lose precision.

```c
double d = 3.99;
int i = d;          /* i = 3, fractional part truncated silently */
```

**Explicit:** programmer inserts cast; documents intent, suppresses warnings.

```c
int a = 7, b = 2;
double result = (double)a / b;   /* 3.5, not 3 */
```

Cast from larger to smaller integer type truncates high bits; behavior is implementation-defined for signed types.

### Constants

| Mechanism | Syntax | Has type | Visible to debugger | Can take address |
|---|---|---|---|---|
| `#define` | `#define PI 3.14159` | No (textual) | No | No |
| `const` | `const double PI = 3.14159;` | Yes | Yes | Yes |

Prefer `const` for typed constants. Use `#define` for compile-time integer constants used in array sizes (before C99 VLAs) or conditional compilation.

```c
#define MAX 100
const int LIMIT = 100;

int arr[MAX];          /* Valid — MAX is compile-time constant */
/* int arr2[LIMIT]; */ /* Valid only in C99+ as VLA or with static storage */
```

---

## 4. Operators

### Arithmetic & Relational

| Operator | Meaning | Example |
|---|---|---|
| `+` `-` `*` `/` | Add, sub, mul, div | `7 / 2 = 3` (integer) |
| `%` | Modulo (remainder) | `7 % 2 = 1` |
| `==` `!=` | Equality, inequality | `(a == b)` returns 0 or 1 |
| `<` `<=` `>` `>=` | Relational | — |
| `&&` `\|\|` `!` | Logical AND, OR, NOT | Short-circuit evaluation |

### Bitwise Operators

| Operator | Meaning | Example (8-bit) |
|---|---|---|
| `&` | AND | `0b1100 & 0b1010 = 0b1000` |
| `\|` | OR | `0b1100 \| 0b1010 = 0b1110` |
| `^` | XOR | `0b1100 ^ 0b1010 = 0b0110` |
| `~` | NOT (bitwise complement) | `~0b00001111 = 0b11110000` |
| `<<` | Left shift (multiply by $2^n$) | `1 << 3 = 8` |
| `>>` | Right shift | `16 >> 2 = 4` |

### Operator Precedence (high to low, partial)

| Precedence | Operators |
|---|---|
| 15 | `()` `[]` `->` `.` |
| 14 | Unary: `!` `~` `++` `--` `*` `&` `sizeof` `(type)` |
| 13 | `*` `/` `%` |
| 12 | `+` `-` |
| 11 | `<<` `>>` |
| 10 | `<` `<=` `>` `>=` |
| 9 | `==` `!=` |
| 8 | `&` |
| 7 | `^` |
| 6 | `\|` |
| 5 | `&&` |
| 4 | `\|\|` |
| 3 | `?:` |
| 2 | `=` `+=` `-=` etc. |
| 1 | `,` |

When in doubt, use parentheses. `a & b == 0` parses as `a & (b == 0)` due to precedence, not `(a & b) == 0`.

### Bitwise Tricks

```c
/* Check if n is even */
if ((n & 1) == 0) { /* even */ }

/* Swap a and b without temp variable */
a ^= b;
b ^= a;
a ^= b;

/* Set bit k in x */
x |= (1 << k);

/* Clear bit k in x */
x &= ~(1 << k);

/* Toggle bit k in x */
x ^= (1 << k);

/* Check if bit k is set */
if ((x >> k) & 1) { /* bit k is set */ }

/* Count set bits (Brian Kernighan's algorithm) — O(set bits) */
int count = 0;
while (x) {
    x &= (x - 1);   /* clears the lowest set bit */
    count++;
}

/* Round up to next power of 2 (32-bit) */
x--;
x |= x >> 1;
x |= x >> 2;
x |= x >> 4;
x |= x >> 8;
x |= x >> 16;
x++;
```

---

## 5. Control Flow

### if / else

```c
if (x > 0) {
    printf("positive\n");
} else if (x < 0) {
    printf("negative\n");
} else {
    printf("zero\n");
}
```

### switch

```c
switch (ch) {
    case 'a':
        printf("vowel\n");
        break;          /* Without break, execution falls through to next case */
    case 'e':
    case 'i':           /* Intentional fallthrough: 'e' and 'i' share same body */
        printf("vowel\n");
        break;
    default:
        printf("consonant\n");
}
```

**Fallthrough gotcha:** forgetting `break` silently executes the next case's body. Always comment intentional fallthrough. GCC warns with `-Wimplicit-fallthrough`.

### Loops

| Loop | Use when |
|---|---|
| `for (init; cond; step)` | Iteration count known, counter needed |
| `while (cond)` | Condition checked before each iteration; may execute 0 times |
| `do { } while (cond)` | Body must execute at least once (e.g., menu prompt) |

```c
/* for */
for (int i = 0; i < n; i++) { /* i scoped to loop in C99+ */ }

/* while */
while (fgets(buf, sizeof(buf), fp) != NULL) { /* process line */ }

/* do-while */
do {
    printf("Enter 1-5: ");
    scanf("%d", &choice);
} while (choice < 1 || choice > 5);
```

### break and continue

`break` exits the innermost enclosing loop or switch. `continue` skips the rest of the current iteration and jumps to the loop's increment/condition.

### goto

`goto` is valid C. Its one legitimate use: jumping out of deeply nested loops or error-cleanup paths in functions without a mechanism like exceptions.

```c
int process(void) {
    FILE *f = fopen("data.bin", "rb");
    if (!f) goto cleanup;

    char *buf = malloc(1024);
    if (!buf) goto cleanup;

    /* ... work ... */

cleanup:
    if (buf) free(buf);
    if (f)   fclose(f);
    return -1;
}
```

This pattern avoids deeply nested if-else and is used in the Linux kernel.

### Breaking Nested Loops

C has no labeled break. Use a flag or `goto`:

```c
int found = 0;
for (int i = 0; i < rows && !found; i++)
    for (int j = 0; j < cols && !found; j++)
        if (matrix[i][j] == target)
            found = 1;
```

---

## 6. Functions

### Declaration vs Definition

A **declaration** (prototype) tells the compiler the function's signature — enough to call it. A **definition** provides the body.

```c
/* Declaration (prototype) — usually in a .h file */
int add(int a, int b);

/* Definition — in a .c file */
int add(int a, int b) {
    return a + b;
}
```

Without a prior declaration, calling a function whose definition appears later is an error in C99+.

### Call Stack

Each function call pushes a **stack frame** containing: return address, saved registers, local variables, parameters.

```
 High addresses
 ┌─────────────────────────┐
 │   main's frame          │  ← stack base
 │   int x = 5             │
 │   return address        │
 ├─────────────────────────┤
 │   add's frame           │  ← current SP after call
 │   int a = 5, int b = 3  │
 │   return address → main │
 └─────────────────────────┘
 Low addresses              ← stack grows downward
```

When `add` returns, its frame is popped (SP incremented), and execution resumes in `main` at the return address.

### Pass by Value

C always passes arguments by value — a copy is made. Modifying a parameter inside a function does not affect the caller's variable.

```c
void increment(int x) {
    x++;        /* modifies local copy only */
}

int n = 5;
increment(n);
printf("%d\n", n);   /* still 5 */
```

To modify the caller's variable, pass a pointer:

```c
void increment(int *x) {
    (*x)++;
}
increment(&n);       /* n is now 6 */
```

### Recursion

Every recursive call creates a new stack frame. Stack depth is limited (typically 1–8 MB).

**Factorial:**

```c
long long factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

Call tree for `factorial(4)`:

```
factorial(4)
  └── 4 * factorial(3)
            └── 3 * factorial(2)
                      └── 2 * factorial(1)
                                └── 1
```

**Fibonacci:**

```c
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
```

Naive `fib` is $O(2^n)$ time — exponential because it recomputes subproblems. Use memoization or iteration for large `n`.

**Power:**

```c
long long power(long long base, int exp) {
    if (exp == 0) return 1;
    if (exp % 2 == 0) {
        long long half = power(base, exp / 2);
        return half * half;
    }
    return base * power(base, exp - 1);
}
```

Fast power (exponentiation by squaring) is $O(\log n)$.

### Tail Recursion

A call is **tail-recursive** if the recursive call is the last operation. Optimizing compilers can convert tail recursion to iteration (no stack growth). C standard does not guarantee tail-call optimization, but GCC does it with `-O2`.

```c
long long fact_tail(int n, long long acc) {
    if (n <= 1) return acc;
    return fact_tail(n - 1, n * acc);   /* tail call */
}
```

### static Functions and Variables

`static` on a function limits its visibility to the current translation unit (file). Use it to implement module-private helpers.

`static` on a local variable makes it persist across calls, stored in Data/BSS segment instead of the stack.

```c
static int counter(void) {
    static int count = 0;   /* initialized once, persists */
    return ++count;
}
/* counter() → 1, counter() → 2, counter() → 3 ... */
```

### inline Hint

`inline` is a hint to the compiler to replace the call site with the function body, eliminating call overhead. The compiler may ignore it.

```c
static inline int max(int a, int b) {
    return a > b ? a : b;
}
```

---

## 7. Arrays

### Declaration and Initialization

```c
int arr[5];                          /* uninitialized — contains garbage */
int arr[5] = {1, 2, 3, 4, 5};       /* fully initialized */
int arr[5] = {1, 2};                 /* arr[2..4] = 0 */
int arr[]  = {10, 20, 30};           /* size inferred = 3 */
int arr[5] = {0};                    /* all zeros */
```

### Memory Layout

An array is a contiguous block of memory. `arr[i]` is syntactic sugar for `*(arr + i)`.

```
int arr[5] = {10, 20, 30, 40, 50};

Address:   0x100  0x104  0x108  0x10C  0x110
Value:      10     20     30     40     50
Index:     [0]    [1]    [2]    [3]    [4]
```

Each `int` takes 4 bytes. `arr + 1` adds `1 * sizeof(int) = 4` to the base address.

### Array Decay

When passed to a function, an array decays to a pointer to its first element. `sizeof` no longer gives total size.

```c
void print_size(int arr[]) {
    printf("%zu\n", sizeof(arr));   /* prints 8 (pointer size), not 20 */
}

int arr[5];
printf("%zu\n", sizeof(arr));       /* 20 — correct, still in scope */
print_size(arr);
```

Always pass the size separately: `void process(int *arr, int n)`.

### Multi-dimensional Arrays

Stored in **row-major order** — elements of each row are contiguous.

```c
int matrix[3][4];   /* 3 rows, 4 columns, 12 integers = 48 bytes */

/* Memory layout: */
/* [0][0] [0][1] [0][2] [0][3] [1][0] [1][1] ... [2][3] */
```

Access pattern matters for cache performance: iterate rows in the outer loop.

### Strings as char Arrays

A string in C is a `char` array terminated by `'\0'` (null byte, ASCII 0).

```c
char name[6] = {'H','e','l','l','o','\0'};
char name[]  = "Hello";    /* compiler adds '\0' automatically */
```

**Common string functions** (`#include <string.h>`):

| Function | Signature | Behavior |
|---|---|---|
| `strlen` | `size_t strlen(const char *s)` | Returns length excluding `'\0'` |
| `strcpy` | `char *strcpy(char *dst, const char *src)` | Copies src to dst including `'\0'` |
| `strncpy` | `char *strncpy(char *dst, const char *src, size_t n)` | Copies at most n bytes |
| `strcmp` | `int strcmp(const char *a, const char *b)` | 0 if equal, <0 or >0 otherwise |
| `strcat` | `char *strcat(char *dst, const char *src)` | Appends src to dst |
| `strstr` | `char *strstr(const char *hay, const char *needle)` | Returns pointer to first match |
| `sprintf` | `int sprintf(char *buf, const char *fmt, ...)` | Write formatted string to buffer |
| `snprintf` | `int snprintf(char *buf, size_t n, const char *fmt, ...)` | Safe: limits to n bytes |

### Buffer Overflow

```c
char buf[8];
strcpy(buf, "This string is way too long");   /* writes past buf, corrupts stack */
```

Buffer overflow overwrites adjacent memory — return addresses, saved registers. This is the most common C security vulnerability. Always use `strncpy`, `snprintf`, or `strlcpy` (BSD) with explicit size bounds.

---

## 8. Pointers — The Core of C

### What a Pointer Is

A pointer is a variable that stores a memory address. On a 64-bit system, every pointer is 8 bytes regardless of the type it points to.

```c
int x = 42;
int *p = &x;     /* p holds the address of x */

printf("%d\n",  x);    /* 42  — value of x */
printf("%p\n",  p);    /* 0x7fff... — address stored in p */
printf("%d\n", *p);    /* 42  — dereference: value at the address */
```

### Memory Diagram

```
 Variable x:              Pointer p:
 ┌───────────┐            ┌───────────────┐
 │    42     │ ← 0x1000   │   0x1000      │ ← 0x2000
 └───────────┘            └───────────────┘
       ▲                         │
       └─────────────────────────┘
```

`p` is at address `0x2000`. Its value is `0x1000` (address of `x`). `*p` reads the int at `0x1000`.

### Pointer Arithmetic

Adding an integer to a pointer advances it by `n * sizeof(*pointer)` bytes.

```c
int arr[5] = {10, 20, 30, 40, 50};
int *p = arr;      /* points to arr[0] */

printf("%d\n", *p);       /* 10 */
printf("%d\n", *(p+1));   /* 20 */
printf("%d\n", *(p+4));   /* 50 */

p++;               /* p now points to arr[1] */
printf("%d\n", *p);       /* 20 */
```

Pointer subtraction gives the number of elements between two pointers (both must point into the same array).

### Pointer to Pointer

```c
int x = 5;
int *p = &x;
int **pp = &p;    /* pp holds address of p */

printf("%d\n", **pp);    /* 5 — double dereference */
**pp = 99;               /* modifies x through pp and p */
```

Used for: modifying a pointer in a function, arrays of strings (`char **argv`), dynamic 2D arrays.

### Pointer Hazards

| Hazard | Definition | Consequence |
|---|---|---|
| NULL pointer | Pointer with value 0 / `NULL` | Dereferencing is UB; typically segfault |
| Dangling pointer | Pointer to memory that has been freed or gone out of scope | UB; reads garbage, corrupts allocator |
| Wild pointer | Uninitialized pointer; contains whatever was on the stack | UB; may crash or corrupt silently |
| Buffer overrun | Writing past the end of an allocated block | Corrupts adjacent data, security exploit |

```c
int *p;          /* wild — uninitialized */
*p = 5;          /* UB */

int *q = malloc(sizeof(int));
free(q);
*q = 5;          /* UB — use-after-free */

int *r = NULL;
*r = 5;          /* UB — null dereference */
```

Always initialize pointers. Set to `NULL` after `free`.

### const with Pointers — 4 Combinations

| Declaration | Pointer modifiable? | Pointee modifiable? | Use case |
|---|---|---|---|
| `int *p` | Yes | Yes | General mutable pointer |
| `const int *p` | Yes | No | Read-only view (pointer can move, value can't change) |
| `int * const p` | No | Yes | Fixed address, mutable data |
| `const int * const p` | No | No | Fully immutable |

```c
const int *p = &x;     /* can't do *p = 5; but can do p = &y; */
int * const q = &x;    /* can do *q = 5; but can't do q = &y; */
```

Read right to left: `const int *p` = "p is a pointer to const int".

### void Pointer

`void *` is a generic pointer — it can hold any address. You must cast to a concrete type before dereferencing.

```c
void *ptr = malloc(sizeof(int));
int *ip = (int *)ptr;
*ip = 42;
```

`malloc` returns `void *`. `memcpy`, `qsort`, and generic containers use `void *`.

### Function Pointers

A function pointer stores the address of a function. Enables callbacks, dispatch tables, and polymorphism in C.

```c
/* Declaration: pointer to function taking two ints, returning int */
int (*op)(int, int);

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

op = add;
printf("%d\n", op(3, 4));   /* 7 */

op = mul;
printf("%d\n", op(3, 4));   /* 12 */
```

**Callback pattern:**

```c
void apply(int *arr, int n, int (*fn)(int)) {
    for (int i = 0; i < n; i++)
        arr[i] = fn(arr[i]);
}

int square(int x) { return x * x; }

apply(arr, 5, square);
```

**typedef for readability:**

```c
typedef int (*BinaryOp)(int, int);
BinaryOp op = add;
```

### Pointer vs Array

| | Array | Pointer |
|---|---|---|
| `sizeof` | Total byte size | Size of pointer (8 bytes) |
| `&name` | Address of array (same value as `name`) | Address of the pointer variable |
| Modifiable | No (array name is not an lvalue) | Yes (`p++` is valid) |
| `name[i]` | Sugar for `*(name + i)` | Same |

---

## 9. Dynamic Memory

### Stack vs Heap

```
 ┌─────────────────────────────────┐  High address
 │            Stack                │
 │  (local vars, function frames)  │
 │  Grows downward ↓               │
 ├─────────────────────────────────┤
 │             ↕ gap               │
 ├─────────────────────────────────┤
 │            Heap                 │
 │  (malloc/calloc/realloc)        │
 │  Grows upward ↑                 │
 ├─────────────────────────────────┤
 │       BSS (uninit globals)      │
 ├─────────────────────────────────┤
 │       Data (init globals)       │
 ├─────────────────────────────────┤
 │       Text (code)               │
 └─────────────────────────────────┘  Low address
```

Stack is fast (just move SP) but limited. Heap is unlimited (up to virtual memory) but requires explicit management via `malloc`/`free`.

### Allocation Functions (`#include <stdlib.h>`)

| Function | Signature | Behavior |
|---|---|---|
| `malloc` | `void *malloc(size_t size)` | Allocates `size` bytes, **uninitialized**. Returns `NULL` on failure. |
| `calloc` | `void *calloc(size_t n, size_t size)` | Allocates `n * size` bytes, **zero-initialized**. |
| `realloc` | `void *realloc(void *ptr, size_t size)` | Resizes allocation. May move memory; original pointer is invalid after call. |
| `free` | `void free(void *ptr)` | Returns memory to allocator. `ptr` must be a value returned by malloc/calloc/realloc. |

```c
int *arr = malloc(10 * sizeof(int));
if (!arr) {
    perror("malloc");
    exit(EXIT_FAILURE);
}
/* use arr */
free(arr);
arr = NULL;    /* prevent dangling pointer use */
```

### Memory Leak

A leak occurs when allocated memory is never freed. The OS reclaims memory when the process exits, so leaks matter in long-running servers and embedded systems.

Detect with **Valgrind**:

```bash
valgrind --leak-check=full ./program
```

Or **AddressSanitizer** (faster, compiler-integrated):

```bash
gcc -fsanitize=address -g program.c -o program && ./program
```

### Double Free and Use-After-Free

Both are undefined behavior and common security vulnerabilities.

```c
char *p = malloc(64);
free(p);
free(p);     /* double free — corrupts allocator's internal structures */

char *q = malloc(32);
free(q);
q[0] = 'x'; /* use-after-free — reads/writes deallocated memory */
```

Pattern to prevent: `free(p); p = NULL;` — freeing `NULL` is a no-op.

### Dynamic Array (Growable Array)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int    *data;
    size_t  size;
    size_t  capacity;
} DynArray;

void da_init(DynArray *da) {
    da->data     = malloc(4 * sizeof(int));
    da->size     = 0;
    da->capacity = 4;
}

void da_push(DynArray *da, int val) {
    if (da->size == da->capacity) {
        da->capacity *= 2;
        int *tmp = realloc(da->data, da->capacity * sizeof(int));
        if (!tmp) { free(da->data); exit(EXIT_FAILURE); }
        da->data = tmp;
    }
    da->data[da->size++] = val;
}

int da_get(const DynArray *da, size_t i) {
    return da->data[i];
}

void da_free(DynArray *da) {
    free(da->data);
    da->data     = NULL;
    da->size     = 0;
    da->capacity = 0;
}

int main(void) {
    DynArray da;
    da_init(&da);
    for (int i = 0; i < 20; i++)
        da_push(&da, i * i);
    for (size_t i = 0; i < da.size; i++)
        printf("%d ", da_get(&da, i));
    printf("\n");
    da_free(&da);
    return 0;
}
```

Amortized $O(1)$ push via doubling. Total work for $n$ pushes: $n + n/2 + n/4 + \cdots = 2n = O(n)$.

---

## 10. Structs & Unions

### Struct Definition and Access

```c
struct Point {
    double x;
    double y;
};

struct Point p1;
p1.x = 3.0;
p1.y = 4.0;

struct Point p2 = {1.0, 2.0};    /* initializer list */
struct Point p3 = {.x = 1.0, .y = 2.0};  /* designated initializer (C99) */
```

Arrow operator for pointer to struct:

```c
struct Point *pp = &p1;
pp->x = 10.0;    /* equivalent to (*pp).x = 10.0 */
```

### Struct Padding and Alignment

The compiler inserts padding bytes to satisfy alignment requirements (each member must be at an offset that is a multiple of its size). This makes structs larger than you might expect.

```c
struct Padded {
    char  a;     /* 1 byte */
                 /* 3 bytes padding */
    int   b;     /* 4 bytes */
    char  c;     /* 1 byte */
                 /* 7 bytes padding */
    double d;    /* 8 bytes */
};               /* total: 24 bytes */

struct Packed {
    char   a;
    int    b;
    char   c;
    double d;
} __attribute__((packed));  /* total: 14 bytes, but unaligned access may be slow or fault */
```

**Minimize padding by ordering members largest-to-smallest:**

```c
struct Optimal {
    double d;    /* 8 */
    int    b;    /* 4 */
    char   a;    /* 1 */
    char   c;    /* 1 */
                 /* 2 bytes padding to align struct to 8 */
};               /* total: 16 bytes */
```

Check with `offsetof(struct T, member)` from `<stddef.h>`.

### Struct as Parameter vs Pointer to Struct

Passing a struct by value copies all its bytes. For large structs, pass a pointer.

```c
void scale(struct Point p, double factor) {    /* copies 16 bytes */
    p.x *= factor;   /* no effect on caller */
}

void scale_inplace(struct Point *p, double factor) {  /* copies 8 bytes (ptr) */
    p->x *= factor;   /* modifies caller's struct */
    p->y *= factor;
}
```

### Self-Referential Struct (Linked List Node)

```c
struct Node {
    int          data;
    struct Node *next;   /* pointer to same struct type */
};
```

A struct cannot contain itself by value (infinite size), but it can contain a pointer to itself.

### Union

A union overlaps all members at the same address. Size = size of the largest member.

```c
union Variant {
    int    i;
    float  f;
    char   bytes[4];
};

union Variant v;
v.i = 0x3F800000;
printf("%f\n", v.f);    /* 1.0 — interpret same bits as float */
```

Use unions for: type punning, tagged unions (discriminated unions), memory-mapped register layouts.

### Enum

```c
typedef enum {
    MON = 0, TUE, WED, THU, FRI, SAT, SUN
} Weekday;

Weekday today = WED;   /* underlying type is int; WED = 2 */
```

Enums are not type-safe in C — you can assign any int to an enum variable.

---

## 11. Memory Layout of a C Program

```
 Virtual Address Space (typical Linux x86-64)
 ┌────────────────────────────────────┐  High (0x7FFF...)
 │    Command-line args, env vars     │
 ├────────────────────────────────────┤
 │             Stack                  │  auto variables, function frames
 │          (grows down ↓)            │
 │               ...                  │
 │               ...                  │
 │          (grows up ↑)              │
 │             Heap                   │  malloc/free
 ├────────────────────────────────────┤
 │              BSS                   │  Uninitialized global/static vars (zero-filled by OS)
 ├────────────────────────────────────┤
 │             Data                   │  Initialized global/static vars
 ├────────────────────────────────────┤
 │             Text                   │  Machine code (read-only, executable)
 └────────────────────────────────────┘  Low (0x0000...)
```

### Where Each Variable Lives

| Declaration | Segment | Initialized |
|---|---|---|
| `int x = 5;` (global) | Data | Yes, to 5 |
| `int x;` (global) | BSS | Yes, to 0 by OS |
| `static int x = 5;` (local) | Data | Yes, to 5 |
| `static int x;` (local) | BSS | Yes, to 0 |
| `int x = 5;` (local) | Stack | At runtime, to 5 |
| `malloc(...)` result | Heap | Not zeroed (malloc) |
| `"hello"` string literal | Data / `.rodata` | Read-only |

Stack grows **downward** on x86. Each function call decrements the stack pointer. A stack overflow occurs when the stack collides with the heap or the guard page.

---

## 12. The Preprocessor

The preprocessor runs before the compiler and performs text substitution. It does not understand C types or scope.

### Object-like Macros

```c
#define PI        3.14159265358979
#define MAX_SIZE  1024
#define NEWLINE   '\n'
```

### Function-like Macros

```c
#define SQUARE(x)    ((x) * (x))
#define MAX(a, b)    ((a) > (b) ? (a) : (b))
#define ABS(x)       ((x) >= 0 ? (x) : -(x))
```

**Always parenthesize arguments and the whole expression.** Without this:

```c
#define SQUARE(x)  x * x
SQUARE(1 + 2)   /* expands to 1 + 2 * 1 + 2 = 5, not 9 */
```

**Double evaluation pitfall:**

```c
#define MAX(a, b)  ((a) > (b) ? (a) : (b))
MAX(i++, j++)   /* i++ or j++ evaluated twice — wrong */
```

Use `static inline` functions instead of macros when possible.

### Conditional Compilation

```c
#define DEBUG 1

#ifdef DEBUG
    printf("x = %d\n", x);
#endif

#ifndef NDEBUG
    assert(ptr != NULL);
#endif

#if defined(__linux__)
    /* Linux-specific code */
#elif defined(_WIN32)
    /* Windows-specific code */
#endif
```

### Include Guards

Prevent a header from being included multiple times in one translation unit.

```c
/* mylib.h */
#ifndef MYLIB_H
#define MYLIB_H

/* declarations */

#endif /* MYLIB_H */
```

`#pragma once` is a non-standard but widely supported alternative (GCC, Clang, MSVC).

### Predefined Macros

| Macro | Expands to |
|---|---|
| `__FILE__` | Current source file name (string literal) |
| `__LINE__` | Current line number (integer) |
| `__DATE__` | Compilation date: `"May 15 2026"` |
| `__TIME__` | Compilation time: `"14:32:00"` |
| `__func__` | Current function name (C99, string literal) |
| `__STDC_VERSION__` | C standard version: `199901L` (C99), `201112L` (C11) |

```c
#define LOG(msg)  fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg)
```

---

## 13. File I/O

### Opening and Closing Files

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

FILE *fp = fopen("data.txt", "r");
if (!fp) {
    fprintf(stderr, "fopen: %s\n", strerror(errno));
    return -1;
}
/* ... use fp ... */
fclose(fp);
```

### File Modes

| Mode | Meaning |
|---|---|
| `"r"` | Read text; file must exist |
| `"w"` | Write text; creates or truncates |
| `"a"` | Append text; creates if absent |
| `"r+"` | Read + write; file must exist |
| `"w+"` | Read + write; creates or truncates |
| `"rb"` `"wb"` `"ab"` | Binary equivalents (no newline translation) |

### Core I/O Functions

| Function | Signature | Use |
|---|---|---|
| `fprintf` | `int fprintf(FILE*, const char*, ...)` | Formatted write |
| `fscanf` | `int fscanf(FILE*, const char*, ...)` | Formatted read |
| `fgets` | `char *fgets(char *buf, int n, FILE*)` | Read one line (safe) |
| `fputs` | `int fputs(const char *s, FILE*)` | Write string |
| `fread` | `size_t fread(void*, size_t, size_t, FILE*)` | Binary block read |
| `fwrite` | `size_t fwrite(void*, size_t, size_t, FILE*)` | Binary block write |
| `fseek` | `int fseek(FILE*, long, int)` | Seek to position |
| `ftell` | `long ftell(FILE*)` | Current position |
| `rewind` | `void rewind(FILE*)` | Seek to start |
| `feof` | `int feof(FILE*)` | Test end-of-file flag |
| `ferror` | `int ferror(FILE*)` | Test error flag |

### Reading a File Line by Line

```c
#include <stdio.h>
#include <string.h>

int read_lines(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char line[4096];
    int lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
            line[len - 1] = '\0';           /* strip trailing newline */
        printf("%4d: %s\n", lineno, line);
    }
    fclose(fp);
    return lineno;
}
```

`fgets` stops at newline or `n-1` characters, always null-terminates. Check `ferror(fp)` after the loop to distinguish EOF from error.

### Error Handling with errno

```c
#include <errno.h>
#include <string.h>

FILE *fp = fopen("/root/secret", "r");
if (!fp) {
    /* errno is set by fopen on failure */
    perror("fopen");                      /* prints: "fopen: Permission denied" */
    fprintf(stderr, "%s\n", strerror(errno));  /* same but to arbitrary stream */
}
```

`errno` is a global (actually thread-local in modern libc) set by system calls on failure. Clear it before a call if you need to distinguish "no error set" from a prior error.

---

## 14. Common Data Structures in C

### Singly Linked List

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int          data;
    struct Node *next;
} Node;

Node *node_new(int val) {
    Node *n = malloc(sizeof(Node));
    if (!n) exit(EXIT_FAILURE);
    n->data = val;
    n->next = NULL;
    return n;
}

void list_push_front(Node **head, int val) {
    Node *n = node_new(val);
    n->next = *head;
    *head = n;
}

void list_push_back(Node **head, int val) {
    Node *n = node_new(val);
    if (!*head) { *head = n; return; }
    Node *cur = *head;
    while (cur->next) cur = cur->next;
    cur->next = n;
}

void list_delete(Node **head, int val) {
    Node *cur = *head, *prev = NULL;
    while (cur && cur->data != val) {
        prev = cur;
        cur  = cur->next;
    }
    if (!cur) return;                  /* not found */
    if (!prev) *head = cur->next;      /* removing head */
    else        prev->next = cur->next;
    free(cur);
}

void list_print(const Node *head) {
    for (const Node *cur = head; cur; cur = cur->next)
        printf("%d -> ", cur->data);
    printf("NULL\n");
}

Node *list_reverse(Node *head) {
    Node *prev = NULL, *cur = head, *next = NULL;
    while (cur) {
        next      = cur->next;
        cur->next = prev;
        prev      = cur;
        cur       = next;
    }
    return prev;
}

void list_free(Node **head) {
    Node *cur = *head;
    while (cur) {
        Node *tmp = cur->next;
        free(cur);
        cur = tmp;
    }
    *head = NULL;
}
```

### Stack Using Array

```c
#define STACK_MAX 256

typedef struct {
    int data[STACK_MAX];
    int top;
} Stack;

void stack_init(Stack *s)       { s->top = -1; }
int  stack_empty(const Stack *s){ return s->top == -1; }
int  stack_full(const Stack *s) { return s->top == STACK_MAX - 1; }

void stack_push(Stack *s, int val) {
    if (stack_full(s)) { fprintf(stderr, "stack overflow\n"); return; }
    s->data[++(s->top)] = val;
}

int stack_pop(Stack *s) {
    if (stack_empty(s)) { fprintf(stderr, "stack underflow\n"); return -1; }
    return s->data[(s->top)--];
}

int stack_peek(const Stack *s) {
    if (stack_empty(s)) { fprintf(stderr, "empty stack\n"); return -1; }
    return s->data[s->top];
}
```

### Queue Using Circular Array

```c
#define QUEUE_MAX 256

typedef struct {
    int data[QUEUE_MAX];
    int head, tail, size;
} Queue;

void queue_init(Queue *q)       { q->head = q->tail = q->size = 0; }
int  queue_empty(const Queue *q){ return q->size == 0; }
int  queue_full(const Queue *q) { return q->size == QUEUE_MAX; }

void queue_enqueue(Queue *q, int val) {
    if (queue_full(q)) { fprintf(stderr, "queue full\n"); return; }
    q->data[q->tail] = val;
    q->tail = (q->tail + 1) % QUEUE_MAX;
    q->size++;
}

int queue_dequeue(Queue *q) {
    if (queue_empty(q)) { fprintf(stderr, "queue empty\n"); return -1; }
    int val = q->data[q->head];
    q->head = (q->head + 1) % QUEUE_MAX;
    q->size--;
    return val;
}
```

Circular indexing (`% QUEUE_MAX`) avoids shifting elements, keeping enqueue and dequeue both $O(1)$.

### Hash Table with Chaining

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HT_SIZE 64

typedef struct HEntry {
    char        *key;
    int          val;
    struct HEntry *next;
} HEntry;

typedef struct {
    HEntry *buckets[HT_SIZE];
} HashTable;

static unsigned int ht_hash(const char *key) {
    unsigned int h = 5381;
    while (*key) h = h * 33 ^ (unsigned char)*key++;
    return h % HT_SIZE;
}

void ht_init(HashTable *ht) {
    memset(ht->buckets, 0, sizeof(ht->buckets));
}

void ht_set(HashTable *ht, const char *key, int val) {
    unsigned int idx = ht_hash(key);
    HEntry *e = ht->buckets[idx];
    while (e) {
        if (strcmp(e->key, key) == 0) { e->val = val; return; }
        e = e->next;
    }
    HEntry *ne = malloc(sizeof(HEntry));
    ne->key  = strdup(key);
    ne->val  = val;
    ne->next = ht->buckets[idx];
    ht->buckets[idx] = ne;
}

int ht_get(const HashTable *ht, const char *key, int *out) {
    unsigned int idx = ht_hash(key);
    for (HEntry *e = ht->buckets[idx]; e; e = e->next)
        if (strcmp(e->key, key) == 0) { *out = e->val; return 1; }
    return 0;
}

void ht_free(HashTable *ht) {
    for (int i = 0; i < HT_SIZE; i++) {
        HEntry *e = ht->buckets[i];
        while (e) {
            HEntry *tmp = e->next;
            free(e->key);
            free(e);
            e = tmp;
        }
        ht->buckets[i] = NULL;
    }
}
```

Average $O(1)$ get/set assuming low load factor. Worst case $O(n)$ if all keys hash to the same bucket.

---

## 15. Undefined Behavior Taxonomy

Undefined behavior means the C standard places no constraint on what the program does. The compiler may assume UB never occurs and optimize accordingly, which can produce surprising results.

| UB | Example | Typical x86 outcome | Why dangerous |
|---|---|---|---|
| Signed integer overflow | `INT_MAX + 1` | Wraps to `INT_MIN` on x86, but compiler may delete overflow-unreachable code | Incorrect comparisons, infinite loops |
| Null pointer dereference | `*(int*)NULL = 0` | SIGSEGV | Crash or security exploit |
| Out-of-bounds array access | `arr[10]` where `arr` has 5 elements | Reads/writes arbitrary memory | Stack corruption, heap corruption |
| Use-after-free | `free(p); *p = 1;` | Corrupts allocator metadata | Heap corruption, arbitrary code execution |
| Double free | `free(p); free(p);` | Corrupts allocator free-list | Arbitrary code execution |
| Uninitialized read | `int x; printf("%d", x);` | Prints whatever was on the stack | Data leaks, nondeterministic behavior |
| Data race | Two threads read/write without sync | Any interleaving; torn reads | Nondeterministic, hard to reproduce |
| Left-shift by negative or ≥ width | `1 << -1`, `1 << 32` | Likely no-op or large value on x86 | Compiler may produce any value |
| Modifying a string literal | `char *s = "hi"; s[0] = 'H';` | SIGSEGV (`.rodata` is read-only) | Crash |
| Returning pointer to local variable | `int *f() { int x; return &x; }` | Returns valid-looking address, but frame is gone | Dangling pointer; subsequent write corrupts stack |
| Violating strict aliasing | `float f; *(int*)&f = 0;` | May work, but compiler assumes `int*` and `float*` don't alias | Silent misoptimization |
| Calling function through wrong type | `void (*fp)(int); fp = (void(*)(int))strlen; fp(0);` | Wrong calling convention | Stack corruption |

Enable `-fsanitize=address,undefined` during development to catch most of these at runtime.

---

## 16. Compilation Flags & Debugging

### Essential GCC / Clang Flags

| Flag | Effect |
|---|---|
| `-Wall` | Enable most warning categories |
| `-Wextra` | Enable extra warnings (unused params, sign compare, etc.) |
| `-Werror` | Treat all warnings as errors |
| `-Wpedantic` | Enforce strict standard compliance |
| `-std=c11` | Use C11 standard |
| `-O0` | No optimization (default); fastest compilation, easiest debugging |
| `-O2` | Aggressive optimization; use for release builds |
| `-O3` | More aggressive; enables auto-vectorization |
| `-g` | Include debug info (DWARF); required for gdb/lldb |
| `-fsanitize=address` | AddressSanitizer: detect heap/stack overflows, use-after-free |
| `-fsanitize=undefined` | UBSanitizer: detect signed overflow, shift UB, etc. |
| `-fsanitize=thread` | ThreadSanitizer: detect data races |
| `-fstack-protector-strong` | Insert stack canaries to detect buffer overflows |
| `-D_FORTIFY_SOURCE=2` | Enable glibc buffer overflow checks (requires `-O1` or higher) |

Recommended development build:

```bash
gcc -std=c11 -Wall -Wextra -Werror -g -fsanitize=address,undefined -o prog prog.c
```

### gdb Basics

```bash
gcc -g -o prog prog.c
gdb ./prog
```

| gdb command | Effect |
|---|---|
| `run [args]` | Start the program |
| `break main` | Set breakpoint at `main` |
| `break file.c:42` | Set breakpoint at line 42 |
| `next` (or `n`) | Execute next line (step over function calls) |
| `step` (or `s`) | Step into function calls |
| `continue` (or `c`) | Run until next breakpoint |
| `print expr` | Print value of expression |
| `display expr` | Print expression at every stop |
| `backtrace` (or `bt`) | Show call stack |
| `frame N` | Switch to frame N in the backtrace |
| `info locals` | Print all local variables |
| `watch var` | Break when `var` changes |
| `quit` | Exit gdb |

### Valgrind

```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./prog
```

Valgrind instruments every memory operation at runtime. Slowdown is ~20×. Reports: memory leaks, use of uninitialized memory, invalid reads/writes, double frees.

### AddressSanitizer

```bash
gcc -fsanitize=address -g -o prog prog.c
./prog
```

5–10× slowdown (much faster than Valgrind). Reports similar errors with exact stack traces. Output is color-coded and includes source line numbers when compiled with `-g`.

### Useful One-Liners

```bash
# Check which symbols are undefined in an object file
nm -u prog.o

# Show all symbols exported by a shared library
nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep " T " | head -20

# Show shared library dependencies
ldd ./prog

# Inspect binary sections
size prog

# Count lines of code
wc -l *.c *.h
```

---

## 17. Interview Q&A — 20 Questions

---

**Q1. What is the difference between `malloc` and `calloc`?**

`malloc(n)` allocates `n` bytes without initializing them — contents are undefined. `calloc(count, size)` allocates `count * size` bytes and zero-initializes all of them. Use `calloc` when you need zeroed memory (e.g., for arrays you'll index before writing). Both return `NULL` on failure; both must be freed with `free`.

*Follow-up:* When is `calloc` slower than `malloc + memset`? Rarely — the OS may provide zero pages from the page allocator already zeroed, making `calloc` effectively free for large allocations.

---

**Q2. What is a pointer and what is the output of the following?**

```c
int x = 10;
int *p = &x;
*p = 20;
printf("%d\n", x);
```

A pointer stores a memory address. `&x` takes x's address; `*p = 20` writes to that address, modifying `x`. Output: `20`.

*Follow-up:* What happens if `p` is `NULL` and you do `*p = 20`? Undefined behavior, typically SIGSEGV.

---

**Q3. What is the difference between `char *s = "hello"` and `char s[] = "hello"`?**

`char *s = "hello"` creates a pointer to a string literal stored in read-only memory (`.rodata`). `s[0] = 'H'` is undefined behavior. `char s[] = "hello"` copies the literal onto the stack; the array is writable.

*Follow-up:* Can you `free(s)` in either case? No — `free` only valid for `malloc`-allocated memory.

---

**Q4. What is undefined behavior? Give two examples.**

UB means the C standard imposes no requirement on what the program does when that situation arises. The compiler may produce any output, crash, or silently produce wrong results. Examples: signed integer overflow (`INT_MAX + 1`), dereferencing a null or freed pointer.

*Follow-up:* Why does the compiler care about UB? Because it assumes UB never occurs, enabling aggressive optimizations (e.g., removing "dead" overflow checks).

---

**Q5. Explain the difference between `struct` and `union`.**

A `struct` allocates space for all its members sequentially (plus padding). Total size = sum of member sizes + padding. A `union` overlays all members at the same address; size = size of the largest member. Writing one member of a union and reading another is type punning (implementation-defined in C, well-defined in C99/C11 via `memcpy`).

*Follow-up:* What is a tagged union? A struct containing a union plus an enum/int indicating which member is currently active.

---

**Q6. What does `static` mean in each of these contexts?**

```c
static int x;              /* (a) file scope */
void f(void) {
    static int count = 0;  /* (b) local scope */
}
static void helper(void) { /* (c) function */
}
```

(a) Limits the variable's linkage to the current translation unit (file) — not visible to other `.c` files. (b) Makes the local variable persist across calls, stored in BSS/Data instead of stack. (c) Limits the function's visibility to the current file.

*Follow-up:* What is the difference between `static` and `extern`? `extern` declares a variable defined in another translation unit (links externally). `static` forces internal linkage.

---

**Q7. What is the output?**

```c
int a = 5, b = 10;
a ^= b;
b ^= a;
a ^= b;
printf("%d %d\n", a, b);
```

Output: `10 5`. XOR swap exchanges `a` and `b` without a temporary variable. Note: if `a` and `b` alias the same location, the result is 0 for both — a known pitfall.

*Follow-up:* Is XOR swap faster than temp-variable swap? Usually not; the compiler optimizes temp-variable swap to register moves; XOR swap introduces sequential data dependencies that hinder superscalar execution.

---

**Q8. What is stack overflow (the bug, not the website)?**

Stack overflow occurs when the call stack grows beyond its allocated limit, typically due to unbounded recursion. Each function call allocates a new stack frame; without a base case or with a very deep recursion, the stack pointer moves into protected memory and the OS delivers SIGSEGV.

*Follow-up:* How can you increase the stack size? On Linux: `ulimit -s unlimited` before running, or `setrlimit(RLIMIT_STACK, ...)` programmatically, or use `alloca` carefully, or rewrite recursion as iteration.

---

**Q9. What is the difference between `++i` and `i++`?**

`++i` (pre-increment) increments `i` and returns the new value. `i++` (post-increment) returns the current value then increments. In standalone expressions, both produce the same result. The difference matters when the result is used: `int x = i++` stores the old value; `int x = ++i` stores the new value.

*Follow-up:* Which is preferred in a `for` loop? `++i` is idiomatic; for integers, both are equivalent after optimization. For iterators in C++, `++i` avoids creating a temporary copy.

---

**Q10. Explain `const int *`, `int * const`, and `const int * const`.**

`const int *p` — pointer is mutable, pointee is read-only. Can point to different integers, cannot modify the integer through `p`. `int * const p` — pointer is fixed (must be initialized), pointee is writable. `const int * const p` — both the pointer and the pointee are read-only.

*Follow-up:* Can you cast away `const`? With an explicit cast (`(int *)p`), yes, but modifying an originally-`const` object through it is UB.

---

**Q11. What is a memory leak? How do you detect it in C?**

A memory leak occurs when heap memory is allocated but never freed, and no pointer to it remains — making it impossible to free. The process's resident memory grows unboundedly. Detect with: Valgrind (`--leak-check=full`), AddressSanitizer (with LeakSanitizer enabled by default on Linux), or static analysis tools like Clang's `scan-build`.

*Follow-up:* Does the OS recover leaked memory? Yes, when the process exits. Leaks matter in long-running processes (servers, daemons, embedded systems).

---

**Q12. What is the size of `sizeof(arr)` vs `sizeof(ptr)` when `arr` is `int arr[10]`?**

`sizeof(arr)` is `40` bytes (10 × 4). When `arr` decays to a pointer (passed to a function), `sizeof(ptr)` is `8` bytes (pointer size on 64-bit). This is why you must pass array sizes explicitly to functions.

*Follow-up:* How do you find the number of elements in an array without a separate size variable? `int n = sizeof(arr) / sizeof(arr[0]);` — only works in scope where `arr` is declared, not inside a function that received the array as a parameter.

---

**Q13. What happens when you dereference a dangling pointer?**

Undefined behavior. After `free(p)`, the memory may be reallocated and used for something else. Reading from `p` gives whatever bytes are now there. Writing through `p` corrupts live data structures, potentially causing crashes far from the bug's origin. This is a common and severe bug.

*Follow-up:* How do you prevent dangling pointers? Set the pointer to `NULL` immediately after `free`. Use wrapper macros: `#define SAFE_FREE(p) do { free(p); (p) = NULL; } while(0)`.

---

**Q14. What is the purpose of `volatile`?**

`volatile` tells the compiler that a variable may change outside the program's control (hardware register, signal handler, another thread without synchronization, memory-mapped I/O). The compiler must not cache the variable in a register or reorder accesses to it. It does **not** imply atomicity or memory ordering.

*Follow-up:* Is `volatile` sufficient for inter-thread communication? No. Use `_Atomic` (C11) or proper synchronization (mutex, semaphore) for thread safety.

---

**Q15. What is the difference between `#include <file.h>` and `#include "file.h"`?**

Angle brackets search standard system include directories first (e.g., `/usr/include`). Double quotes search the directory of the including file first, then system directories. Use angle brackets for system/library headers, double quotes for your own project headers.

*Follow-up:* What does `-I/path` do in gcc? Adds `/path` to the list of directories searched for `#include <...>` and `#include "..."`.

---

**Q16. Implement `strdup` using only `strlen`, `malloc`, and `memcpy`.**

```c
char *my_strdup(const char *s) {
    size_t len = strlen(s) + 1;   /* +1 for '\0' */
    char *copy = malloc(len);
    if (!copy) return NULL;
    memcpy(copy, s, len);
    return copy;
}
```

The caller must `free` the returned pointer. `strcpy` would also work in place of `memcpy`.

*Follow-up:* Is `strdup` part of the C standard? It was POSIX-only until C23, which added it to the standard library.

---

**Q17. What is the difference between `exit(0)` and `return 0` from `main`?**

`return 0` from `main` calls destructors for static objects (in C++), flushes stdio buffers, and runs `atexit` handlers before terminating. `exit(0)` does the same. `_exit(0)` (POSIX) terminates immediately without flushing buffers or calling `atexit` handlers — used in `fork`/`exec` patterns.

*Follow-up:* What does `abort()` do? Sends SIGABRT to the process, producing a core dump. Called by `assert` on failure.

---

**Q18. What is the output?**

```c
int arr[] = {1, 2, 3, 4, 5};
int *p = arr + 2;
printf("%d %d\n", *p, *(p - 1));
```

`p = arr + 2` points to `arr[2] = 3`. `*p = 3`, `*(p-1) = arr[1] = 2`. Output: `3 2`.

*Follow-up:* What is `p[-1]`? It is `*(p - 1)` = `2`. Negative array indices are valid as long as the resulting address is within the same array object.

---

**Q19. What is `restrict`?**

`restrict` (C99) is a hint on a pointer parameter that no other pointer in the current scope aliases the same memory. This enables the compiler to generate more efficient code (e.g., SIMD, fewer loads). `memcpy`'s signature uses `restrict` because source and destination must not overlap (`memmove` handles overlap).

```c
void add(int *restrict dst, const int *restrict src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
}
```

*Follow-up:* What happens if you lie about `restrict`? Undefined behavior — the compiler may produce incorrect vectorized code.

---

**Q20. What is the difference between a process and a thread in terms of memory?**

A process has its own virtual address space (text, data, heap, stack). Threads within a process share the same address space (same heap, same global variables, same code) but each has its own stack and register state. Shared memory means threads can communicate directly but must synchronize to avoid data races. Creating a thread is cheaper than forking a process because no address space copy is needed.

*Follow-up:* What is a race condition in C? When two threads access a shared variable without synchronization and at least one access is a write, the result depends on scheduling order — undefined behavior in C11's memory model.

---

## 18. Solved Practice Problems

---

### Problem 1: Reverse a String In-Place

**Statement:** Given a null-terminated string, reverse it in place without allocating extra memory.

**Approach:** Use two pointers starting at the ends and swap inward until they meet. $O(n)$ time, $O(1)$ space.

```c
#include <stdio.h>
#include <string.h>

void reverse_string(char *s) {
    int left = 0;
    int right = (int)strlen(s) - 1;
    while (left < right) {
        char tmp    = s[left];
        s[left]     = s[right];
        s[right]    = tmp;
        left++;
        right--;
    }
}

int main(void) {
    char s[] = "Hello, World!";
    reverse_string(s);
    printf("%s\n", s);   /* !dlroW ,olleH */
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 2: Check If a String Is a Palindrome

**Statement:** Return 1 if the string reads the same forwards and backwards, 0 otherwise.

**Approach:** Two-pointer convergence, same as reverse but compare instead of swap.

```c
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int is_palindrome(const char *s) {
    int left = 0;
    int right = (int)strlen(s) - 1;
    while (left < right) {
        if (s[left] != s[right]) return 0;
        left++;
        right--;
    }
    return 1;
}

int main(void) {
    printf("%d\n", is_palindrome("racecar"));  /* 1 */
    printf("%d\n", is_palindrome("hello"));    /* 0 */
    printf("%d\n", is_palindrome(""));         /* 1 */
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 3: Implement strlen Without the Standard Library

**Statement:** Return the number of characters before the null terminator.

**Approach:** Walk the pointer until `'\0'`.

```c
#include <stdio.h>

size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p != '\0') p++;
    return (size_t)(p - s);
}

int main(void) {
    printf("%zu\n", my_strlen("Hello"));   /* 5 */
    printf("%zu\n", my_strlen(""));        /* 0 */
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 4: Find the Largest Element Using a Pointer

**Statement:** Given an integer array and its length, return the largest element using pointer arithmetic (no subscript operator).

```c
#include <stdio.h>
#include <limits.h>

int find_max(const int *arr, int n) {
    int max = INT_MIN;
    const int *end = arr + n;
    for (const int *p = arr; p < end; p++)
        if (*p > max) max = *p;
    return max;
}

int main(void) {
    int arr[] = {3, -1, 7, 2, 9, 4};
    printf("%d\n", find_max(arr, 6));   /* 9 */
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 5: Stack with Push/Pop/Peek Using a Linked List

**Statement:** Implement a stack backed by a singly linked list (no size limit, unlike the array version).

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct SNode {
    int          val;
    struct SNode *next;
} SNode;

typedef struct { SNode *top; } LStack;

void ls_init(LStack *s)       { s->top = NULL; }
int  ls_empty(const LStack *s){ return s->top == NULL; }

void ls_push(LStack *s, int val) {
    SNode *n = malloc(sizeof(SNode));
    if (!n) exit(EXIT_FAILURE);
    n->val  = val;
    n->next = s->top;
    s->top  = n;
}

int ls_pop(LStack *s) {
    if (ls_empty(s)) { fprintf(stderr, "empty\n"); return -1; }
    SNode *tmp = s->top;
    int    val = tmp->val;
    s->top     = tmp->next;
    free(tmp);
    return val;
}

int ls_peek(const LStack *s) {
    if (ls_empty(s)) return -1;
    return s->top->val;
}

void ls_free(LStack *s) {
    while (!ls_empty(s)) ls_pop(s);
}

int main(void) {
    LStack s;
    ls_init(&s);
    ls_push(&s, 10);
    ls_push(&s, 20);
    ls_push(&s, 30);
    printf("peek: %d\n", ls_peek(&s));  /* 30 */
    printf("pop:  %d\n", ls_pop(&s));   /* 30 */
    printf("pop:  %d\n", ls_pop(&s));   /* 20 */
    ls_free(&s);
    return 0;
}
```

**Complexity:** Push/pop/peek all $O(1)$. Space $O(n)$.

---

### Problem 6: Count Character Occurrences Using Array as Hash Map

**Statement:** Count how many times each character appears in a string, using a 256-element array indexed by ASCII value.

```c
#include <stdio.h>
#include <string.h>

void count_chars(const char *s) {
    int freq[256] = {0};
    for (const char *p = s; *p; p++)
        freq[(unsigned char)*p]++;
    for (int i = 0; i < 256; i++)
        if (freq[i] > 0)
            printf("'%c' (%3d): %d\n", i, i, freq[i]);
}

int main(void) {
    count_chars("hello world");
    return 0;
}
```

Cast to `unsigned char` before indexing to avoid negative indices for chars with value > 127.

**Complexity:** Time $O(n + 256) = O(n)$, Space $O(1)$ (fixed 256-element table).

---

### Problem 7: Remove Duplicates from a Sorted Array

**Statement:** Remove duplicates in-place from a sorted integer array. Return the new length. Elements beyond the new length are irrelevant.

**Approach:** Slow/fast pointer technique. `slow` points to the last unique element; `fast` scans forward.

```c
#include <stdio.h>

int remove_duplicates(int *arr, int n) {
    if (n == 0) return 0;
    int slow = 0;
    for (int fast = 1; fast < n; fast++) {
        if (arr[fast] != arr[slow]) {
            slow++;
            arr[slow] = arr[fast];
        }
    }
    return slow + 1;
}

int main(void) {
    int arr[] = {1, 1, 2, 3, 3, 3, 4, 5, 5};
    int n = 9;
    int new_len = remove_duplicates(arr, n);
    printf("new length: %d\n", new_len);   /* 5 */
    for (int i = 0; i < new_len; i++)
        printf("%d ", arr[i]);             /* 1 2 3 4 5 */
    printf("\n");
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 8: Binary Search — Iterative and Recursive

**Statement:** Given a sorted array, return the index of `target` or `-1` if not found.

```c
#include <stdio.h>

int binary_search_iter(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;   /* avoids overflow vs (lo+hi)/2 */
        if (arr[mid] == target) return mid;
        if (arr[mid] <  target) lo = mid + 1;
        else                    hi = mid - 1;
    }
    return -1;
}

int binary_search_rec(const int *arr, int lo, int hi, int target) {
    if (lo > hi) return -1;
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] <  target) return binary_search_rec(arr, mid + 1, hi, target);
    return                         binary_search_rec(arr, lo, mid - 1, target);
}

int main(void) {
    int arr[] = {1, 3, 5, 7, 9, 11, 13};
    int n = 7;
    printf("%d\n", binary_search_iter(arr, n, 7));       /* 3 */
    printf("%d\n", binary_search_iter(arr, n, 4));       /* -1 */
    printf("%d\n", binary_search_rec(arr, 0, n-1, 11));  /* 5 */
    return 0;
}
```

**Complexity:** Time $O(\log n)$, Space $O(1)$ iterative, $O(\log n)$ recursive (call stack).

---

### Problem 9: Reverse a Linked List Iteratively

**Statement:** Reverse a singly linked list in place and return the new head.

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int          val;
    struct Node *next;
} Node;

Node *list_reverse(Node *head) {
    Node *prev = NULL;
    Node *cur  = head;
    while (cur) {
        Node *next = cur->next;
        cur->next  = prev;
        prev       = cur;
        cur        = next;
    }
    return prev;
}

Node *make(int val, Node *next) {
    Node *n = malloc(sizeof(Node));
    n->val = val; n->next = next;
    return n;
}

void print_list(const Node *h) {
    for (; h; h = h->next) printf("%d -> ", h->val);
    printf("NULL\n");
}

int main(void) {
    Node *head = make(1, make(2, make(3, make(4, make(5, NULL)))));
    print_list(head);                  /* 1 -> 2 -> 3 -> 4 -> 5 -> NULL */
    head = list_reverse(head);
    print_list(head);                  /* 5 -> 4 -> 3 -> 2 -> 1 -> NULL */
    /* free omitted for brevity */
    return 0;
}
```

**Complexity:** Time $O(n)$, Space $O(1)$.

---

### Problem 10: Simple Calculator Using Function Pointers

**Statement:** Implement `+`, `-`, `*`, `/` using a dispatch table of function pointers indexed by operator character.

```c
#include <stdio.h>
#include <string.h>

typedef double (*BinOp)(double, double);

static double op_add(double a, double b) { return a + b; }
static double op_sub(double a, double b) { return a - b; }
static double op_mul(double a, double b) { return a * b; }
static double op_div(double a, double b) {
    if (b == 0.0) { fprintf(stderr, "division by zero\n"); return 0.0; }
    return a / b;
}

static double calculate(double a, char op, double b) {
    static const char ops[]   = {'+', '-', '*', '/'};
    static const BinOp fns[]  = {op_add, op_sub, op_mul, op_div};
    static const int   nops   = 4;

    for (int i = 0; i < nops; i++)
        if (ops[i] == op) return fns[i](a, b);

    fprintf(stderr, "unknown operator '%c'\n", op);
    return 0.0;
}

int main(void) {
    printf("%.2f\n", calculate(10,  '+', 5));   /* 15.00 */
    printf("%.2f\n", calculate(10,  '-', 3));   /* 7.00  */
    printf("%.2f\n", calculate(6,   '*', 7));   /* 42.00 */
    printf("%.2f\n", calculate(22,  '/', 7));   /* 3.14  */
    printf("%.2f\n", calculate(1,   '/', 0));   /* 0.00 (error) */
    return 0;
}
```

The dispatch table decouples the operation selection from the arithmetic, making it easy to add new operators without modifying `calculate`. This is the C equivalent of a virtual method table.

**Complexity:** Lookup $O(k)$ where $k$ = number of operators (constant); operation $O(1)$.

---

## Quick Reference — Common Pitfalls

| Pitfall | Correct Pattern |
|---|---|
| `if (x = 5)` (assignment in condition) | `if (x == 5)` or `if ((x = f()) != 0)` |
| `scanf("%s", buf)` (unbounded read) | `scanf("%255s", buf)` or use `fgets` |
| `str == "hello"` (pointer comparison) | `strcmp(str, "hello") == 0` |
| Integer division: `1/2 = 0` | Cast: `(double)1 / 2 = 0.5` |
| Array out of bounds: `arr[n]` | Always check `i < n` |
| Forgetting `break` in switch | Add `break;` or comment `/* fallthrough */` |
| `free(ptr)` then use `ptr` | `free(ptr); ptr = NULL;` |
| `realloc(ptr, ...)` losing original | `tmp = realloc(ptr, ...); if (tmp) ptr = tmp;` |
| Off-by-one in `strncpy` | `strncpy(dst, src, n-1); dst[n-1] = '\0';` |
| Returning local variable address | Allocate on heap or use static |
| `sizeof` on function parameter array | Pass size explicitly as a separate parameter |
| Comparing `char` values > 127 | Cast to `unsigned char` before comparison/indexing |

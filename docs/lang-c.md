# C Programming — Systems Foundations

---

## Quick Reference

| Topic | Key Facts |
|-------|-----------|
| Memory regions | Text, Data (init), BSS (uninit), Heap (dynamic), Stack (auto) |
| Pointer size | 8 bytes on 64-bit, 4 bytes on 32-bit |
| `sizeof(array)` | Total bytes; decays to `sizeof(ptr)` when passed to function |
| `NULL` | `(void*)0`; dereferencing is UB |
| `malloc` return | `void*`; returns `NULL` on failure; not zero-initialized |
| `calloc` | Zero-initializes; `calloc(n, size)` = `n * size` bytes |
| `realloc` | May move memory; old pointer invalid after call |
| String literal | Read-only; stored in `.rodata`; modifying is UB |
| `static` (local) | Stored in Data/BSS; persists across calls |
| `volatile` | Prevents compiler from caching reads; used for MMIO/signals |
| `restrict` | Pointer aliasing hint; enables vectorization |
| Integer promotion | Operands narrower than `int` promoted before arithmetic |
| Signed overflow | UB; unsigned overflow wraps modulo $2^n$ |
| Compilation stages | Preprocessing → Compilation → Assembly → Linking |

---

## Core Concepts

### Memory Model

```
High address
┌─────────────────┐
│     Stack       │ ← grows downward; local vars, return addrs, args
│        ↓        │
│   (gap/guard)   │
│        ↑        │
│      Heap       │ ← grows upward; malloc/calloc/realloc
├─────────────────┤
│  BSS segment    │ ← uninitialized globals & statics (zeroed by OS)
├─────────────────┤
│  Data segment   │ ← initialized globals & statics
├─────────────────┤
│  Text segment   │ ← executable code (read-only)
Low address
```

Stack frames are pushed on function call and popped on return. Each frame contains: return address, saved registers, local variables, arguments (platform-dependent — x86-64 uses registers for first 6 integer args: rdi, rsi, rdx, rcx, r8, r9).

Heap fragmentation: repeated malloc/free of varying sizes leaves unusable holes. Solutions: slab allocators (Linux kernel), arena allocators (jemalloc, tcmalloc), pooling.

**BSS vs Data:**
```c
int g_uninit;          // BSS — zero-initialized by loader
int g_init = 42;       // Data segment
static int s_count;    // BSS — persists, zero-initialized
```

### Pointers & Pointer Arithmetic

A pointer stores a virtual address. Arithmetic on `T*` scales by `sizeof(T)`.

```
int arr[] = {10, 20, 30};
int *p = arr;          // p points to arr[0]
p + 1                  // address + sizeof(int) = address + 4
*(p + 2) == arr[2]     // true — pointer indexing identity
```

**Rules:**
- `p[i]` is exactly `*(p + i)` — syntactic sugar.
- Subtracting two pointers of the same type yields `ptrdiff_t` (number of elements).
- Adding two pointers is undefined.
- Pointer past one-beyond-last element is valid for comparison, not dereference.

**Pointer to pointer:**
```c
int x = 5;
int *p = &x;
int **pp = &p;
**pp = 10;   // modifies x
```

**`void*`:** Generic pointer; cannot be dereferenced or used in arithmetic without cast. `malloc` returns `void*`.

**Function pointers:**
```c
int (*cmp)(const void*, const void*);
```
Type reads right-to-left: `cmp` is a pointer to a function taking two `const void*` args and returning `int`.

### Arrays as Pointer Decay

In most expressions an array name decays to a pointer to its first element. Exceptions: `sizeof`, `&`, `_Alignof`, string literal initializer.

```c
int arr[5];
sizeof(arr)      // 20 (5 * sizeof(int))
sizeof(arr + 0)  // 8  — decayed to pointer

void f(int *p);
f(arr);          // decay; f cannot know length
```

2D arrays: `int m[3][4]` — row-major. `m[i][j]` = `*(*(m + i) + j)`. When passing: `void f(int m[][4], int rows)` — column count must be known.

### Structs & Unions

**Struct layout:** Members laid out in declaration order. Padding inserted to satisfy alignment of each member. Final struct size padded to multiple of largest member's alignment.

```c
struct Bad {
    char  a;    // 1 byte + 3 padding
    int   b;    // 4 bytes
    char  c;    // 1 byte + 7 padding (for next double)
    double d;   // 8 bytes
};              // total: 24 bytes

struct Good {
    double d;   // 8 bytes
    int    b;   // 4 bytes
    char   a;   // 1 byte
    char   c;   // 1 byte + 2 padding
};              // total: 16 bytes
```

Use `__attribute__((packed))` (GCC/Clang) to eliminate padding — risks unaligned access UB on strict-alignment architectures.

**Flexible array member (C99):**
```c
struct Buf {
    size_t len;
    char   data[];  // must be last; access via malloc'd extra space
};
```

**Union:** All members share the same memory starting at offset 0. Size = max member size (+ alignment padding). Only one member valid at a time; reading a non-written member is type-punning — use `memcpy` for safe punning in C99.

```c
union FloatBits {
    float    f;
    uint32_t u;
};
```

**Bit fields:**
```c
struct Flags {
    unsigned int rw : 2;
    unsigned int exec : 1;
    unsigned int reserved : 29;
};
```
Layout is implementation-defined; not portable across compilers/ABIs.

### Function Pointers

```c
typedef int (*Comparator)(const void*, const void*);

int int_cmp(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void sort(void *base, size_t n, size_t sz, Comparator cmp) {
    qsort(base, n, sz, cmp);
}
```

Storing function pointers in structs enables runtime dispatch — the basis of C-style OOP (see Linux `file_operations`, GLib `GObject`).

### Dynamic Memory

```c
void *malloc(size_t size);          // allocate; uninitialized
void *calloc(size_t n, size_t sz);  // allocate + zero-init
void *realloc(void *ptr, size_t sz);// resize; may relocate
void  free(void *ptr);              // return to allocator
```

**Pitfalls:**
- `realloc` returns `NULL` on failure; original block still allocated. Always use a temporary:
  ```c
  void *tmp = realloc(p, new_size);
  if (!tmp) { /* handle */ return; }
  p = tmp;
  ```
- Double-free: UB; often causes heap corruption exploitable as security vulnerability.
- Use-after-free: accessing freed memory is UB.
- Memory leak: allocated block with no reachable pointer; detectable with Valgrind.

**Alignment:** `aligned_alloc(alignment, size)` (C11). `size` must be multiple of `alignment`.

### File I/O

```c
FILE *fp = fopen("data.bin", "rb");  // modes: r, w, a, r+, w+, rb, wb
if (!fp) { perror("fopen"); return 1; }

fread(buf, sizeof(elem), count, fp);   // returns elements read
fwrite(buf, sizeof(elem), count, fp);  // returns elements written
fseek(fp, offset, SEEK_SET);           // SEEK_SET, SEEK_CUR, SEEK_END
ftell(fp);                             // current position
rewind(fp);
fclose(fp);
```

Buffered I/O: `FILE*` is buffered by default. `fflush(fp)` forces write. `setvbuf` to control buffer mode: `_IONBF` (unbuffered), `_IOLBF` (line), `_IOFBF` (full).

**Low-level POSIX:**
```c
int fd = open("file", O_RDONLY);
ssize_t n = read(fd, buf, sizeof(buf));
write(fd, buf, n);
close(fd);
```

`mmap`: maps file into virtual address space — zero-copy reads, useful for large datasets.

### Preprocessor Macros

Textual substitution before compilation. No type checking.

```c
#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define STRINGIFY(x) #x
#define CONCAT(a, b) a ## b
```

Always parenthesize arguments and entire expression to avoid operator-precedence bugs: `MAX(x++, y)` evaluates `x++` twice — use `static inline` functions instead.

**Include guards:**
```c
#ifndef MODULE_H
#define MODULE_H
/* declarations */
#endif
```

**Predefined macros:** `__FILE__`, `__LINE__`, `__func__` (C99), `__DATE__`, `__TIME__`, `__STDC_VERSION__`.

**Conditional compilation:**
```c
#ifdef DEBUG
    fprintf(stderr, "val=%d\n", val);
#endif

#if defined(__linux__)
    /* Linux-specific code */
#elif defined(__APPLE__)
    /* macOS-specific code */
#endif
```

### Compilation Pipeline

```
source.c
   │
   ▼  cpp (C Preprocessor)
source.i        ← macros expanded, #include files pasted, comments stripped
   │
   ▼  cc1 (Compiler front-end + optimizer)
source.s        ← architecture-specific assembly
   │
   ▼  as (Assembler)
source.o        ← ELF/Mach-O relocatable object: sections .text .data .bss .rodata
   │
   ▼  ld (Linker)
a.out / libfoo.so  ← resolves symbol references, sets load addresses
```

**Linker steps:** Symbol resolution (matches definitions to references), relocation (patches addresses), section merging.

**Static vs dynamic linking:**
| | Static (`ar`, `.a`) | Dynamic (`.so`, `.dll`) |
|--|--|--|
| Size | Larger binary | Smaller binary |
| Startup | Faster | Slower (dynamic linker) |
| Updates | Recompile needed | Replace `.so` only |
| Symbol conflicts | None | Possible |

**Compiler flags:** `-O2`/`-O3` (optimize), `-g` (debug info), `-Wall -Wextra` (warnings), `-fsanitize=address` (AddressSanitizer), `-fsanitize=undefined` (UBSanitizer).

### Undefined Behavior Taxonomy

UB allows the compiler to assume it never occurs — enabling optimizations that break programs relying on UB.

| Category | Example |
|----------|---------|
| Signed integer overflow | `INT_MAX + 1` |
| Null pointer dereference | `*((int*)NULL)` |
| Out-of-bounds access | `arr[n]` where n == len |
| Use-after-free | Read after `free(p)` |
| Double-free | `free(p); free(p);` |
| Uninitialized read | `int x; printf("%d", x);` |
| Data race | Two threads write same var without sync |
| Modifying string literal | `char *s = "hi"; s[0] = 'H';` |
| Violating strict aliasing | Cast `int*` to `float*` and read |
| Left-shift into sign bit | `1 << 31` (signed int) |

**Strict aliasing:** Compiler assumes pointers of different types do not alias. Exception: `char*` may alias anything. Use `memcpy` or `__attribute__((may_alias))` for safe punning.

Tools: `clang -fsanitize=undefined,address`, Valgrind, `cppcheck`.

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| `scanf("%s", buf)` buffer overflow | `scanf("%255s", buf)` or use `fgets` |
| `strlen` in loop condition | Hoist outside loop |
| Returning pointer to local variable | Return heap-allocated or static |
| Forgetting `'\0'` in manual string copy | Use `strncpy` + manual null-terminate or `strlcpy` |
| Integer truncation in `malloc(n * size)` | Check for overflow before multiply |
| Off-by-one in `strncpy` | Destination must be `len+1` bytes |
| Comparing signed/unsigned | Compiler promotes signed to unsigned; negative becomes huge |

---

## Code Examples

### Pointer Arithmetic

```c
#include <stdio.h>

int main(void) {
    int arr[] = {10, 20, 30, 40, 50};
    int *p = arr;
    int *end = arr + 5;

    while (p < end) {
        printf("addr=%p  val=%d  index=%td\n",
               (void*)p, *p, p - arr);
        p++;
    }

    int *q = &arr[4];
    printf("q - p = %td elements\n", q - arr);

    char *bytes = (char*)arr;
    printf("byte[4] of arr[1] = 0x%02x\n", (unsigned char)bytes[4 + 1]);
    return 0;
}
```

### Struct with Function Pointer

```c
#include <stdio.h>
#include <math.h>

typedef struct {
    double (*area)(double, double);
    double (*perimeter)(double, double);
    const char *name;
} Shape;

static double rect_area(double w, double h) { return w * h; }
static double rect_perim(double w, double h) { return 2.0 * (w + h); }
static double ellipse_area(double a, double b) { return M_PI * a * b; }
static double ellipse_perim(double a, double b) {
    return M_PI * (3*(a+b) - sqrt((3*a+b)*(a+3*b)));
}

void print_shape(const Shape *s, double p, double q) {
    printf("%-12s  area=%.4f  perimeter=%.4f\n",
           s->name, s->area(p, q), s->perimeter(p, q));
}

int main(void) {
    Shape rect     = { rect_area,    rect_perim,    "Rectangle" };
    Shape ellipse  = { ellipse_area, ellipse_perim, "Ellipse"   };
    Shape *shapes[] = { &rect, &ellipse };

    for (size_t i = 0; i < 2; i++)
        print_shape(shapes[i], 4.0, 3.0);
    return 0;
}
```

### Manual Linked List

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node *node_new(int val) {
    Node *n = malloc(sizeof *n);
    if (!n) { perror("malloc"); exit(1); }
    n->data = val;
    n->next = NULL;
    return n;
}

void list_push_front(Node **head, int val) {
    Node *n = node_new(val);
    n->next = *head;
    *head = n;
}

void list_append(Node **head, int val) {
    Node *n = node_new(val);
    if (!*head) { *head = n; return; }
    Node *cur = *head;
    while (cur->next) cur = cur->next;
    cur->next = n;
}

void list_free(Node *head) {
    while (head) {
        Node *tmp = head->next;
        free(head);
        head = tmp;
    }
}

void list_print(const Node *head) {
    while (head) {
        printf("%d%s", head->data, head->next ? " -> " : "\n");
        head = head->next;
    }
}

int main(void) {
    Node *head = NULL;
    list_append(&head, 1);
    list_append(&head, 2);
    list_append(&head, 3);
    list_push_front(&head, 0);
    list_print(head);
    list_free(head);
    return 0;
}
```

### Manual Stack Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int   *data;
    size_t top;
    size_t cap;
} Stack;

void stack_init(Stack *s, size_t cap) {
    s->data = malloc(cap * sizeof *s->data);
    if (!s->data) { perror("malloc"); exit(1); }
    s->top = 0;
    s->cap = cap;
}

bool stack_push(Stack *s, int val) {
    if (s->top == s->cap) {
        size_t new_cap = s->cap * 2;
        int *tmp = realloc(s->data, new_cap * sizeof *s->data);
        if (!tmp) return false;
        s->data = tmp;
        s->cap  = new_cap;
    }
    s->data[s->top++] = val;
    return true;
}

bool stack_pop(Stack *s, int *out) {
    if (s->top == 0) return false;
    *out = s->data[--s->top];
    return true;
}

bool stack_peek(const Stack *s, int *out) {
    if (s->top == 0) return false;
    *out = s->data[s->top - 1];
    return true;
}

void stack_free(Stack *s) {
    free(s->data);
    s->data = NULL;
    s->top = s->cap = 0;
}

int main(void) {
    Stack s;
    stack_init(&s, 4);
    for (int i = 1; i <= 6; i++) stack_push(&s, i * 10);

    int val;
    while (stack_pop(&s, &val))
        printf("popped: %d\n", val);

    stack_free(&s);
    return 0;
}
```

---

## Interview Q&A

**Q1: What is the difference between `malloc` and `calloc`, and when would you prefer one over the other?**

`malloc(n)` allocates `n` bytes uninitialized. `calloc(count, size)` allocates `count * size` bytes and zero-initializes the memory. Prefer `calloc` when the data will be read before full initialization (avoids UB from uninitialized reads) and when the zero state is correct default. Prefer `malloc` + explicit initialization when you'll overwrite every byte immediately — avoids unnecessary memset. `calloc` may also offer OS-level optimization: freshly mapped pages from the OS are already zeroed, so calloc can skip the memset for large new allocations.

---

**Q2: Explain strict aliasing and why it matters for compiler optimizations.**

The C standard (§6.5) states an object may only be accessed through a pointer of its type, a compatible type, `char*`, or `unsigned char*`. This allows the compiler to assume that `float *f` and `int *i` never point to the same memory, enabling independent load/store reordering and register caching. Violating strict aliasing (e.g., casting `int*` to `float*` and reading) produces UB; GCC `-O2` will silently generate wrong code. Safe workaround: use `memcpy` for type punning — modern compilers optimize it to a register move.

---

**Q3: What happens when you `free` a pointer that was obtained by offsetting a `malloc`'d pointer?**

Undefined behavior. `free` must receive the exact pointer returned by `malloc`/`calloc`/`realloc`. The allocator stores metadata (size, free-list pointers) immediately before the returned address. Passing an offset pointer corrupts the heap metadata, typically causing a crash or silent heap corruption exploitable for privilege escalation.

---

**Q4: Describe the difference between `static` applied to a global variable vs a local variable.**

On a global (file-scope) variable, `static` restricts linkage to the translation unit — the symbol is invisible to the linker and cannot be referenced from other `.o` files (internal linkage). On a local (function-scope) variable, `static` changes the storage duration: the variable lives in the Data/BSS segment (not the stack), is initialized once, and retains its value across calls. Both usages share zero-initialization for BSS-allocated statics.

---

**Q5: How does the compiler handle a 2D array `int m[3][4]` in memory, and how do you pass it correctly to a function?**

The array is a contiguous block of `3 * 4 * sizeof(int) = 48` bytes in row-major order. `m[i][j]` computes as `*(&m[0][0] + i*4 + j)`. The first dimension decays away when passed to a function; the inner dimension must be specified so the compiler can compute row stride: `void f(int m[][4], int rows)` or equivalently `void f(int (*m)[4], int rows)`. Passing as `int **` is wrong — that is a pointer to a pointer, not a 2D array.

---

**Q6: What is a dangling pointer and how does it differ from a wild pointer?**

A dangling pointer previously pointed to valid memory that has since been freed or gone out of scope (stack frame popped). A wild pointer was never initialized — its value is whatever garbage was in the register or stack slot. Both cause UB when dereferenced. Common mitigation: set freed pointers to `NULL` immediately after `free`; use `-fsanitize=address` to detect at runtime.

---

**Q7: Explain how `volatile` works and name two legitimate use cases.**

`volatile` tells the compiler that a variable may change by means outside the compiler's analysis (hardware, signal handlers, other threads). The compiler must emit an actual load/store for every access — no caching in registers, no elimination of "redundant" reads/writes. Use cases: (1) Memory-mapped I/O registers — a write to a control register must not be optimized away. (2) Variables modified by signal handlers — without `volatile`, the compiler may hoist the read out of a polling loop. Note: `volatile` does not provide atomicity or ordering for multi-threading; use `_Atomic` or `pthread` primitives instead.

---

**Q8: What is the difference between `++i` and `i++` in C, and when does the distinction matter?**

`++i` (pre-increment) increments `i` and yields the new value. `i++` (post-increment) yields the old value and increments `i` as a side effect. In a statement alone, generated code is typically identical. The distinction matters when the expression result is used: `arr[i++]` uses old `i` as index then increments; `arr[++i]` increments first. Critical: `i++ + i++` has unspecified behavior (both read and modify `i` between sequence points in C99; C11 adds sequencing rules but the expression is still problematic).

---

**Q9: How does `qsort` achieve generic sorting and what are its performance characteristics?**

`qsort(base, n, size, cmp)` sorts `n` elements of `size` bytes starting at `base`. Genericity comes from treating elements as raw byte blocks (copying via `memcpy` internally) and using a user-supplied comparator `int (*cmp)(const void*, const void*)`. The standard mandates $O(n \log n)$ average but allows $O(n^2)$ worst case — glibc uses introsort (quicksort + heapsort fallback). The comparator receives pointers to elements, not the elements themselves; casting is the caller's responsibility. Performance overhead vs type-specific sort: function-pointer indirection per comparison, plus `memcpy` for element swaps.

---

**Q10: Describe the ELF object file sections and what each contains.**

| Section | Contents |
|---------|----------|
| `.text` | Executable machine code |
| `.rodata` | Read-only data: string literals, `const` globals |
| `.data` | Initialized writable globals and statics |
| `.bss` | Uninitialized globals and statics (no file space; zeroed at load) |
| `.symtab` | Symbol table: names, addresses, sizes, binding |
| `.strtab` | String table for symbol names |
| `.rel.text` / `.rela.text` | Relocation entries: where linker must patch addresses |
| `.debug_*` | DWARF debug info (present with `-g`) |
| `.got` / `.plt` | Global offset table / procedure linkage table for dynamic linking |

The linker merges same-named sections from multiple `.o` files, resolves relocations, and produces the final executable or shared library.

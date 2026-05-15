# JavaScript — Zero to Runtime Expert

---

## 1. What JavaScript Is

JavaScript is a dynamically typed, single-threaded, garbage-collected scripting language executed by engines like V8 (Chrome/Node.js), SpiderMonkey (Firefox), and JavaScriptCore (Safari). The engine compiles source to bytecode via an interpreter, then progressively optimizes "hot" functions using a JIT compiler that generates machine code. JavaScript runs in browsers (DOM/fetch), servers (Node.js/Deno), embedded devices, and serverless platforms — everywhere, because the spec (ECMAScript) separates the language from its host environment.

---

## 2. Running JavaScript

### Browser Console

Open DevTools (`F12` or `Cmd+Option+I`) → Console tab. Type any expression and press Enter.

```javascript
> 2 + 2
< 4
> typeof "hello"
< "string"
```

### Node.js

```bash
# Run a file
node file.js

# Interactive REPL
node

# Run inline code
node -e "console.log(process.version)"
```

### Console Methods

```javascript
console.log("standard output");          // stdout
console.error("something failed");       // stderr (red in most terminals)
console.warn("non-fatal issue");         // stderr with warning styling
console.table([{ a: 1 }, { a: 2 }]);    // renders as ASCII table in terminal/browser
console.time("loop");                    // starts timer named "loop"
for (let i = 0; i < 1e6; i++) {}
console.timeEnd("loop");                 // prints "loop: 12.345ms"
console.group("section");               // indent following logs
console.groupEnd();
console.assert(1 === 2, "math broke");  // prints error only when condition is false
```

### Script Tag in HTML — `defer` vs `async`

```html
<!-- blocks HTML parsing until script downloads + executes -->
<script src="app.js"></script>

<!-- downloads in parallel; executes after HTML is fully parsed; preserves order -->
<script src="app.js" defer></script>

<!-- downloads in parallel; executes immediately when ready; order NOT guaranteed -->
<script src="app.js" async></script>
```

```
Timeline comparison (| = HTML parse, D = download, E = execute):

Normal:  |||||DDDDEEEE|||||||||  (parse blocked during download+execute)
defer:   |||||||||||||||||DDDDEEEE  (execute after parse; ordered)
async:   |||||DDDD|||||EEEE|||||   (execute whenever ready; unordered)
```

**Rule:** Use `defer` for app scripts that depend on the DOM. Use `async` for independent scripts like analytics. Never use bare `<script>` in `<head>` without one of these.

---

## 3. Variables & Types

### `var` vs `let` vs `const`

| Property | `var` | `let` | `const` |
|---|---|---|---|
| Scope | Function (or global) | Block `{}` | Block `{}` |
| Hoisting | Hoisted + initialized to `undefined` | Hoisted but NOT initialized (TDZ) | Hoisted but NOT initialized (TDZ) |
| Redeclaration | Allowed | Not allowed | Not allowed |
| Reassignment | Allowed | Allowed | Not allowed |
| On `window` (browser) | Yes (`window.x`) | No | No |
| Before ES version | ES1 | ES6 | ES6 |

### Temporal Dead Zone (TDZ)

The period between the start of a block and the declaration statement is the TDZ. Accessing a `let`/`const` variable in this zone throws `ReferenceError`.

```javascript
{
  // TDZ for `x` starts here
  console.log(x); // ReferenceError: Cannot access 'x' before initialization
  let x = 5;     // TDZ ends here
  console.log(x); // 5
}

// var has no TDZ:
{
  console.log(y); // undefined (hoisted + initialized)
  var y = 5;
  console.log(y); // 5
}
```

### The 8 Types

JavaScript has 7 primitive types and 1 structural type (`object` which includes arrays, functions, dates, null — but null is a quirk).

| Type | Example | `typeof` result |
|---|---|---|
| `undefined` | `let x;` → `x` is `undefined` | `"undefined"` |
| `null` | `let x = null` | `"object"` ← bug, not a real object |
| `boolean` | `true`, `false` | `"boolean"` |
| `number` | `42`, `3.14`, `NaN`, `Infinity` | `"number"` |
| `bigint` | `9007199254740993n` | `"bigint"` |
| `string` | `"hello"`, `` `world` `` | `"string"` |
| `symbol` | `Symbol("id")` | `"symbol"` |
| `object` | `{}`, `[]`, `function(){}` | `"object"` or `"function"` |

```javascript
typeof undefined    // "undefined"
typeof null         // "object"  ← historical bug (1995); cannot be fixed
typeof true         // "boolean"
typeof 42           // "number"
typeof 42n          // "bigint"
typeof "hi"         // "string"
typeof Symbol()     // "symbol"
typeof {}           // "object"
typeof []           // "object"  (use Array.isArray to check arrays)
typeof function(){} // "function"
typeof class C {}   // "function"
```

### Type Coercion: `==` vs `===`

`===` (strict equality): no type conversion — types must match.
`==` (abstract equality): applies Abstract Equality Comparison algorithm which coerces types.

```javascript
0 == false     // true  (false → 0)
1 == true      // true  (true → 1)
"" == false    // true  ("" → 0, false → 0)
null == undefined  // true  (spec rule)
null == 0      // false (null only == undefined)
NaN == NaN     // false (NaN is never equal to anything)
```

**Always use `===` unless you have a specific reason for `==`.**

### Truthy / Falsy

| Falsy values | Notes |
|---|---|
| `false` | literal |
| `0`, `-0`, `0n` | zero number, negative zero, zero bigint |
| `""`, `''`, ` `` ` | empty string |
| `null` | intentional absence |
| `undefined` | uninitialized |
| `NaN` | not-a-number |

Everything else is truthy, including `"0"`, `[]`, `{}`, `function(){}`.

```javascript
if ([]) console.log("truthy");  // prints — empty array is truthy
if ({}) console.log("truthy");  // prints — empty object is truthy
if ("0") console.log("truthy"); // prints — non-empty string is truthy
```

### NaN

`NaN` (Not a Number) is a `number` type value that represents the result of invalid numeric operations.

```javascript
typeof NaN          // "number"
NaN === NaN         // false — NaN is not equal to itself
isNaN("hello")      // true  — converts to number first, then checks
isNaN(undefined)    // true  — undefined → NaN
Number.isNaN("hello")  // false — no conversion, strict NaN check
Number.isNaN(NaN)      // true  — only true for actual NaN
Number.isNaN(undefined)// false
```

**Use `Number.isNaN()` not `isNaN()`** — `isNaN()` silently coerces its argument.

### `null` vs `undefined`

| | `undefined` | `null` |
|---|---|---|
| Meaning | Variable declared, no value assigned | Intentional absence of any value |
| Set by | JavaScript engine (implicit) | Developer (explicit) |
| `typeof` | `"undefined"` | `"object"` |
| JSON serialization | Omitted from object | Included as `null` |
| When you see it | Uninitialized vars, missing object keys, missing arguments | Explicit "no value" signal (e.g. DB returns no row) |

```javascript
let x;          // undefined — JS set it
let y = null;   // null — you set it

function greet(name) {
  console.log(name); // undefined if called as greet()
}

const obj = { a: 1 };
obj.b;           // undefined — key doesn't exist
```

---

## 4. Operators

### Arithmetic

```javascript
5 + 2    // 7
5 - 2    // 3
5 * 2    // 10
5 / 2    // 2.5  (always float division)
5 % 2    // 1    (remainder)
5 ** 2   // 25   (exponentiation, ES2016)
5 / 0    // Infinity
-5 / 0   // -Infinity
0 / 0    // NaN
```

### Comparison

```javascript
>   >=   <   <=   ===   !==
// All return boolean
// Strings compared lexicographically
"b" > "a"  // true
"10" > "9" // false — lexicographic, "1" < "9"
10 > 9     // true  — numeric
```

### Logical

```javascript
&&  // AND — returns first falsy, or last value
||  // OR  — returns first truthy, or last value
!   // NOT — negates boolean

// Short-circuit evaluation:
false && sideEffect()  // sideEffect never called
true || sideEffect()   // sideEffect never called

// Practical patterns:
const name = user && user.name;  // safe access before && existed ?.
const port = process.env.PORT || 3000;  // default value
```

### Nullish Coalescing `??`

Returns right side only when left side is `null` or `undefined` (not other falsy values).

```javascript
null ?? "default"      // "default"
undefined ?? "default" // "default"
0 ?? "default"         // 0       ← 0 is not null/undefined
"" ?? "default"        // ""      ← empty string is not null/undefined
false ?? "default"     // false

// Compare with ||:
0 || "default"         // "default" ← 0 is falsy
0 ?? "default"         // 0         ← 0 is not null/undefined
```

### Optional Chaining `?.`

Short-circuits to `undefined` when accessing a property on `null`/`undefined`.

```javascript
const user = { address: { city: "Berlin" } };
user?.address?.city     // "Berlin"
user?.phone?.number     // undefined (no error)
user?.greet?.()         // undefined (method call variant)
user?.["key"]           // undefined (bracket notation variant)

// Before ?. existed:
const city = user && user.address && user.address.city;
// After:
const city = user?.address?.city;
```

### Logical Assignment (ES2021)

```javascript
a &&= b  // equivalent to: if (a) a = b;
a ||= b  // equivalent to: if (!a) a = b;
a ??= b  // equivalent to: if (a == null) a = b;

// Common use:
config.timeout ??= 5000;  // set default only if not already set
obj.items ||= [];          // initialize array only if falsy
```

### Bitwise Operators

```javascript
5 & 3   // 1   (AND:  0101 & 0011 = 0001)
5 | 3   // 7   (OR:   0101 | 0011 = 0111)
5 ^ 3   // 6   (XOR:  0101 ^ 0011 = 0110)
~5      // -6  (NOT:  ~00000101 = 11111010 in two's complement)
5 << 1  // 10  (left shift = multiply by 2)
5 >> 1  // 2   (right shift = divide by 2, preserves sign)
-5 >>> 1// 2147483645 (unsigned right shift — fills 0 on left)
```

### Comma Operator

Evaluates each operand left to right, returns the last.

```javascript
let x = (1, 2, 3);  // x === 3
for (let i = 0, j = 10; i < j; i++, j--) { /* i,j updated together */ }
```

### Spread `...` and Rest `...`

Same syntax, different contexts:

```javascript
// Spread: expand iterable into elements
const arr = [1, 2, 3];
const arr2 = [...arr, 4, 5];         // [1, 2, 3, 4, 5]
const obj = { a: 1 };
const obj2 = { ...obj, b: 2 };       // { a: 1, b: 2 }
Math.max(...arr);                     // spreads into arguments

// Rest: collect remaining elements into array
function sum(...nums) {               // rest parameter
  return nums.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4);  // 10

const [first, ...rest] = [1, 2, 3];  // rest in destructuring
// first = 1, rest = [2, 3]
```

### Destructuring

```javascript
// Array destructuring
const [a, b] = [1, 2];
const [x, , z] = [1, 2, 3];           // skip index 1
const [p = 10, q = 20] = [5];          // defaults: p=5, q=20
const [first, ...remaining] = [1, 2, 3, 4];  // rest

// Swap without temp variable
let m = 1, n = 2;
[m, n] = [n, m];  // m=2, n=1

// Object destructuring
const { name, age } = { name: "Alice", age: 30 };
const { name: fullName } = { name: "Alice" };  // rename: fullName="Alice"
const { city = "Unknown" } = {};               // default: city="Unknown"

// Nested
const { address: { street } } = { address: { street: "Main St" } };

// Function parameter destructuring
function display({ name, age = 0 }) {
  return `${name} is ${age}`;
}
display({ name: "Bob", age: 25 });  // "Bob is 25"

// Mixed
const { a: [first2] } = { a: [1, 2, 3] };  // first2 = 1
```

---

## 5. Control Flow

### if / else

```javascript
if (score >= 90) {
  grade = "A";
} else if (score >= 80) {
  grade = "B";
} else {
  grade = "F";
}
```

### switch

Uses strict equality (`===`) for comparison. Fallthrough is intentional when `break` is omitted.

```javascript
switch (day) {
  case "Mon":
  case "Tue":
  case "Wed":
  case "Thu":
  case "Fri":
    console.log("Weekday");  // fallthrough across all cases
    break;
  case "Sat":
  case "Sun":
    console.log("Weekend");
    break;
  default:
    console.log("Unknown");
}
```

### for Loops

```javascript
// Classic for
for (let i = 0; i < 5; i++) {
  if (i === 3) continue;  // skip 3
  if (i === 4) break;     // stop at 4
  console.log(i);         // 0, 1, 2
}

// for...in — iterates over enumerable KEYS (string keys)
const obj = { a: 1, b: 2 };
for (const key in obj) {
  console.log(key);  // "a", "b"
}
// Warning: also iterates prototype chain keys — use hasOwnProperty or for...of Object.keys()

// for...of — iterates over VALUES of any iterable (Array, Map, Set, string, generator)
for (const val of [10, 20, 30]) {
  console.log(val);  // 10, 20, 30
}
for (const char of "hi") {
  console.log(char); // "h", "i"
}

// for...of with entries() — index + value
for (const [i, val] of ["a", "b", "c"].entries()) {
  console.log(i, val); // 0 "a", 1 "b", 2 "c"
}

// for...of with Map
const map = new Map([["x", 1], ["y", 2]]);
for (const [key, val] of map) {
  console.log(key, val);
}
```

### while / do...while

```javascript
let i = 0;
while (i < 3) {
  console.log(i++);  // 0, 1, 2
}

// do...while: executes body at least once
let j = 5;
do {
  console.log(j);    // 5
  j++;
} while (j < 5);    // condition false, exits
```

### Labels with break/continue

```javascript
outer: for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    if (j === 1) continue outer;  // skip to next i iteration
    if (i === 2) break outer;     // exit both loops
    console.log(i, j);
  }
}
```

### Ternary and Short-Circuit Patterns

```javascript
const label = isAdmin ? "Admin" : "User";  // ternary

// Short-circuit for conditional execution
isLoggedIn && renderDashboard();
config ?? (config = defaultConfig);

// Guard clause pattern (prefer over deeply nested if/else)
function process(data) {
  if (!data) return;          // guard
  if (data.error) throw data.error;  // guard
  return data.result;
}
```

---

## 6. Functions

### Function Types Comparison

| Property | Declaration | Expression | Arrow |
|---|---|---|---|
| Syntax | `function f(){}` | `const f = function(){}` | `const f = () => {}` |
| Hoisted | Yes (fully) | No (TDZ with let/const) | No |
| `this` | Dynamic (call-site) | Dynamic (call-site) | Lexical (from outer scope) |
| `arguments` object | Yes | Yes | No |
| Can be constructor (`new`) | Yes | Yes | No |
| `prototype` property | Yes | Yes | No |
| Named | Yes | Optional | No (name inferred) |

### Hoisting

```javascript
// Function declaration: fully hoisted
greet();  // works — "Hello"
function greet() { console.log("Hello"); }

// Function expression: not hoisted (TDZ with let)
greet2();  // ReferenceError
const greet2 = function() { console.log("Hello"); };
```

### `arguments` Object

Only available in non-arrow functions. Array-like but not an array.

```javascript
function sum() {
  let total = 0;
  for (let i = 0; i < arguments.length; i++) {
    total += arguments[i];
  }
  return total;
}
sum(1, 2, 3);  // 6

// Convert to real array:
const args = Array.from(arguments);
// or
const args2 = [...arguments];
```

### Default Parameters

```javascript
function greet(name = "World", punctuation = "!") {
  return `Hello, ${name}${punctuation}`;
}
greet();           // "Hello, World!"
greet("Alice");    // "Hello, Alice!"
greet(undefined, "?");  // "Hello, World?" — undefined triggers default
greet(null, "?");       // "Hello, null?" — null does NOT trigger default
```

### Rest Parameters

```javascript
function first(a, b, ...rest) {
  console.log(a, b, rest);
}
first(1, 2, 3, 4, 5);  // 1 2 [3, 4, 5]
// rest is a real Array, unlike arguments
```

### `this` Binding — 4 Rules

Determined at **call site**, not definition site (except arrow functions).

```javascript
// Rule 1: Default binding — standalone call → this = global (or undefined in strict mode)
function show() { console.log(this); }
show();  // window (browser) or global (Node) — or undefined in strict mode

// Rule 2: Implicit binding — method call → this = object before the dot
const obj = {
  name: "Alice",
  greet() { console.log(this.name); }
};
obj.greet();  // "Alice"

// Implicit binding can be lost:
const fn = obj.greet;
fn();  // undefined (or window.name in non-strict) — no object before dot

// Rule 3: Explicit binding — call/apply/bind
function greet(greeting) { return `${greeting}, ${this.name}`; }
const user = { name: "Bob" };

greet.call(user, "Hello");      // "Hello, Bob" — calls immediately, args spread
greet.apply(user, ["Hello"]);   // "Hello, Bob" — calls immediately, args as array
const bound = greet.bind(user); // returns new function with this fixed
bound("Hi");                    // "Hi, Bob"

// Rule 4: new binding — constructor call → this = new object
function Person(name) {
  this.name = name;
}
const p = new Person("Carol");
p.name;  // "Carol"
```

**Priority:** `new` > explicit (`bind/call/apply`) > implicit (method) > default

### Arrow Functions and Lexical `this`

Arrow functions capture `this` from the enclosing lexical scope at definition time — they do not have their own `this`.

```javascript
const timer = {
  count: 0,
  start() {
    // Regular function — this is lost in callback:
    setInterval(function() {
      this.count++;  // Error: this is window/undefined
    }, 1000);

    // Arrow function — this captured from start():
    setInterval(() => {
      this.count++;  // Correct: this = timer
    }, 1000);
  }
};
```

### IIFE (Immediately Invoked Function Expression)

```javascript
(function() {
  const private = "hidden";
  console.log(private);
})();  // executes immediately, scope isolated

// Arrow IIFE:
(() => {
  console.log("IIFE");
})();
```

### Closures

A closure is a function that retains access to variables from its outer (enclosing) scope even after the outer function has returned.

```javascript
function makeCounter() {
  let count = 0;           // closed over by returned function
  return function() {
    return ++count;
  };
}

const counter = makeCounter();
counter();  // 1
counter();  // 2
counter();  // 3
// count is private — unreachable from outside
```

### Higher-Order Functions — Implement from Scratch

```javascript
// map: transform each element
Array.prototype.myMap = function(callback) {
  const result = [];
  for (let i = 0; i < this.length; i++) {
    result.push(callback(this[i], i, this));
  }
  return result;
};

// filter: keep elements where callback returns true
Array.prototype.myFilter = function(callback) {
  const result = [];
  for (let i = 0; i < this.length; i++) {
    if (callback(this[i], i, this)) result.push(this[i]);
  }
  return result;
};

// reduce: accumulate into single value
Array.prototype.myReduce = function(callback, initialValue) {
  let acc = initialValue !== undefined ? initialValue : this[0];
  let start = initialValue !== undefined ? 0 : 1;
  for (let i = start; i < this.length; i++) {
    acc = callback(acc, this[i], i, this);
  }
  return acc;
};

[1, 2, 3].myMap(x => x * 2);              // [2, 4, 6]
[1, 2, 3, 4].myFilter(x => x % 2 === 0); // [2, 4]
[1, 2, 3, 4].myReduce((sum, x) => sum + x, 0); // 10
```

---

## 7. Scope & Closures (Deep)

### Lexical Scoping

Variable lookup is determined by where code is **written** (lexical position), not where it is called.

```javascript
const x = 1;

function outer() {
  const x = 2;
  function inner() {
    console.log(x);  // 2 — resolves to outer's x, not global x
  }
  inner();
}
outer();
```

### Scope Chain

When a variable is referenced, the engine walks the scope chain outward until it finds the variable or reaches global scope.

```
inner() scope → outer() scope → module scope → global scope → undefined/error
```

### Closure Over Loop Variable Bug

```javascript
// Bug with var (all callbacks share same i):
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100);  // 3, 3, 3
}

// Fix 1: use let (block-scoped, new binding per iteration):
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100);  // 0, 1, 2
}

// Fix 2: IIFE to capture current value:
for (var i = 0; i < 3; i++) {
  (function(j) {
    setTimeout(() => console.log(j), 100);  // 0, 1, 2
  })(i);
}
```

### Module Pattern Using Closures

```javascript
const BankAccount = (function() {
  let balance = 0;  // private

  return {
    deposit(amount) { balance += amount; },
    withdraw(amount) {
      if (amount > balance) throw new Error("Insufficient funds");
      balance -= amount;
    },
    getBalance() { return balance; }
  };
})();

BankAccount.deposit(100);
BankAccount.getBalance();  // 100
// balance is not directly accessible
```

### Memoization Using Closures

```javascript
function memoize(fn) {
  const cache = new Map();
  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}

const fib = memoize(function(n) {
  if (n <= 1) return n;
  return fib(n - 1) + fib(n - 2);
});

fib(40);  // fast — results cached
```

### Partial Application Using Closures

```javascript
function partial(fn, ...presetArgs) {
  return function(...laterArgs) {
    return fn(...presetArgs, ...laterArgs);
  };
}

function multiply(a, b, c) { return a * b * c; }

const double = partial(multiply, 2);
double(3, 4);  // 24 — calls multiply(2, 3, 4)

const triple = partial(multiply, 3);
triple(5, 6);  // 90 — calls multiply(3, 5, 6)
```

---

## 8. Objects & Prototypes

### Object Literal

```javascript
const key = "dynamic";
const obj = {
  name: "Alice",        // regular property
  age: 30,
  greet() {             // method shorthand (ES6)
    return `Hi, ${this.name}`;
  },
  [key]: "value",       // computed key — evaluates key variable
  get fullName() {      // getter
    return this.name.toUpperCase();
  },
  set fullName(v) {     // setter
    this.name = v.toLowerCase();
  }
};
```

### Property Descriptors

Every property has a descriptor with 4 attributes:

| Attribute | Default | Meaning |
|---|---|---|
| `value` | `undefined` | The property value |
| `writable` | `true` | Can the value be changed? |
| `enumerable` | `true` | Shows in `for...in`, `Object.keys` |
| `configurable` | `true` | Can the property be deleted or descriptor changed? |

```javascript
const obj = {};
Object.defineProperty(obj, "PI", {
  value: 3.14159,
  writable: false,      // cannot reassign
  enumerable: true,
  configurable: false   // cannot delete or redefine
});

obj.PI = 0;       // silently fails in sloppy mode; TypeError in strict mode
delete obj.PI;    // false, fails silently
Object.keys(obj); // ["PI"] — enumerable

// Get descriptor:
Object.getOwnPropertyDescriptor(obj, "PI");
// { value: 3.14159, writable: false, enumerable: true, configurable: false }
```

### Prototype Chain

Every JavaScript object has an internal slot `[[Prototype]]` pointing to another object (or `null`). Property lookup walks this chain.

```
myObj
  │ [[Prototype]]
  ▼
Constructor.prototype
  │ [[Prototype]]
  ▼
Object.prototype
  │ [[Prototype]]
  ▼
null
```

```javascript
const animal = { breathes: true };
const dog = Object.create(animal);  // dog.[[Prototype]] = animal
dog.bark = function() { return "woof"; };

dog.breathes;  // true — found on animal via prototype chain
dog.bark();    // "woof" — own property

// Access prototype:
Object.getPrototypeOf(dog) === animal;  // true
dog.__proto__ === animal;               // true (deprecated, avoid)
```

### `hasOwnProperty` vs `in`

```javascript
"breathes" in dog;              // true — checks own + prototype chain
dog.hasOwnProperty("breathes"); // false — only checks own properties
dog.hasOwnProperty("bark");     // true

// Modern alternative:
Object.hasOwn(dog, "bark");     // true (ES2022, avoids prototype manipulation issues)
```

### `Object.create`, `Object.assign`, Spread Merge

```javascript
// Create with specific prototype:
const proto = { greet() { return "hi"; } };
const obj = Object.create(proto);
obj.name = "Alice";

// Shallow clone + merge:
const base = { a: 1, b: 2 };
const extra = { b: 99, c: 3 };
const merged1 = Object.assign({}, base, extra);  // { a:1, b:99, c:3 }
const merged2 = { ...base, ...extra };           // same result

// Object.assign mutates first argument — use {} as target for clone
const clone = Object.assign({}, base);
```

### Object Utility Methods

```javascript
const person = { name: "Alice", age: 30, city: "Berlin" };

Object.keys(person);    // ["name", "age", "city"]
Object.values(person);  // ["Alice", 30, "Berlin"]
Object.entries(person); // [["name","Alice"], ["age",30], ["city","Berlin"]]

// fromEntries: inverse of entries
const doubled = Object.fromEntries(
  Object.entries(person).map(([k, v]) => [k, typeof v === "number" ? v * 2 : v])
);
// { name: "Alice", age: 60, city: "Berlin" }

Object.freeze(person);   // makes all properties non-writable + non-configurable
Object.seal(person);     // prevents adding/deleting properties but allows modification
```

### Constructor Functions and `new`

```javascript
function Person(name, age) {
  this.name = name;
  this.age = age;
}
Person.prototype.greet = function() {
  return `Hi, I'm ${this.name}`;
};

const alice = new Person("Alice", 30);
```

What `new` does (4 steps):
```
1. Creates a new empty object: {}
2. Sets its [[Prototype]] to Constructor.prototype
3. Executes Constructor with this = new object
4. Returns new object (unless Constructor explicitly returns an object)
```

### ES6 Class Syntax

Class is syntactic sugar over constructor functions and prototype assignment.

```javascript
class Person {
  #secret = "private";        // private field (ES2022)
  static count = 0;           // static field

  constructor(name, age) {
    this.name = name;
    this.age = age;
    Person.count++;
  }

  greet() {                   // added to Person.prototype
    return `Hi, I'm ${this.name}`;
  }

  get info() {                // getter on prototype
    return `${this.name}, ${this.age}`;
  }

  static create(name) {       // static method — on class, not instances
    return new Person(name, 0);
  }

  getSecret() {
    return this.#secret;      // private field — only accessible inside class
  }
}

// Desugared equivalent (what JS actually creates):
// function Person(name, age) { this.name = name; this.age = age; }
// Person.prototype.greet = function() { ... };
```

### Inheritance with `extends` and `super`

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }
  speak() {
    return `${this.name} makes a sound`;
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name);             // must call super() before using this
    this.breed = breed;
  }
  speak() {
    return `${super.speak()} — woof!`;  // super.method() calls parent
  }
}

const d = new Dog("Rex", "Labrador");
d.speak();  // "Rex makes a sound — woof!"
d instanceof Dog;    // true
d instanceof Animal; // true — prototype chain
```

---

## 9. Arrays (Deep)

### Creation

```javascript
const a = [1, 2, 3];                  // literal
const b = new Array(3);               // [empty × 3] — sparse!
const c = Array.from({ length: 3 }, (_, i) => i);  // [0, 1, 2]
const d = Array.of(3);               // [3] — unlike new Array(3)
const e = [1, , 3];                  // sparse: index 1 is empty slot
e[1] === undefined;                  // true, but 1 in e is false
```

### Mutation Methods (modify original array)

```javascript
const arr = [1, 2, 3];

arr.push(4);           // add to end → [1,2,3,4], returns new length
arr.pop();             // remove from end → returns 4, arr=[1,2,3]
arr.unshift(0);        // add to beginning → [0,1,2,3], returns new length
arr.shift();           // remove from beginning → returns 0, arr=[1,2,3]

// splice(start, deleteCount, ...items)
arr.splice(1, 1);      // remove 1 element at index 1 → arr=[1,3]
arr.splice(1, 0, 2);   // insert 2 at index 1, delete 0 → arr=[1,2,3]
arr.splice(1, 2, 9, 8);// replace indices 1-2 → arr=[1,9,8]

[3, 1, 2].sort();                        // [1, 2, 3] — default: lexicographic!
[10, 9, 100].sort();                     // [10, 100, 9] — WRONG for numbers
[10, 9, 100].sort((a, b) => a - b);     // [9, 10, 100] — correct numeric sort
[10, 9, 100].sort((a, b) => b - a);     // [100, 10, 9] — descending

[1, 2, 3].reverse();                     // [3, 2, 1] — mutates in place

[0, 0, 0].fill(7);                       // [7, 7, 7]
[1, 2, 3, 4].fill(0, 1, 3);             // [1, 0, 0, 4] — fill 0 from index 1 to 3

[1, 2, 3, 4].copyWithin(0, 2);          // [3, 4, 3, 4] — copy from index 2 to position 0
```

### Non-Mutation Methods (return new array)

```javascript
[1, 2, 3].slice(1);        // [2, 3]
[1, 2, 3].slice(1, 2);     // [2]
[1, 2, 3].slice(-1);       // [3]

[1, 2].concat([3, 4], [5]);  // [1, 2, 3, 4, 5]
[1, [2, [3]]].flat();        // [1, 2, [3]] — one level
[1, [2, [3]]].flat(Infinity);// [1, 2, 3] — all levels

[1, 2, 3].flatMap(x => [x, x * 2]);  // [1,2, 2,4, 3,6] — map then flatten one level

[1, 2, 3].join("-");   // "1-2-3"
[1, 2, 3].join("");    // "123"
```

### Search Methods

```javascript
const arr = [1, 2, 3, 2, 1];

arr.indexOf(2);              // 1 — first occurrence, -1 if not found
arr.lastIndexOf(2);          // 3 — last occurrence
arr.includes(3);             // true
arr.includes(NaN);           // false (indexOf uses ===)
[NaN].includes(NaN);         // true — includes uses SameValueZero

arr.find(x => x > 2);        // 3 — first element matching predicate
arr.findIndex(x => x > 2);   // 2 — index of first match
arr.findLast(x => x < 3);    // 2 — last match (ES2023)
arr.findLastIndex(x => x < 3);// 3 — index of last match (ES2023)
```

### Iteration Methods

```javascript
[1, 2, 3].forEach((val, i, arr) => console.log(val));  // no return value
[1, 2, 3].map(x => x * 2);                             // [2, 4, 6]
[1, 2, 3, 4].filter(x => x % 2 === 0);                // [2, 4]
[1, 2, 3, 4].reduce((acc, x) => acc + x, 0);          // 10
[1, 2, 3, 4].reduceRight((acc, x) => acc + x, 0);     // 10 (right to left)
[1, 2, 3].some(x => x > 2);                            // true — at least one match
[1, 2, 3].every(x => x > 0);                           // true — all match
```

### Sorting Gotcha

```javascript
// Default sort converts to strings — always wrong for numbers:
[10, 9, 2, 100].sort()  // [10, 100, 2, 9] — "1" < "2" < "9"

// Correct numeric sort:
[10, 9, 2, 100].sort((a, b) => a - b);  // [2, 9, 10, 100]

// Sort strings (locale-aware):
["café", "apple", "banana"].sort((a, b) => a.localeCompare(b));

// Sort objects:
const people = [{ age: 30 }, { age: 20 }, { age: 25 }];
people.sort((a, b) => a.age - b.age);
```

### Array Destructuring with Rest

```javascript
const [head, ...tail] = [1, 2, 3, 4];  // head=1, tail=[2,3,4]
const [, second, , fourth] = [1, 2, 3, 4];  // second=2, fourth=4
```

---

## 10. The Event Loop (Deep)

JavaScript is single-threaded: one call stack, one thing executing at a time. The event loop enables concurrency by coordinating the call stack with queues.

### Components

```
┌─────────────────────────────────────────────────────┐
│                   JavaScript Engine (V8)            │
│                                                     │
│  ┌─────────────┐        ┌───────────────────────┐  │
│  │  Call Stack │        │      Heap (Memory)    │  │
│  │─────────────│        │   Objects live here   │  │
│  │ frame n     │        └───────────────────────┘  │
│  │ frame n-1   │                                   │
│  │ ...         │                                   │
│  └─────────────┘                                   │
└─────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│              Web APIs / Node.js APIs               │
│   setTimeout, fetch, DOM events, fs, http, ...     │
└────────────────────────────────────────────────────┘

┌──────────────────────┐   ┌────────────────────────┐
│  Microtask Queue     │   │  Macrotask Queue        │
│  Promise callbacks   │   │  setTimeout callbacks   │
│  queueMicrotask      │   │  setInterval callbacks  │
│  MutationObserver    │   │  I/O callbacks          │
└──────────────────────┘   └────────────────────────┘
```

### Event Loop Algorithm

```
while (true) {
  1. Execute all synchronous code on the call stack until empty
  2. Drain the microtask queue completely:
       - Execute each microtask
       - If a microtask adds more microtasks, execute those too
       - Repeat until microtask queue is empty
  3. (Browser only) Run rendering steps if a frame is due
  4. Take ONE task from the macrotask queue and push to call stack
  5. Go to step 2
}
```

**Key rule:** Every microtask must complete before the next macrotask begins. One macrotask per loop iteration, then all microtasks.

### `setTimeout(fn, 0)`

Does NOT execute immediately. It schedules `fn` in the macrotask queue after the minimum delay (≥0ms, typically ≥1ms in browsers, ≥1ms in Node). All current synchronous code and queued microtasks run first.

### Worked Examples — Predict the Output

**Example 1: Basic order**
```javascript
console.log("1");
setTimeout(() => console.log("2"), 0);
console.log("3");
// Output: 1, 3, 2
// Reason: sync runs first (1,3), then macrotask (2)
```

**Example 2: Microtask vs macrotask**
```javascript
console.log("start");
setTimeout(() => console.log("timeout"), 0);
Promise.resolve().then(() => console.log("promise"));
console.log("end");
// Output: start, end, promise, timeout
// Reason: sync(start,end) → microtask(promise) → macrotask(timeout)
```

**Example 3: Microtask starvation**
```javascript
function loop() {
  Promise.resolve().then(loop);  // adds microtask each iteration
}
loop();
// setTimeout NEVER fires — microtask queue never empties
```

**Example 4: Multiple promises**
```javascript
Promise.resolve()
  .then(() => {
    console.log("a");
    return Promise.resolve("b");
  })
  .then(v => console.log(v));

Promise.resolve().then(() => console.log("c"));
// Output: a, c, b
// Reason: "a" queued, "c" queued, then "a" runs, queues resolution of "b" as new microtask,
//         "c" runs, THEN "b" runs
```

**Example 5: async/await order**
```javascript
async function main() {
  console.log("start");
  await Promise.resolve();
  console.log("after await");
}
main();
console.log("sync");
// Output: start, sync, after await
// Reason: async runs sync until await, which suspends and schedules continuation as microtask
```

---

## 11. Promises

### Callback Hell → Promise Motivation

```javascript
// Callback hell (pyramid of doom):
getUser(id, function(user) {
  getPosts(user.id, function(posts) {
    getComments(posts[0].id, function(comments) {
      // 3 levels deep — error handling duplicated everywhere
    });
  });
});

// Promise chain:
getUser(id)
  .then(user => getPosts(user.id))
  .then(posts => getComments(posts[0].id))
  .then(comments => console.log(comments))
  .catch(err => console.error(err));  // single error handler
```

### Promise States

```
           ┌──────────┐
           │ pending  │  ← initial state
           └────┬─────┘
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  fulfilled   │  │   rejected   │
│ (with value) │  │ (with reason)│
└──────────────┘  └──────────────┘
```

Once settled (fulfilled or rejected), a Promise is immutable — it never changes state.

### Core Methods

```javascript
const p = new Promise((resolve, reject) => {
  // executor runs synchronously
  if (Math.random() > 0.5) {
    resolve("success");  // fulfill with value
  } else {
    reject(new Error("failed"));  // reject with error
  }
});

p.then(value => console.log(value))   // runs if fulfilled
 .catch(err => console.error(err))    // runs if rejected
 .finally(() => console.log("done")); // always runs

// then() returns a new Promise — enables chaining:
fetch(url)
  .then(res => res.json())   // returns Promise<parsed json>
  .then(data => data.users)  // transforms value
  .catch(console.error);
```

### Promise Static Methods

| Method | Behavior | Short-circuit |
|---|---|---|
| `Promise.all(arr)` | Resolves when ALL resolve; rejects if ANY rejects | On first rejection |
| `Promise.allSettled(arr)` | Waits for ALL to settle (fulfill or reject) | Never — always gets all results |
| `Promise.race(arr)` | Settles with FIRST to settle (either way) | On first settle |
| `Promise.any(arr)` | Resolves with FIRST to fulfill; rejects if ALL reject | On first fulfillment |

```javascript
const p1 = Promise.resolve(1);
const p2 = Promise.resolve(2);
const p3 = Promise.reject("err");

Promise.all([p1, p2]);         // resolves → [1, 2]
Promise.all([p1, p3]);         // rejects → "err"
Promise.allSettled([p1, p3]);  // resolves → [{status:"fulfilled",value:1}, {status:"rejected",reason:"err"}]
Promise.race([p1, p3]);        // resolves → 1 (p1 resolves first since it's already resolved)
Promise.any([p3, p1]);         // resolves → 1 (first fulfillment)
Promise.any([p3]);             // rejects with AggregateError (all rejected)
```

### Promise Executor: Synchronous or Asynchronous?

The executor function (`(resolve, reject) => {...}`) runs **synchronously**. Only the callbacks passed to `.then()` run asynchronously.

```javascript
console.log("1");
new Promise((resolve) => {
  console.log("2");  // synchronous — runs now
  resolve();
}).then(() => console.log("3"));  // async — runs after sync code
console.log("4");
// Output: 1, 2, 4, 3
```

### Implement Promise from Scratch (Simplified)

```javascript
class MyPromise {
  #state = "pending";
  #value = undefined;
  #handlers = [];

  constructor(executor) {
    const resolve = (value) => {
      if (this.#state !== "pending") return;
      this.#state = "fulfilled";
      this.#value = value;
      this.#handlers.forEach(h => h.onFulfilled && queueMicrotask(() => h.onFulfilled(value)));
    };
    const reject = (reason) => {
      if (this.#state !== "pending") return;
      this.#state = "rejected";
      this.#value = reason;
      this.#handlers.forEach(h => h.onRejected && queueMicrotask(() => h.onRejected(reason)));
    };
    try { executor(resolve, reject); }
    catch (e) { reject(e); }
  }

  then(onFulfilled, onRejected) {
    return new MyPromise((resolve, reject) => {
      const handle = (fn, settle) => (value) => {
        if (!fn) return settle(value);
        try { resolve(fn(value)); }
        catch (e) { reject(e); }
      };
      const handler = {
        onFulfilled: handle(onFulfilled, resolve),
        onRejected: handle(onRejected, reject)
      };
      if (this.#state === "fulfilled") queueMicrotask(() => handler.onFulfilled(this.#value));
      else if (this.#state === "rejected") queueMicrotask(() => handler.onRejected(this.#value));
      else this.#handlers.push(handler);
    });
  }

  catch(onRejected) { return this.then(null, onRejected); }
  finally(fn) {
    return this.then(
      v => MyPromise.resolve(fn()).then(() => v),
      r => MyPromise.resolve(fn()).then(() => { throw r; })
    );
  }

  static resolve(value) { return new MyPromise(res => res(value)); }
  static reject(reason) { return new MyPromise((_, rej) => rej(reason)); }
}
```

### Common Mistakes

```javascript
// Mistake 1: Forgetting to return in .then()
fetch(url)
  .then(res => {
    res.json();    // ← missing return! next .then gets undefined
  })
  .then(data => console.log(data));  // undefined

// Fix:
fetch(url)
  .then(res => res.json())  // arrow function returns implicitly
  .then(data => console.log(data));

// Mistake 2: Unhandled rejections
const p = Promise.reject("error");  // no .catch() → UnhandledPromiseRejection warning

// Mistake 3: Nested promises instead of chaining
fetch(url).then(res => {
  return res.json().then(data => {  // unnecessary nesting
    return data;
  });
});
// Better:
fetch(url).then(res => res.json()).then(data => data);
```

---

## 12. Async/Await

### `async` Function

An `async` function always returns a Promise. If it returns a non-Promise value, it wraps it in `Promise.resolve()`. If it throws, it returns a rejected Promise.

```javascript
async function greet() {
  return "Hello";  // implicitly returns Promise.resolve("Hello")
}
greet().then(console.log);  // "Hello"

async function fail() {
  throw new Error("oops");  // implicitly returns Promise.reject(new Error("oops"))
}
fail().catch(console.error);
```

### `await`

`await` can only be used inside an `async` function (or top-level in ES modules). It pauses the async function's execution, yields control back to the caller, and resumes when the Promise settles.

```javascript
async function getData() {
  const res = await fetch("https://api.example.com/data");  // suspends here
  const data = await res.json();                             // suspends here
  return data;
}
```

`await` does NOT block the thread — other code continues while awaiting.

### Error Handling

```javascript
// try/catch:
async function fetchUser(id) {
  try {
    const res = await fetch(`/users/${id}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Failed:", err.message);
    throw err;  // re-throw if caller needs to handle it
  }
}

// .catch() on the returned Promise:
fetchUser(1).catch(err => console.error(err));
```

### Parallel vs Sequential

```javascript
// Sequential — total time = time(a) + time(b):
async function sequential() {
  const a = await fetch("/api/a");
  const b = await fetch("/api/b");
  return [a, b];
}

// Parallel — total time = max(time(a), time(b)):
async function parallel() {
  const [a, b] = await Promise.all([fetch("/api/a"), fetch("/api/b")]);
  return [a, b];
}

// Mixed: start both, then await in order:
async function mixed() {
  const promA = fetch("/api/a");  // starts immediately
  const promB = fetch("/api/b");  // starts immediately
  const a = await promA;
  const b = await promB;
  return [a, b];
}
```

### `for await...of`

Iterates over async iterables (e.g., streams, generators that yield Promises).

```javascript
async function processStream(stream) {
  for await (const chunk of stream) {
    process(chunk);
  }
}

// Async generator:
async function* paginate(url) {
  let page = 1;
  while (true) {
    const res = await fetch(`${url}?page=${page}`);
    const data = await res.json();
    if (!data.length) return;
    yield data;
    page++;
  }
}

for await (const page of paginate("/api/items")) {
  console.log(page);
}
```

### Top-Level Await (ES2022)

Available in ES modules (`.mjs` or `type="module"` in package.json).

```javascript
// In a module file:
const config = await fetch("/config.json").then(r => r.json());
export default config;  // other modules that import this wait for it
```

---

## 13. Modules (ES Modules & CommonJS)

### CommonJS (CJS) — Node.js original

```javascript
// math.js
function add(a, b) { return a + b; }
const PI = 3.14159;
module.exports = { add, PI };  // or assign directly
module.exports.square = x => x * x;

// app.js
const { add, PI } = require("./math");  // synchronous, cached after first load
const math = require("./math");         // same cached object
```

### ES Modules (ESM) — modern standard

```javascript
// math.mjs
export function add(a, b) { return a + b; }
export const PI = 3.14159;
export default class Calculator { /* ... */ }  // one default export per file

// app.mjs
import { add, PI } from "./math.mjs";
import Calculator from "./math.mjs";          // default import
import * as math from "./math.mjs";           // namespace import
import { add as plus } from "./math.mjs";     // rename
```

### CJS vs ESM Comparison

| Property | CJS (`require`) | ESM (`import`) |
|---|---|---|
| Loading | Synchronous | Asynchronous (statically analyzed) |
| Analysis | Dynamic (runtime) | Static (parse time) |
| Tree shaking | No | Yes (unused exports eliminated) |
| Circular deps | Partially supported | Supported (live bindings) |
| Live bindings | No (snapshot) | Yes (exported value updates) |
| Top-level await | No | Yes (ES2022) |
| `this` at top level | `module.exports` | `undefined` |
| File extension | `.js`, `.cjs` | `.mjs` or `"type":"module"` in package.json |
| Default in Node.js | Yes (legacy) | Yes (with config) |

### Dynamic `import()`

Loads a module at runtime — returns a Promise. Enables code splitting.

```javascript
// Load only when needed:
button.addEventListener("click", async () => {
  const { ChartModule } = await import("./chart.js");
  ChartModule.render(data);
});

// Conditional loading:
const locale = "fr";
const messages = await import(`./locales/${locale}.js`);
```

### `import.meta`

```javascript
import.meta.url;     // URL of current module file
import.meta.dirname; // directory of current module (Node ≥21.2)
import.meta.filename;// path of current module file (Node ≥21.2)
```

---

## 14. Error Handling

### Error Types

| Type | When it occurs | Example |
|---|---|---|
| `Error` | Base class, general errors | `new Error("something failed")` |
| `TypeError` | Wrong type used | `null.property`, calling non-function |
| `RangeError` | Value out of valid range | `new Array(-1)`, `(1).toFixed(200)` |
| `ReferenceError` | Accessing undefined variable | `console.log(undeclared)` |
| `SyntaxError` | Invalid syntax (parse time) | `eval("let {")` |
| `URIError` | Invalid URI encoding | `decodeURIComponent("%")` |
| `EvalError` | Rare; related to `eval()` | Rarely thrown in modern JS |

### try/catch/finally

```javascript
try {
  const data = JSON.parse(invalidJson);    // throws SyntaxError
  processData(data);
} catch (err) {
  if (err instanceof SyntaxError) {
    console.error("Invalid JSON:", err.message);
  } else {
    throw err;  // re-throw unexpected errors
  }
} finally {
  cleanup();  // always runs — even if catch throws
}
```

`finally` runs even if `try` or `catch` have a `return` statement. The `finally` return value overrides.

### Custom Error Classes

```javascript
class AppError extends Error {
  constructor(message, code, statusCode = 500) {
    super(message);
    this.name = "AppError";   // override name for better stack traces
    this.code = code;
    this.statusCode = statusCode;
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, AppError);  // V8-specific — cleaner stack
    }
  }
}

class NotFoundError extends AppError {
  constructor(resource) {
    super(`${resource} not found`, "NOT_FOUND", 404);
    this.name = "NotFoundError";
  }
}

throw new NotFoundError("User");
// NotFoundError: User not found
// code: "NOT_FOUND", statusCode: 404
```

### Global Error Handlers

```javascript
// Browser — uncaught synchronous errors:
window.onerror = function(message, source, lineno, colno, error) {
  console.error("Uncaught:", message);
  return true;  // prevents browser default error behavior
};

// Better (supports all error types):
window.addEventListener("error", (event) => {
  console.error(event.error);
});

// Browser — unhandled Promise rejections:
window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled rejection:", event.reason);
  event.preventDefault();  // suppress console warning
});

// Node.js:
process.on("uncaughtException", (err) => {
  console.error("Uncaught:", err);
  process.exit(1);  // mandatory — process state is undefined after uncaughtException
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled rejection at:", promise, "reason:", reason);
});
```

---

## 15. Memory & Performance

### V8 Memory Spaces

| Space | Contents | GC strategy |
|---|---|---|
| New space (Young gen) | Newly allocated objects; most objects die here | Scavenge (minor GC) — fast, frequent |
| Old space (Old gen) | Objects that survived 2 Scavenges | Mark-Sweep-Compact (major GC) — slower |
| Large object space | Objects > 512KB | Separate tracking; never moved |
| Code space | Compiled JIT bytecode and machine code | Managed separately |
| Map space | Hidden class (Map) objects | Managed separately |

### Garbage Collection

**Scavenge (Minor GC):**
- Runs on new space only
- Uses Cheney's algorithm: copies live objects to a "to" space, discards old "from" space
- Very fast — milliseconds
- Objects surviving 2 scavenges promoted to old space

**Mark-Sweep-Compact (Major GC):**
- Marks all reachable objects from roots (global, stack)
- Sweeps unmarked objects (reclaim memory)
- Compacts old space to eliminate fragmentation
- Can cause "stop-the-world" pauses (V8 uses incremental + concurrent marking to reduce these)

### Hidden Classes

V8 assigns an internal "hidden class" (also called Shape or Map) to objects. Objects with the same properties added in the same order share a hidden class, enabling fast O(1) property access.

```javascript
// Good: consistent property order → single hidden class, optimized
function Point(x, y) {
  this.x = x;
  this.y = y;
}
const p1 = new Point(1, 2);
const p2 = new Point(3, 4);
// p1 and p2 share the same hidden class → IC can optimize

// Bad: different property order → different hidden classes
const a = {};
a.x = 1; a.y = 2;  // hidden class A → B → C

const b = {};
b.y = 1; b.x = 2;  // hidden class A → D → E (different!)
// Code that handles both a and b degrades to polymorphic IC
```

**Rule:** Define all properties in constructor, in consistent order. Avoid adding properties later.

### Inline Caches (IC)

V8 caches the resolution of property accesses at each call site.

| IC state | Trigger | Performance |
|---|---|---|
| Monomorphic | Same hidden class seen at call site | Fast — direct offset |
| Polymorphic | 2–4 different hidden classes seen | Moderate — small lookup table |
| Megamorphic | >4 hidden classes seen | Slow — generic hash lookup |

```javascript
// Monomorphic (fastest):
function getX(obj) { return obj.x; }
getX({ x: 1 });  // always same shape → monomorphic

// Megamorphic (slowest):
function getValue(obj) { return obj.value; }
getValue({ value: 1 });
getValue({ value: 2, extra: true });  // different shape
getValue({ x: 1, value: 3 });        // different shape
// IC goes megamorphic after ~4 shapes
```

### Memory Leaks

| Leak source | Example | Fix |
|---|---|---|
| Global variables | `function f() { leak = "data"; }` (missing `var/let/const`) | Use strict mode; always declare variables |
| Forgotten timers | `setInterval(fn, 100)` never cleared | `clearInterval` when done |
| Closures over large objects | Event listener capturing large DOM context | Remove listener, nullify references |
| Detached DOM nodes | `const el = div; div.remove();` but `el` still referenced | Set to `null` when done |
| Unbounded caches | `cache[key] = data` forever | Use `WeakMap` or LRU cache with max size |

### WeakMap, WeakSet, WeakRef, FinalizationRegistry

```javascript
// WeakMap: keys must be objects; GC can collect key+value if key has no other references
const cache = new WeakMap();
function process(obj) {
  if (cache.has(obj)) return cache.get(obj);
  const result = expensiveCompute(obj);
  cache.set(obj, result);
  return result;
}
// When obj is collected, cache entry is automatically removed — no memory leak

// WeakSet: set of objects; object removed from set when GC collects it
const seen = new WeakSet();
function processOnce(obj) {
  if (seen.has(obj)) return;
  seen.add(obj);
  doWork(obj);
}

// WeakRef: hold a weak reference — GC can still collect the target
const ref = new WeakRef(largeObject);
const obj = ref.deref();  // undefined if GC collected it
if (obj) { obj.doWork(); }

// FinalizationRegistry: callback when object is collected
const registry = new FinalizationRegistry((heldValue) => {
  console.log(`${heldValue} was collected`);
});
let target = { data: "big" };
registry.register(target, "myObject");
target = null;  // target eligible for GC; callback fires eventually
```

**WeakMap vs Map for caching:** WeakMap prevents memory leaks when keys are DOM elements or request objects that get discarded.

---

## 16. Browser APIs

### fetch + AbortController

```javascript
async function fetchWithTimeout(url, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      signal: controller.signal,
      method: "GET",
      headers: { "Content-Type": "application/json" }
    });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Request timed out");
    throw err;
  }
}
```

### Storage APIs

| API | Scope | Persistence | Size limit | Sync/Async | Use when |
|---|---|---|---|---|---|
| `localStorage` | Origin | Permanent (until cleared) | ~5–10MB | Sync | Small key-value: theme, preferences |
| `sessionStorage` | Origin + tab | Tab closed | ~5–10MB | Sync | Session-scoped data: form state |
| `IndexedDB` | Origin | Permanent | Hundreds of MB | Async | Large structured data: offline app data, blobs |

```javascript
// localStorage:
localStorage.setItem("theme", "dark");
const theme = localStorage.getItem("theme");  // "dark"
localStorage.removeItem("theme");
localStorage.clear();  // wipe all

// sessionStorage: same API, tab-scoped
sessionStorage.setItem("draft", JSON.stringify(formData));

// IndexedDB (use idb library for cleaner API):
const db = await openDB("myapp", 1, {
  upgrade(db) {
    db.createObjectStore("users", { keyPath: "id" });
  }
});
await db.put("users", { id: 1, name: "Alice" });
const user = await db.get("users", 1);
```

### Web Workers

Run CPU-intensive code on a background thread without blocking the main thread.

```javascript
// worker.js:
self.onmessage = function(event) {
  const { data } = event;
  const result = heavyComputation(data);
  self.postMessage(result);
};

// main.js:
const worker = new Worker("worker.js");
worker.postMessage(inputData);
worker.onmessage = (event) => {
  console.log("Result:", event.data);
};
worker.onerror = (err) => console.error(err);
worker.terminate();  // clean up when done
```

Workers have no access to DOM. Communication is via `postMessage` (structured clone — deep copy).

### IntersectionObserver

```javascript
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // element entered viewport
        const img = entry.target;
        img.src = img.dataset.src;  // lazy load
        observer.unobserve(img);    // stop observing once loaded
      }
    });
  },
  { rootMargin: "100px", threshold: 0.1 }  // 100px before viewport, 10% visible
);

document.querySelectorAll("img[data-src]").forEach(img => observer.observe(img));
```

### MutationObserver

```javascript
const observer = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    if (mutation.type === "childList") {
      console.log("Children changed:", mutation.addedNodes, mutation.removedNodes);
    }
    if (mutation.type === "attributes") {
      console.log("Attribute changed:", mutation.attributeName);
    }
  });
});

observer.observe(document.getElementById("app"), {
  childList: true,   // watch for added/removed children
  attributes: true,  // watch attribute changes
  subtree: true      // watch all descendants
});

observer.disconnect();  // stop observing
```

### Service Worker Lifecycle and Caching Strategies

```javascript
// Register:
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js");
}

// sw.js lifecycle:
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("v1").then(cache => cache.addAll(["/", "/app.js", "/style.css"]))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== "v1").map(k => caches.delete(k)))
    )
  );
});

// Caching strategies:
self.addEventListener("fetch", (event) => {
  event.respondWith(
    // Cache First: serve from cache, fall back to network
    caches.match(event.request).then(cached => cached || fetch(event.request))
    
    // Network First: try network, fall back to cache
    // fetch(event.request).catch(() => caches.match(event.request))
    
    // Stale While Revalidate: serve cache immediately, update cache in background
    // caches.open("v1").then(cache => {
    //   return cache.match(event.request).then(cached => {
    //     const fetched = fetch(event.request).then(res => { cache.put(event.request, res.clone()); return res; });
    //     return cached || fetched;
    //   });
    // })
  );
});
```

---

## 17. Modern JavaScript Features (ES6–ES2024)

| Feature | Version | Description | Example |
|---|---|---|---|
| Arrow functions | ES6 | Concise syntax, lexical `this` | `const f = x => x * 2` |
| Template literals | ES6 | String interpolation + multiline | `` `Hello ${name}` `` |
| Destructuring | ES6 | Unpack arrays/objects | `const {a, b} = obj` |
| Default params | ES6 | `function f(x = 0)` | `f()` → x is 0 |
| Rest/Spread | ES6 | `...args`, `...arr` | `[...a, ...b]` |
| `let`/`const` | ES6 | Block-scoped variables | `let x = 1` |
| Classes | ES6 | Syntax sugar over prototypes | `class Dog extends Animal` |
| Modules (`import/export`) | ES6 | Static module system | `import { fn } from './mod'` |
| Promises | ES6 | Async primitive | `.then().catch()` |
| `Symbol` | ES6 | Unique, non-string keys | `Symbol("id")` |
| `Map`/`Set` | ES6 | Keyed collections | `new Map()`, `new Set()` |
| `WeakMap`/`WeakSet` | ES6 | Weak-keyed collections | `new WeakMap()` |
| `for...of` | ES6 | Iterate iterables | `for (const x of arr)` |
| Generator functions | ES6 | `function*`, `yield` | `function* gen() { yield 1 }` |
| Proxy/Reflect | ES6 | Meta-programming | `new Proxy(obj, handler)` |
| `Object.assign` | ES6 | Shallow merge | `Object.assign({}, a, b)` |
| Exponentiation `**` | ES2016 | Power operator | `2 ** 10` → 1024 |
| `Array.includes` | ES2016 | Membership test (handles NaN) | `[NaN].includes(NaN)` → true |
| `async`/`await` | ES2017 | Syntactic sugar for Promises | `const x = await fetch(url)` |
| `Object.entries/values` | ES2017 | Key-value iteration | `Object.entries(obj)` |
| `String.padStart/End` | ES2017 | Pad strings | `"5".padStart(3, "0")` → "005" |
| Rest in objects | ES2018 | `const {a, ...rest} = obj` | `rest` has remaining props |
| `Promise.allSettled` | ES2020 | Wait for all, get all results | Never short-circuits |
| Optional chaining `?.` | ES2020 | Safe property access | `obj?.a?.b` |
| Nullish coalescing `??` | ES2020 | Null-safe default | `x ?? "default"` |
| `BigInt` | ES2020 | Arbitrary precision integers | `9007199254740993n` |
| `globalThis` | ES2020 | Universal global object | Works in browser, Node, worker |
| `Promise.any` | ES2021 | First fulfillment | `Promise.any([p1, p2])` |
| Logical assignment `&&=` `\|\|=` `??=` | ES2021 | Conditional assignment | `a ??= default` |
| Numeric separators | ES2021 | Readability | `1_000_000` |
| Class static blocks | ES2022 | Static initializer | `static { this.x = 1; }` |
| Private fields `#` | ES2022 | True encapsulation | `this.#secret` |
| `Object.hasOwn` | ES2022 | Safe hasOwnProperty | `Object.hasOwn(obj, "key")` |
| Top-level `await` | ES2022 | `await` in modules | `const x = await fetch(...)` |
| `Error.cause` | ES2022 | Chain errors | `new Error("msg", { cause: err })` |
| Array `at(-1)` | ES2022 | Negative indexing | `arr.at(-1)` → last element |
| `findLast/findLastIndex` | ES2023 | Search from end | `arr.findLast(x => x > 0)` |
| Array `toSorted/toReversed/toSpliced/with` | ES2023 | Non-mutating variants | `arr.toSorted()` (original unchanged) |
| `Object.groupBy` | ES2024 | Group array into object | `Object.groupBy(arr, fn)` |
| `Map.groupBy` | ES2024 | Group array into Map | `Map.groupBy(arr, fn)` |
| `Promise.withResolvers` | ES2024 | Expose resolve/reject | `const {promise, resolve, reject} = Promise.withResolvers()` |

---

## 18. Interview Q&A — 20 Questions

**Q1: What is the difference between `==` and `===`?**

`===` (strict equality) compares value and type — no coercion. `==` applies the Abstract Equality Comparison algorithm which coerces types before comparing.

```javascript
0 == false   // true  (false coerced to 0)
0 === false  // false (number vs boolean — different types)
null == undefined  // true  (spec exception)
null === undefined // false
```

Always prefer `===`. The only valid use of `==` is `x == null` as a shorthand for `x === null || x === undefined`.

**Follow-up:** What does `NaN == NaN` return? → `false`. `NaN` is not equal to anything, including itself. Use `Number.isNaN()`.

---

**Q2: Explain event loop, microtasks, and macrotasks.**

The event loop monitors the call stack and queues. When the stack is empty: (1) drain all microtasks (Promise `.then`, `queueMicrotask`), (2) run one macrotask (`setTimeout`, I/O), (3) drain microtasks again, (4) repeat.

```javascript
setTimeout(() => console.log("macro"), 0);
Promise.resolve().then(() => console.log("micro"));
console.log("sync");
// Output: sync → micro → macro
```

**Follow-up:** Can microtasks starve macrotasks? → Yes. If microtask callbacks keep adding more microtasks, macrotasks never run.

---

**Q3: What is a closure and why is it useful?**

A closure is a function that retains access to its enclosing scope's variables after the outer function returns.

```javascript
function makeAdder(x) {
  return function(y) { return x + y; };  // closes over x
}
const add5 = makeAdder(5);
add5(3);  // 8 — x=5 still accessible
```

Uses: data privacy, function factories, memoization, partial application, module pattern.

**Follow-up:** What is the closure loop bug? → Using `var` in a loop creates one shared binding; all closures see the final value. Fix with `let` (per-iteration binding) or IIFE.

---

**Q4: How does `this` work in JavaScript?**

`this` is determined by the call site, not the definition site. Four rules:
1. Default: standalone call → global or `undefined` (strict mode)
2. Implicit: `obj.method()` → `this = obj`
3. Explicit: `.call(ctx)`, `.apply(ctx)`, `.bind(ctx)` → `this = ctx`
4. `new`: constructor call → `this = new object`

Arrow functions have no own `this` — they inherit lexically from enclosing scope.

**Follow-up:** How do you fix `this` loss in a callback? → Use `.bind(this)`, an arrow function, or store `const self = this`.

---

**Q5: What is prototypal inheritance?**

Every object has `[[Prototype]]` pointing to another object. Property lookup walks the chain until found or `null`.

```javascript
const animal = { breathes: true };
const dog = Object.create(animal);
dog.bark = () => "woof";
dog.breathes;  // true — found on animal via chain
```

ES6 `class` is syntax sugar — `extends` sets up the prototype chain; `super` calls the parent constructor/method.

**Follow-up:** Difference between `Object.create(null)` and `{}`? → `Object.create(null)` has no prototype at all — no `toString`, `hasOwnProperty`, etc. Useful for pure dictionary objects.

---

**Q6: What is the difference between `var`, `let`, and `const`?**

`var` is function-scoped, hoisted (initialized to `undefined`), can be redeclared, and attaches to `window`. `let` and `const` are block-scoped, in TDZ before declaration, cannot be redeclared. `const` additionally prevents reassignment of the binding (not deep immutability — object properties can still change).

```javascript
const obj = { a: 1 };
obj.a = 2;     // OK — property changed
obj = {};      // TypeError — binding reassigned
```

**Follow-up:** What is TDZ? → Temporal Dead Zone: the period from block start to the `let`/`const` declaration where accessing the variable throws `ReferenceError`.

---

**Q7: Explain `Promise.all` vs `Promise.allSettled` vs `Promise.race` vs `Promise.any`.**

| Method | Resolves | Rejects | Use case |
|---|---|---|---|
| `all` | All fulfilled | Any rejects | Parallel fetches that must ALL succeed |
| `allSettled` | Always (with status objects) | Never | When you need all results regardless of failures |
| `race` | First to settle | First to settle (if rejection) | Timeout pattern |
| `any` | First fulfillment | All rejected (AggregateError) | First successful response from multiple sources |

**Follow-up:** How do you implement a timeout for `Promise.all`? → `Promise.race([Promise.all(promises), timeoutPromise])`.

---

**Q8: What are generators and when would you use them?**

Generator functions (`function*`) return an iterator. They pause at each `yield`, resuming when `.next()` is called. The function's stack frame is preserved between calls.

```javascript
function* range(start, end) {
  for (let i = start; i < end; i++) yield i;
}
[...range(0, 5)];  // [0, 1, 2, 3, 4]

// Infinite sequence:
function* naturals() {
  let n = 0;
  while (true) yield n++;
}
const gen = naturals();
gen.next().value;  // 0
gen.next().value;  // 1
```

Uses: lazy sequences, infinite streams, custom iterables, cooperative multitasking.

**Follow-up:** How does `async/await` relate to generators? → `async/await` is desugared to generator-based state machines by transpilers. Conceptually, `await` is like `yield` where the runtime handles resumption.

---

**Q9: What are WeakMap and WeakRef? How do they help with memory?**

`WeakMap` holds keys weakly — if the key object has no other references, GC can collect it and the entry is removed automatically. Useful for associating data with DOM nodes or request objects without preventing their collection.

`WeakRef` holds a weak reference to an object — the GC can collect the target. Call `.deref()` to get the object (or `undefined` if collected).

```javascript
const cache = new WeakMap();
function enhance(el) {
  if (cache.has(el)) return cache.get(el);
  const result = computeExpensive(el);
  cache.set(el, result);  // auto-cleared when el is GC'd
  return result;
}
```

**Follow-up:** Can you iterate a WeakMap? → No. Iteration is impossible by design — keys could be collected mid-iteration.

---

**Q10: How does V8 optimize JavaScript?**

V8 compiles source → bytecode (Ignition interpreter). Hot functions (called frequently) are JIT-compiled to optimized machine code by TurboFan. Optimizations include: inline caching (cache property offsets by hidden class), hidden class-based property access, inlining function calls, escape analysis.

V8 can deoptimize (bail out to bytecode) when assumptions break: e.g., passing a different type to an optimized function.

**Follow-up:** What can you do to help V8 optimize your code? → Initialize all properties in constructors in consistent order, avoid changing property types, avoid `arguments` object, use typed arrays for numerical data.

---

**Q11: What is the difference between `call`, `apply`, and `bind`?**

All three explicitly set `this`. Difference is when the function executes and how arguments are passed.

```javascript
function greet(greeting, punct) { return `${greeting}, ${this.name}${punct}`; }
const obj = { name: "Alice" };

greet.call(obj, "Hello", "!");       // executes immediately, args spread
greet.apply(obj, ["Hello", "!"]);    // executes immediately, args as array
const fn = greet.bind(obj, "Hello"); // returns new function, this+first arg bound
fn("!");                              // "Hello, Alice!"
```

**Follow-up:** What does `bind` return? → A new function with `this` and optionally some arguments pre-set (partial application).

---

**Q12: Explain the difference between shallow and deep cloning.**

Shallow clone copies top-level properties; nested objects are still shared by reference.

```javascript
const a = { x: 1, nested: { y: 2 } };
const shallow = { ...a };
shallow.x = 99;       // original unchanged
shallow.nested.y = 99;// MODIFIES original — shared reference

// Deep clone options:
const deep1 = JSON.parse(JSON.stringify(a));  // simple but loses functions, Dates become strings, fails on circular refs
const deep2 = structuredClone(a);             // native, handles circular refs, Dates, Maps, Sets (ES2022)
```

**Follow-up:** When does `structuredClone` fail? → Functions, DOM nodes, class instances (loses methods — only data is cloned).

---

**Q13: What is debouncing and throttling?**

Debounce: delay execution until after N ms of inactivity. Useful for search input, resize events.
Throttle: execute at most once per N ms. Useful for scroll events, rate limiting.

```javascript
// Debounce: only fires after user stops typing for 300ms
input.addEventListener("input", debounce(search, 300));

// Throttle: fires at most once per 300ms during scroll
window.addEventListener("scroll", throttle(updatePosition, 300));
```

**Follow-up:** What's the difference in feel? → Debounce: delay then fire once. Throttle: fire immediately, then no more for N ms.

---

**Q14: What is `Symbol` and why use it?**

`Symbol` creates unique, non-string property keys. Two Symbols created with the same description are never equal.

```javascript
const id = Symbol("id");
const id2 = Symbol("id");
id === id2;  // false — always unique

const user = { [id]: 123, name: "Alice" };
user[id];   // 123
// Symbol-keyed properties don't appear in for...in, Object.keys, JSON.stringify
// Only Object.getOwnPropertySymbols() or Reflect.ownKeys() reveals them

// Well-known symbols override built-in behavior:
class Range {
  constructor(start, end) { this.start = start; this.end = end; }
  [Symbol.iterator]() {
    let current = this.start;
    const end = this.end;
    return { next() { return current <= end ? { value: current++, done: false } : { done: true }; } };
  }
}
[...new Range(1, 5)];  // [1, 2, 3, 4, 5]
```

**Follow-up:** What are well-known Symbols? → `Symbol.iterator`, `Symbol.toPrimitive`, `Symbol.hasInstance`, `Symbol.asyncIterator`, etc. — override built-in JS behavior.

---

**Q15: How does `async/await` error handling differ from Promise chains?**

```javascript
// Promise chain: catch at the end
fetch(url)
  .then(res => res.json())
  .then(processData)
  .catch(err => console.error(err));  // catches any error in chain

// async/await: try/catch like synchronous code
async function load() {
  try {
    const res = await fetch(url);
    const data = await res.json();
    return processData(data);
  } catch (err) {
    console.error(err);  // catches any await rejection or thrown error
  }
}
```

With async/await, each `await` can be individually wrapped in try/catch for granular handling.

**Follow-up:** What happens if you don't catch a rejected async function? → Returns a rejected Promise; if unhandled, triggers `unhandledrejection`.

---

**Q16: What is the module pattern and why use ES modules instead?**

Module pattern (pre-ESM) uses closures to create private state:

```javascript
const Counter = (function() {
  let count = 0;
  return { increment() { count++; }, get() { return count; } };
})();
```

ES modules are better: static analysis enables tree-shaking (bundlers can eliminate unused exports), imports are read-only live bindings (not copies), circular dependencies work correctly, async loading is supported, and tooling can do type checking and code splitting.

**Follow-up:** What does "live binding" mean? → When a module exports `let count`, importing modules see the current value of `count` if the exporter changes it — unlike CJS which exports a copy.

---

**Q17: What are Proxy and Reflect?**

`Proxy` intercepts operations on an object (get, set, delete, function calls, etc.) via "traps".

```javascript
const handler = {
  get(target, prop) {
    console.log(`Getting ${prop}`);
    return prop in target ? target[prop] : `Property ${prop} not found`;
  },
  set(target, prop, value) {
    if (typeof value !== "number") throw new TypeError("Numbers only");
    target[prop] = value;
    return true;  // must return true for success
  }
};

const obj = new Proxy({}, handler);
obj.x = 42;       // set trap fires
obj.x;            // get trap fires → 42
obj.y;            // get trap fires → "Property y not found"
```

`Reflect` provides methods that mirror Proxy traps — use inside traps to preserve default behavior: `Reflect.get(target, prop)`.

**Follow-up:** What is Proxy used for in practice? → Vue 3 reactivity system, validation layers, API mocking, data binding, observability.

---

**Q18: What is the difference between `Object.freeze` and `const`?**

`const` prevents reassignment of the binding (variable), not mutation of the value. `Object.freeze` prevents mutation of the object's properties.

```javascript
const arr = [1, 2, 3];
arr.push(4);     // OK — const doesn't prevent mutation
arr = [1, 2];    // TypeError — const prevents reassignment

const frozen = Object.freeze({ x: 1, nested: { y: 2 } });
frozen.x = 99;       // silently fails (TypeError in strict mode)
frozen.nested.y = 99;// WORKS — freeze is shallow!
```

**Follow-up:** How do you deep freeze? → Recursively freeze all nested objects.

---

**Q19: Explain IIFE and why it was used.**

IIFE (Immediately Invoked Function Expression) creates a private scope and executes immediately. Used pre-ES6 to avoid polluting global scope and to create module-like encapsulation.

```javascript
(function() {
  var privateVar = "hidden";
  window.myLib = { /* public API */ };
})();
// privateVar unreachable from outside
```

Mostly replaced by ES modules and block-scoped `let`/`const`, but still useful for: inline async code (`(async () => { await ... })()`), creating block scopes with `var`, one-time initialization.

**Follow-up:** Why the wrapping parentheses? → Without `()`, `function` at statement start is parsed as a declaration (not expression), and declarations can't be immediately invoked. The outer `()` turns it into an expression.

---

**Q20: What is tail call optimization (TCO)?**

A tail call is a function call as the last action in a function. TCO allows engines to reuse the current stack frame instead of creating a new one, enabling recursive algorithms without stack overflow.

```javascript
// Regular recursion — O(n) stack frames:
function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);  // NOT tail call — must multiply after return
}

// Tail-recursive — can be TCO'd:
function factorial(n, acc = 1) {
  if (n <= 1) return acc;
  return factorial(n - 1, n * acc);  // tail call — no work after recursive call
}
```

ES6 specifies TCO but only Safari/JSC implements it. V8 removed TCO support due to tooling concerns. In practice, use iteration for performance-critical recursion in Node.js.

**Follow-up:** How do you get tail-recursive performance in V8? → Use trampolining: return a function instead of calling it, and drive the loop externally.

---

## 19. Solved Practice Problems

### Problem 1: Implement `debounce`

**Problem:** Create a function that delays invoking `fn` until after `delay` ms have elapsed since the last call.

**Approach:** Store a timeout ID. On each call, clear the previous timeout and set a new one.

```javascript
function debounce(fn, delay) {
  let timeoutId = null;

  function debounced(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      fn.apply(this, args);
      timeoutId = null;
    }, delay);
  }

  debounced.cancel = function() {
    clearTimeout(timeoutId);
    timeoutId = null;
  };

  debounced.flush = function() {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      fn.apply(this, arguments);
      timeoutId = null;
    }
  };

  return debounced;
}

// Usage:
const search = debounce((query) => fetch(`/search?q=${query}`), 300);
input.addEventListener("input", (e) => search(e.target.value));
```

**Complexity:** O(1) time, O(1) space (one timeout ID).

---

### Problem 2: Implement `throttle`

**Problem:** Create a function that invokes `fn` at most once per `limit` ms.

**Approach:** Track last invocation time. If enough time has passed, invoke immediately. Otherwise, schedule for when the window ends.

```javascript
function throttle(fn, limit) {
  let lastRan = 0;
  let timeoutId = null;

  return function(...args) {
    const now = Date.now();
    const remaining = limit - (now - lastRan);

    if (remaining <= 0) {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      lastRan = now;
      fn.apply(this, args);
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        lastRan = Date.now();
        timeoutId = null;
        fn.apply(this, args);
      }, remaining);
    }
  };
}

// Usage:
const onScroll = throttle(() => updateScrollIndicator(), 100);
window.addEventListener("scroll", onScroll);
```

**Complexity:** O(1) time, O(1) space.

---

### Problem 3: Implement `Promise.all`

**Problem:** Implement `Promise.all(promises)` — resolves with array of values when all resolve; rejects with first rejection.

**Approach:** Track resolved count. Reject immediately on any failure. Resolve when all complete.

```javascript
function promiseAll(promises) {
  return new Promise((resolve, reject) => {
    if (!Array.isArray(promises)) return reject(new TypeError("Argument must be an array"));
    if (promises.length === 0) return resolve([]);

    const results = new Array(promises.length);
    let remaining = promises.length;

    promises.forEach((promise, index) => {
      Promise.resolve(promise).then(
        (value) => {
          results[index] = value;
          remaining--;
          if (remaining === 0) resolve(results);
        },
        (reason) => reject(reason)
      );
    });
  });
}

// Test:
promiseAll([Promise.resolve(1), Promise.resolve(2), Promise.resolve(3)])
  .then(console.log);  // [1, 2, 3]

promiseAll([Promise.resolve(1), Promise.reject("error")])
  .catch(console.error);  // "error"
```

**Complexity:** O(n) time and space.

---

### Problem 4: Deep Clone an Object

**Problem:** Deep clone handling nested objects, arrays, Date, circular references.

**Approach:** Recursive traversal with a `WeakMap` to track visited objects and handle cycles.

```javascript
function deepClone(value, seen = new WeakMap()) {
  // Primitives: return as-is
  if (value === null || typeof value !== "object") return value;

  // Handle circular references
  if (seen.has(value)) return seen.get(value);

  // Date
  if (value instanceof Date) return new Date(value.getTime());

  // RegExp
  if (value instanceof RegExp) return new RegExp(value.source, value.flags);

  // Array
  if (Array.isArray(value)) {
    const clone = [];
    seen.set(value, clone);
    for (let i = 0; i < value.length; i++) {
      clone[i] = deepClone(value[i], seen);
    }
    return clone;
  }

  // Map
  if (value instanceof Map) {
    const clone = new Map();
    seen.set(value, clone);
    for (const [k, v] of value) {
      clone.set(deepClone(k, seen), deepClone(v, seen));
    }
    return clone;
  }

  // Set
  if (value instanceof Set) {
    const clone = new Set();
    seen.set(value, clone);
    for (const v of value) clone.add(deepClone(v, seen));
    return clone;
  }

  // Plain object
  const clone = Object.create(Object.getPrototypeOf(value));
  seen.set(value, clone);
  for (const key of Reflect.ownKeys(value)) {  // includes Symbols
    clone[key] = deepClone(value[key], seen);
  }
  return clone;
}

// Test circular:
const a = { x: 1 };
a.self = a;
const b = deepClone(a);
b.self === b;  // true — circular ref preserved correctly
```

**Complexity:** O(n) time and space where n is total number of nodes.

---

### Problem 5: Flatten Nested Array (No `Array.flat`)

**Problem:** Flatten array to arbitrary depth without using `Array.prototype.flat`.

**Approach:** Recursive reduction. For each element, if array → recurse; else push.

```javascript
function flatten(arr, depth = Infinity) {
  if (depth === 0) return arr.slice();

  return arr.reduce((acc, item) => {
    if (Array.isArray(item) && depth > 0) {
      acc.push(...flatten(item, depth - 1));
    } else {
      acc.push(item);
    }
    return acc;
  }, []);
}

// Iterative version (avoids stack overflow for deeply nested):
function flattenIterative(arr) {
  const stack = [...arr];
  const result = [];
  while (stack.length) {
    const item = stack.pop();  // pop from end → reverse order
    if (Array.isArray(item)) {
      stack.push(...item);     // expand and push back
    } else {
      result.push(item);
    }
  }
  return result.reverse();    // correct order
}

flatten([1, [2, [3, [4]]]]);            // [1, 2, 3, 4]
flatten([1, [2, [3, [4]]]], 2);         // [1, 2, 3, [4]]
flattenIterative([1, [2, [3, [4]]]]);   // [1, 2, 3, 4]
```

**Complexity:** O(n) time and space.

---

### Problem 6: Implement `EventEmitter`

**Problem:** Build EventEmitter with `on(event, listener)`, `off(event, listener)`, `emit(event, ...args)`.

**Approach:** Map from event name to Set of listeners.

```javascript
class EventEmitter {
  #events = new Map();

  on(event, listener) {
    if (!this.#events.has(event)) {
      this.#events.set(event, new Set());
    }
    this.#events.get(event).add(listener);
    return this;  // chainable
  }

  off(event, listener) {
    if (this.#events.has(event)) {
      this.#events.get(event).delete(listener);
      if (this.#events.get(event).size === 0) {
        this.#events.delete(event);
      }
    }
    return this;
  }

  emit(event, ...args) {
    if (!this.#events.has(event)) return false;
    this.#events.get(event).forEach(listener => listener(...args));
    return true;
  }

  once(event, listener) {
    const wrapper = (...args) => {
      listener(...args);
      this.off(event, wrapper);
    };
    return this.on(event, wrapper);
  }

  listenerCount(event) {
    return this.#events.has(event) ? this.#events.get(event).size : 0;
  }

  removeAllListeners(event) {
    if (event) this.#events.delete(event);
    else this.#events.clear();
    return this;
  }
}

// Test:
const emitter = new EventEmitter();
const handler = (msg) => console.log("Received:", msg);
emitter.on("data", handler);
emitter.emit("data", "hello");  // "Received: hello"
emitter.off("data", handler);
emitter.emit("data", "world");  // nothing — handler removed
```

**Complexity:** `on`/`off` O(1) (Set operations), `emit` O(n) where n = listener count.

---

### Problem 7: Curry a Function

**Problem:** Implement `curry` such that `curry(add)(1)(2)(3) === 6` and `curry(add)(1, 2)(3) === 6`.

**Approach:** Compare accumulated argument count with function arity. Invoke when enough args collected.

```javascript
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    }
    return function(...moreArgs) {
      return curried.apply(this, args.concat(moreArgs));
    };
  };
}

// Test:
const add = (a, b, c) => a + b + c;
const curriedAdd = curry(add);

curriedAdd(1)(2)(3);   // 6
curriedAdd(1, 2)(3);   // 6
curriedAdd(1)(2, 3);   // 6
curriedAdd(1, 2, 3);   // 6

// Reusable partial:
const add10 = curriedAdd(10);
add10(5)(2);  // 17
```

**Complexity:** O(1) per call, O(n) space for argument accumulation across calls.

---

### Problem 8: Implement Memoize with Generic Cache

**Problem:** Implement `memoize(fn)` that caches results by arguments. Handle multiple arguments and edge cases.

**Approach:** Use a nested Map trie for multi-argument functions — avoids key serialization pitfalls.

```javascript
function memoize(fn) {
  const cache = new Map();

  return function(...args) {
    // Traverse/build a trie of Maps for multi-arg cache
    let node = cache;
    for (const arg of args) {
      if (!node.has(arg)) node.set(arg, new Map());
      node = node.get(arg);
    }

    const RESULT = Symbol("result");
    if (node.has(RESULT)) return node.get(RESULT);

    const result = fn.apply(this, args);
    node.set(RESULT, result);
    return result;
  };
}

// Alternative: simple JSON key (fails for objects, functions, circular refs):
function memoizeSimple(fn) {
  const cache = new Map();
  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}

// Test:
let callCount = 0;
const expensiveFn = memoize((a, b) => { callCount++; return a + b; });
expensiveFn(1, 2);  // 3, callCount=1
expensiveFn(1, 2);  // 3, callCount=1 (cached)
expensiveFn(2, 3);  // 5, callCount=2
```

**Complexity:** O(k) per call where k = number of arguments (trie depth). O(n·k) space total.

---

### Problem 9: Detect Cycle in Linked List (Floyd's Algorithm)

**Problem:** Given a linked list implemented as JS objects (`{ val, next }`), detect if it has a cycle.

**Approach:** Floyd's tortoise and hare algorithm — two pointers moving at different speeds. If they meet, cycle exists.

```javascript
function hasCycle(head) {
  if (!head || !head.next) return false;

  let slow = head;       // moves 1 step
  let fast = head.next;  // moves 2 steps

  while (slow !== fast) {
    if (!fast || !fast.next) return false;  // reached end — no cycle
    slow = slow.next;
    fast = fast.next.next;
  }

  return true;  // slow === fast — cycle detected
}

// Find start of cycle:
function detectCycleStart(head) {
  let slow = head;
  let fast = head;
  let hasCycle = false;

  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) { hasCycle = true; break; }
  }

  if (!hasCycle) return null;

  slow = head;  // reset slow to head
  while (slow !== fast) {
    slow = slow.next;
    fast = fast.next;
  }
  return slow;  // start of cycle
}

// Build test list with cycle:
const a = { val: 1, next: null };
const b = { val: 2, next: null };
const c = { val: 3, next: null };
const d = { val: 4, next: null };
a.next = b; b.next = c; c.next = d; d.next = b;  // cycle: d → b

hasCycle(a);           // true
detectCycleStart(a);   // b (node with val: 2)
```

**Complexity:** O(n) time, O(1) space — no extra data structures needed.

---

### Problem 10: Build a Simple Observable

**Problem:** Implement an Observable with `subscribe(observer)` and `unsubscribe()`, where observers have `next`, `error`, `complete` callbacks.

**Approach:** Constructor takes a subscriber function. `subscribe` invokes it with a wrapped observer that handles teardown.

```javascript
class Observable {
  #subscribeFn;

  constructor(subscribeFn) {
    this.#subscribeFn = subscribeFn;
  }

  subscribe(observerOrNext, onError, onComplete) {
    const observer = typeof observerOrNext === "function"
      ? { next: observerOrNext, error: onError, complete: onComplete }
      : observerOrNext;

    let active = true;
    let teardown = null;

    const safeObserver = {
      next(value) {
        if (active && observer.next) observer.next(value);
      },
      error(err) {
        if (active) {
          active = false;
          if (observer.error) observer.error(err);
        }
      },
      complete() {
        if (active) {
          active = false;
          if (observer.complete) observer.complete();
        }
      }
    };

    teardown = this.#subscribeFn(safeObserver);

    return {
      unsubscribe() {
        active = false;
        if (teardown && typeof teardown === "function") teardown();
      }
    };
  }

  // Operator: transform values
  map(transformFn) {
    return new Observable(observer => {
      return this.subscribe({
        next: value => observer.next(transformFn(value)),
        error: err => observer.error(err),
        complete: () => observer.complete()
      }).unsubscribe;
    });
  }

  // Operator: filter values
  filter(predicateFn) {
    return new Observable(observer => {
      return this.subscribe({
        next: value => { if (predicateFn(value)) observer.next(value); },
        error: err => observer.error(err),
        complete: () => observer.complete()
      }).unsubscribe;
    });
  }

  static fromArray(arr) {
    return new Observable(observer => {
      arr.forEach(item => observer.next(item));
      observer.complete();
    });
  }

  static interval(ms) {
    return new Observable(observer => {
      let i = 0;
      const id = setInterval(() => observer.next(i++), ms);
      return () => clearInterval(id);  // teardown
    });
  }
}

// Test:
const numbers$ = Observable.fromArray([1, 2, 3, 4, 5]);
const evens$ = numbers$.filter(n => n % 2 === 0).map(n => n * 10);

const sub = evens$.subscribe({
  next: v => console.log(v),     // 20, 40
  complete: () => console.log("done")
});

// Interval with unsubscribe:
const tick$ = Observable.interval(1000);
const sub2 = tick$.subscribe(i => console.log(`tick ${i}`));
setTimeout(() => sub2.unsubscribe(), 3500);  // unsubscribes after ~3 ticks
```

**Complexity:** subscribe O(1), emit O(n) where n = active subscribers.

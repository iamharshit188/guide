# JavaScript — Runtime + Modern Patterns

---

## Quick Reference

| Topic | Key Facts |
|-------|-----------|
| V8 Ignition | Bytecode interpreter; compiles source → bytecode on first run |
| V8 TurboFan | Optimizing JIT compiler; triggered after function is "hot" |
| Hidden classes | V8's internal representation for object shape; property access $O(1)$ |
| Inline caching | Caches resolved property offsets at call sites; monomorphic → polymorphic → megamorphic |
| Event loop | Single-threaded; macrotask → microtask drain → render → next macrotask |
| Microtasks | `Promise.then`, `queueMicrotask`, `MutationObserver`; drain before next macrotask |
| Macrotasks | `setTimeout`, `setInterval`, `setImmediate`, I/O callbacks |
| `typeof null` | `"object"` — historical bug; `null` is not an object |
| `===` vs `==` | `===` no coercion; `==` applies Abstract Equality Comparison rules |
| `var` vs `let/const` | `var`: function-scoped, hoisted, on `window`; `let/const`: block-scoped, TDZ |
| Prototype lookup | Walk `__proto__` chain until `null`; `Object.prototype` is final link |
| `this` binding | Call-site determines `this`; arrow functions capture lexical `this` |
| `WeakMap` | Keys are objects; not enumerable; GC can collect keys |
| CJS vs ESM | `require` sync, dynamic; `import` static, async, tree-shakeable |

---

## Core Concepts

### V8 Engine Internals

**Compilation pipeline:**

```
Source code
    │
    ▼ Parser
AST (Abstract Syntax Tree)
    │
    ▼ Ignition (bytecode compiler)
Bytecode (compact, interpreted)
    │
    ▼ Profiler (counts function invocations)
    │  "hot" function detected (typically ~1000–5000 calls)
    ▼ TurboFan (optimizing JIT)
Machine code (type-specialized, inlined)
    │
    ▼ Deoptimization (type assumption violated)
Back to bytecode
```

**Hidden classes (Shapes/Maps):**

V8 assigns every object an internal "hidden class" (called Map in V8 source) that describes its property layout — property names and their offsets. Objects with the same properties in the same insertion order share a hidden class, enabling $O(1)$ property access via direct memory offset.

```js
const a = {};  a.x = 1;  a.y = 2;   // hidden class: {x:0, y:4}
const b = {};  b.x = 5;  b.y = 6;   // same hidden class — fast path

const c = {};  c.y = 1;  c.x = 2;   // different hidden class {y:0, x:4}
```

Property access compiled to: `obj + offset` — a load from a fixed memory offset, same as C struct member access.

**Adding properties dynamically after construction breaks the hidden class — creates a new hidden class and degrades performance.** Use object literals with all properties, or classes with properties initialized in the constructor.

**Inline caching (IC):**

At each property access site, V8 caches the hidden class → offset mapping. States:
- **Uninitialized** — first execution; not cached
- **Monomorphic** — one hidden class seen; fastest path
- **Polymorphic** — 2–4 hidden classes; checks each
- **Megamorphic** — 5+ hidden classes; generic slow path (cannot inline)

Functions receiving objects of consistent shape stay monomorphic → maximum JIT performance.

**Memory spaces:**

| Space | Contents |
|-------|----------|
| New space | Young generation; objects allocated here first |
| Old space | Survivors of two minor GCs |
| Large object space | Objects > 512 KB |
| Code space | Compiled machine code |
| Map space | Hidden class objects |

Minor GC (Scavenger) is $O(\text{live objects in new space})$ — very fast. Major GC (Mark-Compact) is $O(\text{heap size})$ — causes noticeable pauses; incremental/concurrent marking in V8 reduces pause time.

### Event Loop & Task Queues

```
┌─────────────────────────────────────────────────────┐
│                    Call Stack                        │
│  (executing JS; must be empty before event loop     │
│   picks up next task)                               │
└──────────────────────────┬──────────────────────────┘
                           │ empty
                           ▼
┌─────────────────────────────────────────────────────┐
│              Microtask Queue                         │
│  Promise.then/catch/finally callbacks               │
│  queueMicrotask()                                   │
│  MutationObserver                                   │
│  (drain ALL before next macrotask)                  │
└──────────────────────────┬──────────────────────────┘
                           │ empty
                           ▼
┌─────────────────────────────────────────────────────┐
│              Macrotask Queue                         │
│  setTimeout / setInterval callbacks                 │
│  I/O callbacks (network, file)                      │
│  setImmediate (Node.js)                             │
│  (pick ONE per loop iteration)                      │
└─────────────────────────────────────────────────────┘
```

**Key ordering rule:** After each macrotask completes, the microtask queue is fully drained before the next macrotask begins. This means a chain of resolved promises can run to completion before any timer fires.

```js
console.log('1');
setTimeout(() => console.log('4'), 0);
Promise.resolve()
    .then(() => console.log('2'))
    .then(() => console.log('3'));
// Output: 1, 2, 3, 4
```

`requestAnimationFrame` callbacks run between macrotasks and before the browser renders — distinct from both queues.

### Closure & Lexical Scoping

A closure is a function bundled with its lexical environment — the variable bindings in scope at definition time.

```js
function makeAdder(x) {
    return function(y) { return x + y; };
}
const add5 = makeAdder(5);
add5(3);   // 8 — x=5 captured in closure
```

The closure keeps the entire scope chain alive. If the captured variables hold large objects, they prevent GC — a common memory leak source in event listeners.

**IIFE (Immediately Invoked Function Expression):** creates a scope without polluting the outer scope — obsolete with `let`/`const` but still seen in legacy code.

**`var` hoisting:**
```js
console.log(x);   // undefined (hoisted declaration, not initialization)
var x = 5;

// function declarations are fully hoisted:
foo();   // works
function foo() {}
```

`let`/`const` are hoisted but not initialized — accessing before declaration throws `ReferenceError` (Temporal Dead Zone).

### Prototype Chain & `Object.create`

Every object has an internal `[[Prototype]]` link (accessible as `__proto__` or via `Object.getPrototypeOf`). Property lookup walks the chain until found or `null` is reached.

```
instance.__proto__ === Constructor.prototype
Constructor.prototype.__proto__ === Object.prototype
Object.prototype.__proto__ === null
```

`Object.create(proto)` — creates an object with `proto` as its prototype without calling a constructor:

```js
const animalProto = {
    speak() { return `${this.name} makes a sound`; }
};
const dog = Object.create(animalProto);
dog.name = 'Rex';
dog.speak();   // "Rex makes a sound"
```

**`hasOwnProperty` vs `in`:** `'prop' in obj` checks the full prototype chain; `obj.hasOwnProperty('prop')` checks only own properties. Use `Object.hasOwn(obj, 'prop')` (ES2022) to avoid shadowing issues.

**ES6 class syntax:** Syntactic sugar over prototype-based delegation. `class` does not create a new type system.

```js
class Animal {
    #name;   // private field (ES2022)
    constructor(name) { this.#name = name; }
    speak() { return `${this.#name} speaks`; }
    static create(name) { return new Animal(name); }
}

class Dog extends Animal {
    #breed;
    constructor(name, breed) {
        super(name);
        this.#breed = breed;
    }
    info() { return `${super.speak()}, breed: ${this.#breed}`; }
}
```

`extends` sets up: `Dog.prototype.__proto__ = Animal.prototype` and `Dog.__proto__ = Animal` (for static inheritance).

### Promise Internals

A Promise is a state machine with three states: **pending** → **fulfilled** or **rejected**. Transitions are one-way and irreversible.

Internally, a Promise holds:
- A state field
- A result value (once settled)
- Arrays of fulfillment and rejection reaction callbacks

`.then(onFulfilled, onRejected)` returns a new Promise chained to the reaction. Reactions are always called asynchronously (scheduled as microtasks), even if the promise is already settled when `.then` is called.

**Unhandled rejection:** If a rejected promise has no rejection handler, Node.js emits `unhandledRejection`; browsers fire `unhandledrejection` event. Use `process.on('unhandledRejection', handler)` or `window.addEventListener('unhandledrejection', handler)` as a safety net.

**`Promise.all` vs `Promise.allSettled` vs `Promise.race` vs `Promise.any`:**

| Method | Resolves when | Rejects when |
|--------|--------------|-------------|
| `all(promises)` | All fulfilled | Any rejects (fast-fail) |
| `allSettled(promises)` | All settled (never rejects) | Never |
| `race(promises)` | First settles (any outcome) | First rejects |
| `any(promises)` | First fulfills | All reject (`AggregateError`) |

### async/await Mechanics

`async function` always returns a Promise. `await expr` desugars to: suspend coroutine, attach `.then` continuation to `expr`'s promise, return to caller. When the promise resolves, the continuation is scheduled as a microtask and execution resumes.

```js
async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}
```

Compiles to roughly:
```js
function fetchJson(url) {
    return fetch(url).then(resp => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    });
}
```

**Common pitfall — sequential vs parallel:**
```js
// Sequential: 2 seconds total
const a = await fetchA();   // wait 1s
const b = await fetchB();   // wait 1s

// Parallel: 1 second total
const [a, b] = await Promise.all([fetchA(), fetchB()]);
```

**Top-level await:** Available in ES modules (`type="module"`); not available in CommonJS.

### WeakMap/WeakRef for Memory-Safe Caching

`WeakMap` keys must be objects; the map does not prevent GC of the key object. When the key is garbage-collected, the entry is automatically removed.

```js
const cache = new WeakMap();

function process(obj) {
    if (cache.has(obj)) return cache.get(obj);
    const result = expensiveCompute(obj);
    cache.set(obj, result);
    return result;
}
// When obj is GC'd, the cache entry disappears automatically
```

`WeakRef` wraps an object without preventing GC. Use `.deref()` — returns the object or `undefined` if collected.

```js
let cache = new WeakRef(largeObject);
largeObject = null;   // allow GC

// later:
const obj = cache.deref();
if (obj) { /* still alive */ }
else { /* was collected, rebuild */ }
```

`FinalizationRegistry`: register a callback called after an object is GC'd (not guaranteed or timely). Use for resource cleanup, not program logic.

`WeakSet`: like `WeakMap` but stores objects with no associated value. Use case: tracking DOM nodes without preventing removal.

### Module Systems: CJS vs ESM

**CommonJS (Node.js traditional):**
```js
const fs = require('fs');           // synchronous, dynamic
module.exports = { fn };            // single export object
const { fn } = require('./utils'); // can require() inside if-blocks
```

- `require` resolves and executes the module synchronously on first call; subsequent calls return cached `module.exports`.
- Circular requires are handled by returning the partially-completed `module.exports` — can cause `undefined` exports.
- Not statically analyzable — no tree-shaking.

**ES Modules:**
```js
import { fn } from './utils.js';   // static; top-level only
import('./utils.js').then(...);    // dynamic import — returns Promise
export function fn() {}
export default class Foo {}
```

- Bindings are live — `import { count }` reflects updates to `count` in the exporting module.
- Static `import` declarations are hoisted and analyzed at parse time — enables tree-shaking.
- Circular imports resolve via live bindings (no partial export issue of CJS).
- `import.meta.url` — URL of the current module.

**Interop:** In Node.js, `.mjs` = ESM, `.cjs` = CJS. In `package.json`, `"type": "module"` makes `.js` files ESM. `require()` cannot import ESM; ESM can import CJS via default import only.

### Web APIs

**`fetch` with AbortController:**
```js
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 5000);

fetch(url, { signal: controller.signal })
    .then(r => r.json())
    .catch(err => {
        if (err.name === 'AbortError') console.log('timed out');
        else throw err;
    })
    .finally(() => clearTimeout(timeoutId));
```

**`IntersectionObserver`:** Reports when elements enter/exit the viewport without polling. Uses browser's compositor thread — zero scroll jank.

```js
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) loadImage(entry.target);
    });
}, { threshold: 0.1 });   // fire when 10% visible

document.querySelectorAll('img[data-src]').forEach(img => observer.observe(img));
```

**`ResizeObserver`:** Replaces window `resize` event for element-level size changes. Fires after layout, before paint.

**`requestIdleCallback`:** Schedule non-urgent work during browser idle periods (frame budget remaining). Falls back to `setTimeout` for non-browser environments.

### Service Workers

Service workers are scripts running in a separate worker thread with no DOM access. They intercept network requests for their scope (URL prefix).

**Lifecycle:** Register → Install (cache assets) → Activate (claim clients, delete old caches) → Fetch intercept.

```js
// sw.js
const CACHE = 'v1';
self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE).then(c => c.addAll(['/index.html', '/app.js']))
    );
});

self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(cached => cached ?? fetch(e.request))
    );
});
```

Strategies: Cache-first (offline-capable, stale), Network-first (always fresh, slow), Stale-while-revalidate (instant + background update).

Service workers use `postMessage` to communicate with the main thread. `BroadcastChannel` enables one-to-many communication across tabs.

### WASM Interop Basics

WebAssembly (WASM) is a binary instruction format for a stack-based VM. JavaScript loads and instantiates WASM modules:

```js
const { instance } = await WebAssembly.instantiateStreaming(
    fetch('lib.wasm'),
    { env: { memory: new WebAssembly.Memory({ initial: 10 }) } }
);
const result = instance.exports.compute(42);
```

WASM functions accept and return only numeric types (i32, i64, f32, f64). Passing strings or arrays requires writing to/reading from the shared `WebAssembly.Memory` buffer.

```js
const mem = new Uint8Array(instance.exports.memory.buffer);
// write string to WASM memory
const enc = new TextEncoder();
const bytes = enc.encode("hello");
mem.set(bytes, ptr);   // ptr from WASM allocator
```

WASM runs at near-native speed for compute-intensive tasks (image processing, codecs, physics). Does not bypass the JS security sandbox.

---

## Code Examples

### Custom Promise Implementation

```js
const STATE = { PENDING: 0, FULFILLED: 1, REJECTED: 2 };

class MyPromise {
    #state = STATE.PENDING;
    #value = undefined;
    #handlers = [];

    constructor(executor) {
        const resolve = (value) => this.#settle(STATE.FULFILLED, value);
        const reject  = (reason) => this.#settle(STATE.REJECTED, reason);
        try {
            executor(resolve, reject);
        } catch (e) {
            reject(e);
        }
    }

    #settle(state, value) {
        if (this.#state !== STATE.PENDING) return;
        if (value instanceof MyPromise) {
            value.then(
                v => this.#settle(STATE.FULFILLED, v),
                r => this.#settle(STATE.REJECTED, r)
            );
            return;
        }
        this.#state = state;
        this.#value = value;
        queueMicrotask(() => this.#handlers.forEach(h => this.#run(h)));
    }

    #run({ onFulfilled, onRejected, resolve, reject }) {
        const handler = this.#state === STATE.FULFILLED ? onFulfilled : onRejected;
        if (typeof handler !== 'function') {
            this.#state === STATE.FULFILLED ? resolve(this.#value) : reject(this.#value);
            return;
        }
        try {
            resolve(handler(this.#value));
        } catch (e) {
            reject(e);
        }
    }

    then(onFulfilled, onRejected) {
        return new MyPromise((resolve, reject) => {
            const handler = { onFulfilled, onRejected, resolve, reject };
            if (this.#state === STATE.PENDING) {
                this.#handlers.push(handler);
            } else {
                queueMicrotask(() => this.#run(handler));
            }
        });
    }

    catch(onRejected) { return this.then(undefined, onRejected); }

    finally(onFinally) {
        return this.then(
            v  => MyPromise.resolve(onFinally()).then(() => v),
            r  => MyPromise.resolve(onFinally()).then(() => { throw r; })
        );
    }

    static resolve(value) { return new MyPromise(res => res(value)); }
    static reject(reason) { return new MyPromise((_, rej) => rej(reason)); }
    static all(promises) {
        return new MyPromise((resolve, reject) => {
            const results = [];
            let remaining = promises.length;
            if (remaining === 0) { resolve(results); return; }
            promises.forEach((p, i) =>
                MyPromise.resolve(p).then(v => {
                    results[i] = v;
                    if (--remaining === 0) resolve(results);
                }, reject)
            );
        });
    }
}

MyPromise.resolve(1)
    .then(v => v + 1)
    .then(v => { throw new Error(`fail at ${v}`); })
    .catch(e => `caught: ${e.message}`)
    .then(console.log);
```

### Event Loop Demonstration

```js
function eventLoopDemo() {
    const log = [];

    log.push('sync 1');

    setTimeout(() => log.push('macrotask 1 (setTimeout 0)'), 0);

    Promise.resolve()
        .then(() => { log.push('microtask 1'); })
        .then(() => { log.push('microtask 2'); });

    queueMicrotask(() => log.push('microtask 3 (queueMicrotask)'));

    setTimeout(() => log.push('macrotask 2 (setTimeout 0)'), 0);

    log.push('sync 2');

    setTimeout(() => {
        console.log('Event loop order:');
        log.forEach((entry, i) => console.log(`  ${i + 1}. ${entry}`));
    }, 10);
}

eventLoopDemo();
// Order: sync 1, sync 2, microtask 1, microtask 3, microtask 2,
//        macrotask 1, macrotask 2
```

### Prototype Inheritance Chain

```js
function Vehicle(make, speed) {
    this.make = make;
    this.speed = speed;
}
Vehicle.prototype.describe = function() {
    return `${this.make} at ${this.speed} km/h`;
};

function Car(make, speed, doors) {
    Vehicle.call(this, make, speed);
    this.doors = doors;
}
Car.prototype = Object.create(Vehicle.prototype);
Car.prototype.constructor = Car;
Car.prototype.honk = function() { return `${this.make}: beep!`; };

function ElectricCar(make, speed, doors, range) {
    Car.call(this, make, speed, doors);
    this.range = range;
}
ElectricCar.prototype = Object.create(Car.prototype);
ElectricCar.prototype.constructor = ElectricCar;
ElectricCar.prototype.charge = function() {
    return `${this.make} charging; range ${this.range} km`;
};

const ec = new ElectricCar('Tesla', 200, 4, 500);
console.log(ec.describe());   // from Vehicle.prototype
console.log(ec.honk());       // from Car.prototype
console.log(ec.charge());     // own prototype

console.log('\nPrototype chain:');
let proto = Object.getPrototypeOf(ec);
while (proto) {
    console.log(' ', proto.constructor?.name ?? 'null');
    proto = Object.getPrototypeOf(proto);
}
```

### Module Pattern

```js
const EventBus = (() => {
    const listeners = new Map();

    function on(event, handler) {
        if (!listeners.has(event)) listeners.set(event, new Set());
        listeners.get(event).add(handler);
        return () => listeners.get(event)?.delete(handler);
    }

    function once(event, handler) {
        const off = on(event, function wrapper(...args) {
            handler(...args);
            off();
        });
        return off;
    }

    function emit(event, ...args) {
        listeners.get(event)?.forEach(fn => {
            try { fn(...args); }
            catch (e) { console.error(`EventBus handler error [${event}]:`, e); }
        });
    }

    function off(event, handler) {
        listeners.get(event)?.delete(handler);
    }

    function clear(event) {
        if (event) listeners.delete(event);
        else listeners.clear();
    }

    return { on, once, emit, off, clear };
})();

const unsub = EventBus.on('data', payload => console.log('received:', payload));
EventBus.once('data', payload => console.log('once:', payload));
EventBus.emit('data', { id: 1, value: 42 });
EventBus.emit('data', { id: 2, value: 99 });   // 'once' handler not called again
unsub();
EventBus.emit('data', { id: 3 });              // no handlers left
```

---

## Interview Q&A

**Q1: Explain the difference between microtasks and macrotasks and give an example of each.**

Macrotasks (tasks) are work items in the main task queue: `setTimeout`, `setInterval`, I/O callbacks, `postMessage`, `setImmediate` (Node.js). The event loop picks one macrotask per iteration. Microtasks are higher-priority callbacks that drain completely after the current task and before the next macrotask: `Promise.then/catch/finally`, `queueMicrotask`, `MutationObserver`. Stacking microtasks that produce more microtasks (a `.then` chain that each resolve immediately) can starve macrotasks indefinitely — the event loop never advances. This is why `setTimeout(fn, 0)` always runs after resolved promises, even if the promise was resolved synchronously before the `setTimeout` call.

---

**Q2: What are hidden classes in V8 and how can you write code that keeps property accesses monomorphic?**

V8 assigns each object a hidden class (Map) describing its property layout — names and their byte offsets. Objects with identical property sets in identical insertion order share a hidden class, enabling direct offset-based property access (equivalent to C struct). Monomorphic = one hidden class seen at a call site → maximally optimized. Rules for staying monomorphic: (1) Always initialize all properties in the constructor, never add them later. (2) Initialize properties in the same order across all instances. (3) Avoid `delete` (creates a different hidden class). (4) Do not mix types for the same property across instances. Megamorphic access (5+ shapes) causes V8 to give up on specialization and use the generic lookup path.

---

**Q3: What is the prototype chain, and what is the difference between `Object.create(null)` and `{}`?**

The prototype chain is a linked list of objects: each object's `[[Prototype]]` points to another object, ending at `Object.prototype` (whose `[[Prototype]]` is `null`). Property lookup walks the chain. `{}` creates an object with `Object.prototype` as its prototype — inheriting `toString`, `hasOwnProperty`, `valueOf`, etc. `Object.create(null)` creates an object with no prototype — no inherited properties, no `toString`. Use case: safe dictionaries/maps where you do not want inherited properties to interfere (e.g., a key of `"toString"` would shadow the inherited method on a plain object).

---

**Q4: How does `this` binding work, and what are the four rules?**

`this` is determined at call time, not definition time (exception: arrow functions).

1. **Default binding:** `fn()` in non-strict mode → `this = global/window`; strict mode → `undefined`.
2. **Implicit binding:** `obj.fn()` → `this = obj`.
3. **Explicit binding:** `fn.call(ctx, args)`, `fn.apply(ctx, [args])`, `fn.bind(ctx)` → `this = ctx`.
4. **`new` binding:** `new Fn()` → creates a new object; `this = new object`.

Priority: `new` > explicit > implicit > default. Arrow functions capture `this` lexically at definition time from the enclosing scope — no own `this`, `call`/`apply`/`bind` cannot change it.

---

**Q5: Explain how closures can cause memory leaks and how to prevent them.**

A closure keeps the entire lexical scope alive as long as the closure itself is reachable. Common leak: adding an event listener with a closure that captures a large object — the DOM node and the object both stay alive as long as the listener is attached, even if the DOM node is removed from the document. Prevention: (1) Remove event listeners when elements are removed (`removeEventListener` or `AbortController.signal`). (2) Use `WeakMap`/`WeakRef` for caches keyed on DOM nodes. (3) Avoid capturing large objects in closures; capture only the data needed. (4) Break long-lived closure chains by nulling references no longer needed.

---

**Q6: What are the differences between CommonJS and ES Modules, and why does ESM enable tree-shaking?**

CJS uses `require()` which is synchronous and dynamic (can be called in conditionals, inside functions, at any point). Exports are a plain object — a snapshot. ESM uses static `import` declarations that must be at the top level and are analyzed before execution. ESM bindings are live — an `import { count }` reflects mutations to `count` in the source module. Tree-shaking requires static analysis: a bundler (Webpack, Rollup) reads `import`/`export` statements without executing code and can determine which exports are never imported — then eliminates them. CJS `require` is dynamic; a bundler cannot know which exports will be used until runtime, making dead-code elimination unreliable.

---

**Q7: How does `async/await` transform under the hood, and why does `await` not block the thread?**

`async function` returns a Promise. `await expr` is syntactic sugar: it attaches a continuation (the rest of the function) as a `.then` callback on `expr`'s promise, then returns to the caller. The function "pauses" by simply not running its continuation — it is a callback scheduled in the microtask queue. The call stack is empty; the event loop is free to process other work. When the awaited promise settles, the continuation is queued as a microtask and runs in the next microtask checkpoint. No thread is blocked — the thread is busy doing other things or idling in the OS scheduler. This is fundamentally different from synchronous blocking (`while (!ready) {}`) which burns CPU and blocks the event loop.

---

**Q8: What is the difference between `==` and `===`, and what does Abstract Equality Comparison do?**

`===` (strict equality) requires same type and same value — no coercion. `==` (abstract equality) applies the Abstract Equality Comparison algorithm: if types differ, coerce operands. Key rules: `null == undefined` → `true` (and `null === undefined` → `false`); number vs string → convert string to number; boolean vs anything → convert boolean to number; object vs primitive → call `ToPrimitive` on object (`valueOf` then `toString`). Notorious gotchas: `[] == false` → `true` (empty array's `ToPrimitive` = `""`, `""` == `false` → `0 == 0`). Use `===` in all production code; use `==` only for intentional null-check shorthand (`x == null` catches both `null` and `undefined`).

---

**Q9: What is a `WeakMap` and when would you use it instead of a `Map`?**

`WeakMap` holds keys by weak reference — if no other reference to the key object exists, the GC can collect it, and the `WeakMap` entry vanishes automatically. Not enumerable (no `.keys()`, `.size`). Use `WeakMap` when: (1) Associating private data with DOM nodes without preventing their GC. (2) Memoization caches keyed on objects — cache cleans itself when object is collected. (3) Attaching metadata to objects in a library without leaking. Use regular `Map` when you need enumeration, string/number keys, or the `Map` should keep objects alive.

---

**Q10: How do service workers differ from Web Workers, and what is the stale-while-revalidate caching strategy?**

Web Workers run arbitrary JavaScript in a background thread with no DOM access — used for CPU-intensive tasks. They are tied to the creating page's lifetime. Service workers are a special proxy worker that intercepts network requests for their registered scope, survives page close (until idle-terminated by the browser), and enable offline functionality. Stale-while-revalidate: on a fetch, immediately return the cached response (fast, possibly stale), and simultaneously trigger a background network request to update the cache. The next request gets the fresh content. Ideal for resources that update frequently but where a slightly stale version is acceptable (avatars, non-critical UI assets). Implemented with `cache.match(request).then(cached => { const update = fetch(request).then(r => cache.put(request, r.clone())); return cached ?? update; })`.

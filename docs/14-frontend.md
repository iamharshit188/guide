# Module 14: Frontend Architecture (React Context & Styling)

## Intuition
Modern frontends build upon a Directed Acyclic Graph (DAG) of UI components. Instead of manually mutating the document (DOM) imperatively, we declare the UI as a pure function of application state: $UI = f(state)$. When the state updates, the framework calculates the optimal set of mutations to transition the DOM to the new state cleanly and efficiently.

## 1. JavaScript Execution Model
Before analyzing React, one must understand the underlying JS engine. JavaScript is single-threaded, reliant on the Event Loop to handle concurrency.
### The Event Loop
- **Call Stack:** Executes synchronous block code sequentially.
- **Microtask Queue:** Resolves Promises (`.then()`, `await`).
- **Macrotask Queue:** Executes web APIs (`setTimeout`, `setInterval`).
*Microtasks are always flushed completely before the next Macrotask begins, making Promise resolution strictly prioritized.*

### Closures and Scope
A closure allows a function to retain access to its lexical scope even when executed outside that scope. This is the fundamental mechanism behind React's Hooks memory allocation and state retention between renders.

## 2. React Fundamentals & The Virtual DOM
React keeps a lightweight JavaScript representation of the DOM in memory (Virtual DOM). 

### Fiber Architecture & Reconciliation
When state changes, React runs a diffing algorithm called Reconciliation. Exact tree diffing is $O(n^3)$, but React applies a heuristic $O(n)$ algorithm resting on two assumptions:
1. Different component types yield entirely different trees.
2. Sibling elements can be tracked as stable using a unique `key` prop, preventing unnecessary unmounting.

### Core Hooks Rigor
- **`useState(initialState)`:** Returns a stateful value and an updater function. In the Fiber node, states are stored as a linked list in memory order. Calling the updater schedules a re-render.
- **`useEffect(setup, dependencies)`:** Handles side effects (network requests, subscriptions). React compares the `dependencies` array against the previous render using `Object.is()`. If evaluated to false, the `setup` function executes asynchronously after the UI paint sequence.

## 3. Tailwind CSS & Atomic Design
Traditional CSS suffers from global scope bloat and specificity wars. Tailwind provides atomic, single-purpose utility classes natively scaling to complex UI patterns.

### PostCSS Compilation
Tailwind is not a runtime CSS library; it is a PostCSS compiler plugin. It parses the JavaScript/HTML Abstract Syntax Tree (AST), extracts utility class tokens, and generates the exact CSS required while purging unused styles.
- Typical button class format: `px-4 py-2 bg-blue-600 text-white rounded shadow-sm hover:bg-blue-700`
- Architecture guarantee: It ensures minimal CSS payload and predictable $O(1)$ specificity computation in the browser engine.

## 4. State Management and Data Fetching
### Unidirectional Data Flow
Data strictly flows down the component DAG via `props`. Events flow linearly up via callback functions. For global states (e.g., User Identity, Theme), Context Providers bypass manual prop-drilling by providing a broadcast subscription mechanism to wrapped sub-trees.

### Async Interceptors
When connecting ML backends, wrapping `fetch` utilities to seamlessly append authorization headers (JWT Bearer tokens) centrally prevents fragmented security implementations across the component tree.

🏁 End of Module 14

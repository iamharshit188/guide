# Module 14: Frontend Architecture (React Context & Styling)

## Intuition: The UI as a Function of State
Modern frontends build upon a Directed Acyclic Graph (DAG) of UI components. Instead of manually mutating the document (DOM) imperatively, we declare the UI as a pure function of application state: $UI = f(state)$. When the state updates, the framework calculates the optimal set of mutations to transition the DOM to the new state cleanly and efficiently.

---

## 1. Core JavaScript (ES6) Prerequisites
Before learning React, you must understand the syntax of modern JavaScript. JS is a single-threaded language utilizing an Event Loop.

### Destructuring & Arrow Functions
Destructuring allows extracting variables from objects instantly. Arrow functions provide concise syntax and inherit lexical `this`.
```javascript
// Object definition
const modelUser = { name: "Harshit", role: "admin", tokens: 1500 };

// Extracting properties (Destructuring)
const { name, role } = modelUser;

// Arrow function mapping
const numbers = [1, 2, 3];
const squared = numbers.map(x => x * x); // [1, 4, 9]
```

### Async/Await and Promises
Web requests (like calling your ML API) take time. The `await` keyword pauses execution of the local block until the Promise resolves inside the microtask queue.
```javascript
async function fetchPredictions(prompt) {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ text: prompt })
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Network failed:", error);
  }
}
```

---

## 2. React Fundamentals & JSX
React keeps a lightweight JavaScript representation of the DOM in memory (Virtual DOM). When state changes, its Fiber Architecture runs a heuristic $O(n)$ diffing algorithm (Reconciliation) to update only what changed.

### Writing a Component (JSX)
A component is just a function returning JSX (HTML embedded inside JavaScript).
```jsx
// A simple functional component taking 'props' (arguments)
function ModelCard({ title, paramSize }) {
  return (
    <div className="card-container">
      <h2>{title}</h2>
      <p>Parameter Size: {paramSize} Billion</p>
    </div>
  );
}

// Using the component
export function App() {
  return (
    <section>
      <ModelCard title="Llama 3" paramSize={70} />
      <ModelCard title="GPT-4" paramSize={1800} />
    </section>
  );
}
```

---

## 3. Core React Hooks in Action
Hooks let you "hook into" React's internal fiber node memory.

### `useState`: Remembering Data
State triggers a re-render when its value changes.
```jsx
import { useState } from 'react';

export function TokenCounter() {
  // Destructuring the array: [currentValue, updaterFunction]
  const [tokens, setTokens] = useState(0);

  return (
    <div>
      <p>Tokens used: {tokens}</p>
      {/* Updating state via an anonymous arrow function */}
      <button onClick={() => setTokens(tokens + 10)}>
        Generate Text
      </button>
    </div>
  );
}
```

### `useEffect`: Handling Side Effects
Effect hooks run code *after* the UI paints. Perfect for API calls.
```jsx
import { useState, useEffect } from 'react';

export function Dashboard() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // This runs exactly once on mount because the dependency array [] is empty
    async function loadMetrics() {
      const data = await fetch('/api/metrics').then(res => res.json());
      setMetrics(data);
    }
    loadMetrics();
  }, []); // <-- Dependency array

  if (!metrics) return <p>Loading model weights...</p>;
  return <div>Active Users: {metrics.users}</div>;
}
```

---

## 4. Tailwind CSS: Code-Level Styling
Tailwind is a PostCSS compiler plugin that extracts utility class tokens from your JS Abstract Syntax Tree (AST) and generates minimal CSS with $O(1)$ specificity computation.

You style elements by composing utility classes directly in the `className`.

```jsx
// Building a clean, responsive button without writing any CSS files
export function SubmitButton({ label }) {
  return (
    <button 
      className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg 
                 shadow-md hover:bg-blue-700 transition duration-300 ease-in-out"
    >
      {label}
    </button>
  );
}
```
**Common Utilities:**
- `flex flex-col items-center`: Transforms a div into a centered vertical flexbox.
- `w-full max-w-4xl`: Makes the element 100% wide but caps it at an optimal reading width.
- `p-4 m-2`: Applies 1rem padding and 0.5rem margin.

🏁 End of Module 14

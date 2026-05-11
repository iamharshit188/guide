# Module 14: Frontend (React + Tailwind)

## ✨ Beginner Foundations: Getting an Intuition
Think of a web application like a Lego set. Instead of building the entire ship in one massive chunk of plastic (traditional HTML/JS), you build small, reusable blocks (React Components). 
- **React** is the blueprint and the connective mechanics that hold the legos together.
- **Tailwind CSS** is your massive box of paint and stickers allowing you to instantly style those blocks without leaving your HTML file.

Before React, we must clear up three key JavaScript (ES6) concepts.

---

## 1. JavaScript Prerequisites (The Non-Negotiables)
You don't need to be a JS master, but you must know these.

### A. Destructuring
Pulling variables out of objects or arrays instantly.
```javascript
const user = { name: "Harshit", role: "Admin" };
// Old way
const name = user.name; 
// ES6 Way
const { name, role } = user;
```

### B. Promises & Async/Await
Web requests take time. We use `async/await` so our code waits for the server response without freezing the whole site.
```javascript
async function fetchModels() {
  const response = await fetch("https://api.openai.com/v1/models");
  const data = await response.json();
  console.log(data);
}
```

### C. Array `.map()`
React doesn't use `for` loops to render lists. It maps an array of data directly into UI components.
```javascript
const fruits = ['Apple', 'Banana'];
const listHtml = fruits.map(f => `<li>${f}</li>`);
```

---

## 2. React Fundamentals

### The Component
A Component is just a JavaScript function that returns HTML (specifically, JSX).
```jsx
function ModelCard({ title, params }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <p>Parameters: {params}B</p>
    </div>
  );
}
```

### State (`useState`)
State is React's memory. When state changes, the UI automatically updates (re-renders).
```jsx
import { useState } from 'react';

function LikeButton() {
  // state variable, setter function = useState(initial value)
  const [likes, setLikes] = useState(0);

  return (
    <button onClick={() => setLikes(likes + 1)}>
      Likes: {likes}
    </button>
  );
}
```

### Effects (`useEffect`)
Use this when you need something to happen automatically *after* the component loads (like fetching data from your ML backend).
```jsx
import { useEffect, useState } from 'react';

function Dashboard() {
  const [data, setData] = useState(null);

  useEffect(() => {
    // This runs exactly ONCE when the component mounts
    fetch('/api/metrics').then(res => res.json()).then(setData);
  }, []); // Empty array = run once

  return <div>{data ? data.status : "Loading..."}</div>;
}
```

---

## 3. Tailwind CSS
Tailwind avoids massive `.css` files by giving you raw utility classes directly in the `className` attribute.
Instead of:
```css
.btn { display: flex; padding: 10px; background-color: blue; border-radius: 8px; }
```
You write:
```jsx
<button className="flex p-2 bg-blue-500 rounded-md text-white hover:bg-blue-600">
  Click Me
</button>
```

**Common Tailwind utilities you will use constantly:**
- `flex flex-col items-center justify-center` (Centered column layout)
- `w-full max-w-2xl` (Responsive width)
- `p-4 m-2` (Padding and Margin)
- `text-lg font-bold text-gray-800` (Typography)

## 📌 Summary
With **React** managing your Application State and **Tailwind** styling your components, you can build production-ready UIs rapidly to serve your backend AI models.

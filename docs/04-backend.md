# Module 4: Backend Engineering with Flask, Authentication, APIs, and Cloud Databases

# Backend Engineering for AI Products

---

# 1. Understanding Backend Systems

## What is a Backend?

A backend is the server-side logic of an application. It is responsible for:

* Processing requests
* Running business logic
* Communicating with databases
* Authenticating users
* Running AI/ML models
* Returning responses to the frontend

If the frontend is what users see, the backend is the hidden engine powering everything.

---

# Real World Analogy

Think of a food delivery app:

| Component      | Real World Equivalent  |
| -------------- | ---------------------- |
| Frontend       | Waiter                 |
| Backend        | Restaurant Kitchen     |
| Database       | Storage Room           |
| Authentication | Security Guard         |
| AI Model       | Specialist Chef        |
| API            | Communication Language |

Flow:

User → Frontend → Backend → Database / AI → Backend → Frontend → User

---

# Why Backend is Important in AI Products

AI models alone are not products.

A production AI application requires:

* APIs
* Authentication
* Databases
* User management
* File storage
* Billing systems
* Logging
* Deployment infrastructure

Without backend engineering:

* Models remain local experiments
* No scalability
* No user handling
* No persistence
* No production system

---

# 2. Understanding How the Internet Works

Before learning Flask, understand the HTTP request lifecycle.

---

# Client-Server Architecture

```text
Browser / React Frontend
        |
        | HTTP Request
        v
Flask Backend Server
        |
        | Database Query / AI Processing
        v
Database or ML Model
        |
        | Response
        v
Frontend
```

---

# HTTP Basics

HTTP = HyperText Transfer Protocol

It is the communication protocol of the web.

---

# Important HTTP Methods

| Method | Purpose        |
| ------ | -------------- |
| GET    | Fetch data     |
| POST   | Create data    |
| PUT    | Update data    |
| PATCH  | Partial update |
| DELETE | Remove data    |

---

# Example

```http
GET /users
```

Fetch all users.

```http
POST /login
```

Attempt login.

---

# Status Codes

| Code | Meaning      |
| ---- | ------------ |
| 200  | Success      |
| 201  | Created      |
| 400  | Bad Request  |
| 401  | Unauthorized |
| 403  | Forbidden    |
| 404  | Not Found    |
| 500  | Server Error |

---

# JSON: The Language of APIs

Modern APIs communicate using JSON.

Example:

```json
{
  "name": "Alpha",
  "role": "Student"
}
```

---

# 3. What is Flask?

## Flask Overview

Flask is a lightweight Python web framework used for:

* APIs
* AI backends
* Web applications
* Authentication systems
* Microservices

Flask is minimal and flexible.

---

# Why Flask is Popular for AI

| Reason         | Explanation                |
| -------------- | -------------------------- |
| Python-based   | Easy ML integration        |
| Lightweight    | Fast development           |
| Flexible       | No forced architecture     |
| Huge ecosystem | Many libraries             |
| API-friendly   | Excellent for React/NextJS |

---

# Flask vs FastAPI

| Feature | Flask | FastAPI |
|---|---|
| Simplicity | Very Easy | Moderate |
| Performance | Good | Excellent |
| Async Support | Limited | Native |
| AI Usage | Very Common | Growing Fast |
| Learning Curve | Beginner Friendly | Slightly Higher |

For beginners:

* Learn Flask first
* Then move to FastAPI later

---

# 4. Installing Flask Properly

# Install Python

Check installation:

```bash
python --version
```

---

# Create Virtual Environment

Virtual environments isolate project dependencies.

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

# Install Flask

```bash
pip install flask
```

Verify:

```bash
pip list
```

---

# Recommended Packages for AI Backends

```bash
pip install flask
pip install python-dotenv
pip install flask-cors
pip install gunicorn
pip install sqlalchemy
pip install psycopg2-binary
pip install supabase
pip install firebase-admin
pip install pyjwt
```

---

# Project Structure

```text
backend/
│
├── app.py
├── requirements.txt
├── .env
├── routes/
├── models/
├── services/
├── utils/
├── static/
├── templates/
└── venv/
```

---

# 5. Your First Flask Server

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Backend Running"

if __name__ == '__main__':
    app.run(debug=True)
```

---

# Running Flask

```bash
python app.py
```

Output:

```text
Running on http://127.0.0.1:5000
```

---

# Understanding the Code

| Line            | Purpose          |
| --------------- | ---------------- |
| Flask(**name**) | Creates app      |
| @app.route()    | Creates endpoint |
| def home()      | Route logic      |
| app.run()       | Starts server    |

---

# 6. Flask Routing Deep Dive

Routes map URLs to Python functions.

---

# Basic Route

```python
@app.route('/about')
def about():
    return "About Page"
```

---

# Dynamic Routes

```python
@app.route('/user/<username>')
def user(username):
    return f"Hello {username}"
```

---

# Route Methods

```python
@app.route('/submit', methods=['POST'])
def submit():
    return "Submitted"
```

---

# Multiple Methods

```python
@app.route('/api/data', methods=['GET', 'POST'])
def data():
    if request.method == 'GET':
        return "Fetching"

    return "Creating"
```

---

# 7. Request and Response Objects

## Reading JSON Requests

```python
from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    text = data.get('text')

    return {"received": text}
```

---

# Example Request

```json
{
  "text": "Hello AI"
}
```

---

# Returning JSON Responses

```python
from flask import jsonify

return jsonify({
    "status": "success",
    "prediction": "positive"
})
```

---

# 8. Building APIs Properly

## REST API Principles

REST = Representational State Transfer

Good APIs:

* Predictable
* Structured
* Stateless
* JSON-based

---

# Good API Naming

| Bad         | Good        |
| ----------- | ----------- |
| /getusers   | /users      |
| /createUser | /users      |
| /deleteuser | /users/<id> |

---

# Example REST API

```python
@app.route('/todos', methods=['GET'])
def get_todos():
    pass

@app.route('/todos', methods=['POST'])
def create_todo():
    pass

@app.route('/todos/<id>', methods=['DELETE'])
def delete_todo(id):
    pass
```

---

# 9. Connecting Frontend to Flask

# React Fetch Example

```javascript
fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        text: "Hello"
    })
})
```

---

# Flask CORS

Browsers block unknown origins.

Install:

```bash
pip install flask-cors
```

Use:

```python
from flask_cors import CORS

CORS(app)
```

---

# 10. Authentication Fundamentals

# What is Authentication?

Authentication verifies identity.

Examples:

* Password login
* Google login
* GitHub OAuth
* JWT authentication

---

# Authentication vs Authorization

| Term           | Meaning              |
| -------------- | -------------------- |
| Authentication | Who are you?         |
| Authorization  | What can you access? |

---

# Why JWT is Important

JWT = JSON Web Token

JWTs allow stateless authentication.

---

# JWT Structure

A JWT contains:

```text
HEADER.PAYLOAD.SIGNATURE
```

---

# JWT Authentication Flow

```text
User Logs In
      |
      v
Server Generates JWT
      |
      v
Frontend Stores JWT
      |
      v
JWT Sent in Headers
      |
      v
Backend Verifies JWT
```

---

# Authorization Header

```http
Authorization: Bearer TOKEN
```

---

# 11. Firebase Authentication

## Why Firebase Auth?

Firebase handles:

* Password hashing
* Google login
* Session management
* Security

Never build authentication manually in production.

---

# Installing Firebase Admin SDK

```bash
pip install firebase-admin
```

---

# Firebase Verification

```python
from firebase_admin import auth

def verify_user(token):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']

    except Exception:
        return None
```

---

# Protected Route Example

```python
@app.route('/protected')
def protected():

    token = request.headers.get("Authorization")

    user = verify_user(token)

    if not user:
        return jsonify({
            "error": "Unauthorized"
        }), 401

    return jsonify({
        "message": "Access Granted"
    })
```

---

# 12. Environment Variables

Never hardcode secrets.

BAD:

```python
API_KEY = "my-secret-key"
```

GOOD:

```python
import os

API_KEY = os.getenv("API_KEY")
```

---

# Using .env Files

Install:

```bash
pip install python-dotenv
```

Create:

```env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
SECRET_KEY=my_secret
```

Load:

```python
from dotenv import load_dotenv

load_dotenv()
```

---

# 13. Databases for AI Applications

# Why Databases Matter

AI applications need storage for:

* Users
* Prompts
* AI responses
* Usage tracking
* Billing
* Analytics

---

# SQL vs NoSQL

| SQL              | NoSQL            |
| ---------------- | ---------------- |
| Structured       | Flexible         |
| Tables           | Documents        |
| PostgreSQL       | MongoDB          |
| Strong relations | Flexible schemas |

For production AI apps:

* PostgreSQL is usually best.

---

# PostgreSQL Overview

Postgres is:

* Open source
* Powerful
* Reliable
* Production-grade

Used by:

* Supabase
* Railway
* Render
* Enterprise systems

---

# 14. Supabase

## What is Supabase?

Supabase provides:

* PostgreSQL Database
* Authentication
* Storage
* APIs
* Realtime features

---

# Installing Supabase SDK

```bash
pip install supabase
```

---

# Connecting to Supabase

```python
from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)
```

---

# Insert Data

```python
supabase.table("todos").insert({
    "title": "Learn Flask"
}).execute()
```

---

# Fetch Data

```python
data = supabase.table("todos").select("*").execute()
```

---

# Update Data

```python
supabase.table("todos").update({
    "title": "Updated"
}).eq("id", 1).execute()
```

---

# Delete Data

```python
supabase.table("todos").delete().eq("id", 1).execute()
```

---

# 15. SQLAlchemy ORM

## What is ORM?

ORM = Object Relational Mapping

Instead of SQL:

```sql
SELECT * FROM users;
```

You write Python:

```python
User.query.all()
```

---

# Installing SQLAlchemy

```bash
pip install flask-sqlalchemy
```

---

# Basic Model

```python
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
```

---

# Creating Database

```python
db.create_all()
```

---

# Insert Record

```python
todo = Todo(title="Study Flask")

db.session.add(todo)
db.session.commit()
```

---

# 16. AI Backend Architecture

Typical AI backend flow:

```text
Frontend
   |
   v
Flask API
   |
   v
Authentication Layer
   |
   v
Validation Layer
   |
   v
AI Service Layer
   |
   v
Database
```

---

# Service Layer Pattern

Do not place AI logic directly in routes.

BAD:

```python
@app.route('/ai')
def ai():
    huge_ai_code()
```

GOOD:

```python
@app.route('/ai')
def ai():
    result = generate_response()
    return jsonify(result)
```

---

# services/ai_service.py

```python
def generate_response(prompt):
    return model.predict(prompt)
```

---

# 17. File Uploads

AI apps often need:

* PDFs
* Images
* Audio
* Video

---

# Flask File Upload Example

```python
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']

    file.save(f"uploads/{file.filename}")

    return "Uploaded"
```

---

# 18. Building a TODO API Project

# Goal

Build:

* Flask backend
* PostgreSQL/Supabase DB
* CRUD operations
* Authentication-ready architecture

---

# Step 1: Create Project

```bash
mkdir flask-todo
cd flask-todo
```

---

# Step 2: Setup Environment

```bash
python -m venv venv
```

Install:

```bash
pip install flask
pip install flask-sqlalchemy
pip install flask-cors
```

---

# Step 3: app.py

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'

db = SQLAlchemy(app)

class Todo(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(db.String(200))

@app.route('/todos', methods=['GET'])
def get_todos():

    todos = Todo.query.all()

    result = []

    for todo in todos:
        result.append({
            "id": todo.id,
            "title": todo.title
        })

    return jsonify(result)

@app.route('/todos', methods=['POST'])
def create_todo():

    data = request.json

    todo = Todo(title=data['title'])

    db.session.add(todo)

    db.session.commit()

    return jsonify({
        "message": "Todo created"
    })

if __name__ == '__main__':

    with app.app_context():
        db.create_all()

    app.run(debug=True)
```

---

# API Testing with Postman

POST Request:

```json
{
  "title": "Learn Backend"
}
```

---

# Expected Response

```json
{
  "message": "Todo created"
}
```

---

# 19. Production Deployment

# Development vs Production

| Development  | Production      |
| ------------ | --------------- |
| Debug Mode   | Secure          |
| Localhost    | Public Internet |
| SQLite       | PostgreSQL      |
| Flask Server | Gunicorn/Nginx  |

---

# Gunicorn

Install:

```bash
pip install gunicorn
```

Run:

```bash
gunicorn app:app
```

---

# Deployment Platforms

| Platform     | Usage                |
| ------------ | -------------------- |
| Render       | Easy Flask hosting   |
| Railway      | Fullstack deployment |
| Fly.io       | Docker-based         |
| AWS          | Enterprise           |
| DigitalOcean | VPS hosting          |

---

# 20. Docker Basics

## Why Docker?

Docker packages:

* Code
* Dependencies
* Environment

Into one portable container.

---

# Simple Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

---

# Build Docker Image

```bash
docker build -t flask-app .
```

---

# Run Container

```bash
docker run -p 5000:5000 flask-app
```

---

# 21. Backend Security

# Important Rules

Never:

* Trust frontend validation
* Store plain passwords
* Expose API keys
* Disable authentication

---

# Rate Limiting

Prevents spam attacks.

Libraries:

* Flask-Limiter

---

# Input Validation

Always validate inputs.

BAD:

```python
title = request.json['title']
```

BETTER:

```python
title = request.json.get('title')

if not title:
    return {"error": "Missing title"}
```

---

# 22. Logging and Debugging

# Logging Example

```python
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Server started")
```

---

# Common Backend Errors

| Error      | Cause                     |
| ---------- | ------------------------- |
| 404        | Route missing             |
| 500        | Server crashed            |
| CORS Error | Frontend blocked          |
| JWT Error  | Invalid token             |
| DB Error   | Database connection issue |

---

# 23. API Testing Tools

| Tool     | Purpose                 |
| -------- | ----------------------- |
| Postman  | API testing             |
| Insomnia | Lightweight API testing |
| curl     | Terminal testing        |

---

# curl Example

```bash
curl -X POST http://localhost:5000/todos \
-H "Content-Type: application/json" \
-d '{"title":"Learn Flask"}'
```

---

# 24. Suggested Folder Structure for Production

```text
backend/
│
├── app/
│   ├── routes/
│   ├── models/
│   ├── services/
│   ├── middleware/
│   ├── utils/
│   └── config/
│
├── uploads/
├── tests/
├── requirements.txt
├── Dockerfile
├── .env
└── run.py
```

---

# 25. Practical AI API Example

```python
@app.route('/summarize', methods=['POST'])
def summarize():

    data = request.json

    text = data.get("text")

    result = ai_model.summarize(text)

    return jsonify({
        "summary": result
    })
```

---

# 26. Recommended GitHub Repositories

## Flask

* Flask Official:

  * [https://github.com/pallets/flask](https://github.com/pallets/flask)

* Flask Mega Tutorial:

  * [https://github.com/miguelgrinberg/flasky](https://github.com/miguelgrinberg/flasky)

---

# FastAPI

* FastAPI:

  * [https://github.com/fastapi/fastapi](https://github.com/fastapi/fastapi)

---

# SQLAlchemy

* SQLAlchemy:

  * [https://github.com/sqlalchemy/sqlalchemy](https://github.com/sqlalchemy/sqlalchemy)

---

# Supabase

* Supabase:

  * [https://github.com/supabase/supabase](https://github.com/supabase/supabase)

---

# Firebase

* Firebase Admin Python:

  * [https://github.com/firebase/firebase-admin-python](https://github.com/firebase/firebase-admin-python)

---

# 27. Glossary

| Term       | Meaning                      |
| ---------- | ---------------------------- |
| API        | Communication interface      |
| Endpoint   | URL exposed by backend       |
| JWT        | Authentication token         |
| ORM        | Database abstraction layer   |
| CRUD       | Create Read Update Delete    |
| CORS       | Cross-origin communication   |
| Middleware | Logic before route execution |
| Deployment | Making app public            |
| Gunicorn   | Production WSGI server       |
| PostgreSQL | Relational database          |

---

# 28. Final Production Flow

```text
React Frontend
       |
       v
JWT Authentication
       |
       v
Flask Backend API
       |
       v
Middleware Validation
       |
       v
Business Logic Layer
       |
       v
AI Processing Layer
       |
       v
PostgreSQL / Supabase
       |
       v
JSON Response
       |
       v
Frontend UI
```

---

# Key Learning Outcomes

After completing this module, you should understand:

* Flask fundamentals
* API routing
* Authentication systems
* JWT verification
* Cloud databases
* PostgreSQL basics
* Supabase integration
* AI backend architecture
* CRUD APIs
* Deployment
* Docker basics
* Production security
* Backend project structure
* Fullstack communication
* AI product backend engineering

# Module 4: Backend (Flask, Auth, and Cloud DBs)

## ✨ Beginner Foundations: Getting an Intuition
Your Machine Learning model is useless if it's trapped on your laptop. The **Backend** is the restaurant kitchen. The user (waiter) takes a request, passes it to the backend (kitchen), the backend securely queries databases and AI models (cooks the food), and serves it back to the user.

- **Flask / FastAPI:** The framework that listens to the internet and routes requests.
- **Supabase / PostgreSQL:** Your persistent structured storage (Users, Posts).
- **Firebase Auth:** The terrifyingly secure bouncer checking IDs before anyone enters the kitchen.

---

## 1. Flask & API Routing
An API endpoint is just a function exposed to the internet.
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # 1. Run ML Model
    result = dummy_model_predict(text)
    
    # 2. Return JSON to React frontend
    return jsonify({"prediction": result, "status": "success"})
```

## 2. Authentication (Firebase / JWT)
Never build your own password hashes. Use Firebase or Supabase Auth.
1. The user logs in on your React frontend using Google OAuth via Firebase SDK.
2. Firebase gives the React frontend a **JWT (JSON Web Token)**.
3. React attaches that JWT to the headers of every request sent to your Flask Backend: `Authorization: Bearer <TOKEN>`.
4. Flask verifies the token mathematically using the Firebase Admin SDK to ensure the user is real before running expensive AI tasks.

```python
from firebase_admin import auth

def verify_user(token):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid'] # Secure User ID
    except Exception as e:
        return None
```

## 3. Cloud Database (Supabase)
Supabase is an open-source Firebase alternative heavily revolving around Postgres. It provides a REST API over a standard database. We can query it via Python `supabase-py` inside our Flask routes.

```python
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Inside a route:
def save_generation(user_id, prompt, response):
    data, count = supabase.table("generations").insert({
        "user_id": user_id, 
        "prompt": prompt, 
        "response": response
    }).execute()
```

## 📌 Bringing it together
The typical production flow:
`React Frontend (User clicks button)` ➔ `Sends JWT + Data` ➔ `Flask Route Validates JWT` ➔ `Flask queries Supabase for quota` ➔ `Flask runs ML Model` ➔ `Flask saves result to Supabase` ➔ `Returns JSON to React`.

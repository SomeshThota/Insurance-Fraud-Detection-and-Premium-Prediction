# 🛡️ Insurance Fraud Detection & Premium Prediction

A full-stack Machine Learning web application built with Flask that helps insurance companies detect fraudulent claims and helps customers predict their insurance premiums.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 Fraud Detection | ML model detects fraudulent insurance claims with risk scoring |
| 💰 Premium Prediction | Predicts recommended insurance premium amount |
| 📋 Plan Recommendation | Suggests the best insurance plan based on user profile |
| 👤 User Authentication | Secure login/signup with role-based access (Customer & Admin) |
| 📊 Admin Dashboard | Visual analytics on fraud patterns and claim distributions |
| 📝 Claim Submission | Customers can submit claims for admin review |
| ⚙️ Admin Claim Management | Admins can review, approve or flag claims as fraud |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Models | Scikit-learn, Joblib |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, Jinja2 Templates |
| Database | SQLite |
| Auth | Werkzeug (Password Hashing) |

---

## 🧠 ML Models

- **Fraud Detection Model** — Classifies claims as fraudulent or genuine based on 20 features including claim amount, policy tenure, prior fraud history, accident type and more
- **Premium Amount Model** — Regression model predicting insurance premium based on age, vehicle price, driving experience etc.
- **Premium Plan Model** — Classification model recommending the best insurance plan

### Fraud Risk Scoring System
Beyond ML prediction, the app uses a custom rule-based risk scoring system:
- Policy tenure < 30 days → High risk
- Prior fraud history → Very high risk
- Claim amount > Vehicle market value → High risk
- No police report filed → Medium risk
- Long claim delay → Medium risk

---

## 📁 Project Structure

```
insurance_project/
│
├── templates/
│   ├── index.html               # Home page
│   ├── login.html               # Login page
│   ├── signup.html              # Signup page
│   ├── fraud.html               # Fraud detection form (Admin)
│   ├── premium.html             # Combined premium form
│   ├── submit_claim.html        # Customer claim submission
│   └── admin_claims.html        # Admin claims list
│
├── app.py                       # Main Flask application
├── fraud_model.py               # Fraud model training script
├── premium_model.py             # Premium model training script
├── generate_fraud_dataset.py    # Synthetic fraud dataset generator
├── generate_premium_dataset.py  # Synthetic premium dataset generator
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/SomeshThota/Insurance-Fraud-Detection-and-Premium-Prediction.git
cd Insurance-Fraud-Detection-and-Premium-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate datasets and train models
```bash
python generate_fraud_dataset.py
python generate_premium_dataset.py
python fraud_model.py
python premium_model.py
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## 👥 User Roles

### Customer
- Sign up and log in
- Get premium amount prediction
- Get insurance plan recommendation
- Submit insurance claims

### Admin
- Default credentials: `admin` / `admin123`
- Access fraud detection tool
- View admin dashboard with analytics
- Review and manage all submitted claims
- Mark claims as genuine or fraudulent

---

## 🔑 Key Implementation Details

- Passwords hashed using **Werkzeug** — never stored in plain text
- Role-based access using **custom decorators** (`@login_required`, `@admin_required`)
- Session management with Flask sessions
- SQLite database with foreign key relationships
- ML models loaded once at startup using **Joblib** for performance

---

## 📦 Requirements

```
flask
pandas
numpy
scikit-learn
joblib
werkzeug
```

---

## 🙋 Author

**Somesh Thota**  
GitHub: [@SomeshThota](https://github.com/SomeshThota)
from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "dev-secret-key"  # TODO: move to environment variable in production

# ---------------------------
# Database helpers (SQLite)
# ---------------------------
DB_PATH = "insurance.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Users table: basic auth with roles (customer/admin)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('customer', 'admin'))
        )
        """
    )

    # Claims table: customer-submitted claims admins can review
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            policy_tenure_days INTEGER,
            premium_amount REAL,
            coverage_amount REAL,
            policy_recently_upgraded INTEGER,
            customer_age INTEGER,
            num_previous_claims INTEGER,
            prior_fraud_flag INTEGER,
            late_premium_history INTEGER,
            claim_amount REAL,
            repair_estimate REAL,
            claim_delay_days INTEGER,
            police_report_filed INTEGER,
            witness_present INTEGER,
            photos_submitted INTEGER,
            accident_time INTEGER,
            accident_type INTEGER,
            weather_condition INTEGER,
            vehicle_age INTEGER,
            vehicle_market_value REAL,
            injury_reported INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()

    # Create a default admin user if not present
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin"),
        )
        conn.commit()

    conn.close()


# Initialize database once at startup
init_db()


# ---------------------------
# Auth decorators
# ---------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            next_url = request.path
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)

    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "admin":
            next_url = request.path
            flash("Admin login required to access this page.", "warning")
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)

    return wrapper

# ---------------------------
# Load trained models
# ---------------------------
# ✅ Fix - train if models don't exist
import os

if not os.path.exists("premium_amount_model.pkl"):
    os.system("python generate_premium_dataset.py")
    os.system("python train_premium_model.py")

if not os.path.exists("fraud_model.pkl"):
    os.system("python generate_fraud_dataset.py")
    os.system("python fraud_model.py")

premium_amount_model = joblib.load("premium_amount_model.pkl")
premium_plan_model = joblib.load("premium_plan_model.pkl")
fraud_model = joblib.load("fraud_model.pkl")
# ---------------------------
# Home & Auth Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("signup.html")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), "customer"),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username already taken. Please choose another.", "danger")
            conn.close()
            return render_template("signup.html")

        # Log the user in after successful signup
        cur.execute("SELECT id, role FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        session["user_id"] = user["id"]
        session["username"] = username
        session["role"] = user["role"]

        flash("Signup successful. You are now logged in.", "success")
        return redirect(url_for("home"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        flash("Logged in successfully.", "success")
        next_url = request.args.get("next")
        return redirect(next_url or url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/premium_login")
def premium_login():
    if "user_id" not in session:
        return redirect(url_for("login", next="/premium"))
    return redirect(url_for("premium"))


@app.route("/admin_login")
def admin_login():
    if "user_id" not in session or session.get("role") != "admin":
        flash("Admin login required.", "warning")
        return redirect(url_for("login", next="/fraud"))
    return redirect(url_for("fraud"))

# ---------------------------
# Premium Amount Prediction
# ---------------------------
@app.route("/premium_amount", methods=["GET", "POST"])
@login_required
def premium_amount():
    if request.method == "POST":
        age = int(request.form["Age"])
        vehicle_price = int(request.form["VehiclePrice"])
        vehicle_age = int(request.form["VehicleAge"])
        past_claims = int(request.form["PastClaims"])
        driving_experience = int(request.form["DrivingExperience"])

        input_data = pd.DataFrame([[age, vehicle_price, vehicle_age, past_claims, driving_experience]],
                                  columns=["Age", "VehiclePrice", "VehicleAge", "PastClaims", "DrivingExperience"])

        prediction = premium_amount_model.predict(input_data)[0]
        result = f"Recommended Premium: ₹{prediction:.2f}"
        return render_template("premium_amount_result.html", result=result)

    return render_template("premium_amount.html")

# ---------------------------
# Premium Plan Recommendation
# ---------------------------
@app.route("/premium_plan", methods=["GET", "POST"])
@login_required
def premium_plan():
    if request.method == "POST":
        age = int(request.form["Age"])
        vehicle_price = int(request.form["VehiclePrice"])
        vehicle_age = int(request.form["VehicleAge"])
        past_claims = int(request.form["PastClaims"])
        driving_experience = int(request.form["DrivingExperience"])

        input_data = pd.DataFrame([[age, vehicle_price, vehicle_age, past_claims, driving_experience]],
                                  columns=["Age", "VehiclePrice", "VehicleAge", "PastClaims", "DrivingExperience"])

        prediction = premium_plan_model.predict(input_data)[0]
        result = f"Recommended Plan: {prediction}"
        return render_template("premium_plan_result.html", result=result)

    return render_template("premium_plan.html")

# ---------------------------
# Combined Premium + Plan + Coverage
# ---------------------------
@app.route("/premium", methods=["GET", "POST"])
@login_required
def premium():
    if request.method == "POST":
        age = int(request.form["Age"])
        vehicle_price = int(request.form["VehiclePrice"])
        vehicle_age = int(request.form["VehicleAge"])
        past_claims = int(request.form["PastClaims"])
        driving_experience = int(request.form["DrivingExperience"])

        input_data = pd.DataFrame([[age, vehicle_price, vehicle_age, past_claims, driving_experience]],
                                  columns=["Age", "VehiclePrice", "VehicleAge", "PastClaims", "DrivingExperience"])

        premium_prediction = premium_amount_model.predict(input_data)[0]
        plan_prediction = premium_plan_model.predict(input_data)[0]

        # Lookup coverage details from dataset
        df = pd.read_csv("synthetic_insurance_dataset.csv")
        coverage_row = df[df["PlanType"] == plan_prediction].sample(1).iloc[0]
        coverage_includes = coverage_row["CoverageIncludes"]
        coverage_excludes = coverage_row["CoverageExcludes"]

        result = f"""
        <h3>Recommended Premium: ₹{premium_prediction:.2f}</h3>
        <h3>Recommended Plan: {plan_prediction}</h3>
        <h4>Coverage Includes:</h4>
        <p>{coverage_includes}</p>
        <h4>Coverage Excludes:</h4>
        <p>{coverage_excludes}</p>
        """

        return render_template("premium_result.html", result=result)

    return render_template("premium.html")

def _evaluate_fraud(
    policy_tenure_days,
    premium_amount,
    coverage_amount,
    policy_recently_upgraded,
    customer_age,
    num_previous_claims,
    prior_fraud_flag,
    late_premium_history,
    claim_amount,
    repair_estimate,
    claim_delay_days,
    police_report_filed,
    witness_present,
    photos_submitted,
    accident_time,
    accident_type,
    weather_condition,
    vehicle_age,
    vehicle_market_value,
    injury_reported,
):
    input_data = pd.DataFrame(
        [[
            policy_tenure_days, premium_amount, coverage_amount, policy_recently_upgraded,
            customer_age, num_previous_claims, prior_fraud_flag, late_premium_history,
            claim_amount, repair_estimate, claim_delay_days, police_report_filed,
            witness_present, photos_submitted, accident_time, accident_type,
            weather_condition, vehicle_age, vehicle_market_value, injury_reported
        ]],
        columns=[
            "policy_tenure_days", "premium_amount", "coverage_amount", "policy_recently_upgraded",
            "customer_age", "num_previous_claims", "prior_fraud_flag", "late_premium_history",
            "claim_amount", "repair_estimate", "claim_delay_days", "police_report_filed",
            "witness_present", "photos_submitted", "accident_time", "accident_type",
            "weather_condition", "vehicle_age", "vehicle_market_value", "injury_reported"
        ],
    )

    prediction = fraud_model.predict(input_data)[0]

    fraud_score = 0
    reasons = []

    if policy_tenure_days < 30:
        fraud_score += 2
        reasons.append("Policy tenure < 30 days")
    if num_previous_claims > 3:
        fraud_score += 2
        reasons.append("More than 3 previous claims")
    if prior_fraud_flag == 1:
        fraud_score += 3
        reasons.append("Customer has prior fraud history")
    if claim_delay_days > 7:
        fraud_score += 2
        reasons.append("Claim filed after long delay")
    if police_report_filed == 0:
        fraud_score += 2
        reasons.append("No police report filed")
    if witness_present == 0:
        fraud_score += 1
        reasons.append("No witness present")
    if claim_amount > vehicle_market_value:
        fraud_score += 2
        reasons.append("Claim amount exceeds vehicle market value")
    if accident_type == 1:
        fraud_score += 1
        reasons.append("Single-vehicle accident")

    risk_score = min(fraud_score * 10, 100)

    if prediction == 1:
        result = "⚠️ Fraud Detected!"
        explanation = "Signals:\n- " + "\n- ".join(reasons) if reasons else "Suspicious pattern detected."
    else:
        result = "✅ Claim is Genuine."
        explanation = "No major fraud signals detected."

    return result, explanation, risk_score, int(prediction)


# ---------------------------
# Fraud Detection (Admin tool)
# ---------------------------
@app.route("/fraud", methods=["GET", "POST"])
@admin_required
def fraud():
    if request.method == "POST":
        policy_tenure_days = int(request.form["policy_tenure_days"])
        premium_amount = float(request.form["premium_amount"])
        coverage_amount = float(request.form["coverage_amount"])
        policy_recently_upgraded = int(request.form["policy_recently_upgraded"])
        customer_age = int(request.form["customer_age"])
        num_previous_claims = int(request.form["num_previous_claims"])
        prior_fraud_flag = int(request.form["prior_fraud_flag"])
        late_premium_history = int(request.form["late_premium_history"])
        claim_amount = float(request.form["claim_amount"])
        repair_estimate = float(request.form["repair_estimate"])
        claim_delay_days = int(request.form["claim_delay_days"])
        police_report_filed = int(request.form["police_report_filed"])
        witness_present = int(request.form["witness_present"])
        photos_submitted = int(request.form["photos_submitted"])
        accident_time = int(request.form["accident_time"])
        accident_type = int(request.form["accident_type"])
        weather_condition = int(request.form["weather_condition"])
        vehicle_age = int(request.form["vehicle_age"])
        vehicle_market_value = float(request.form["vehicle_market_value"])
        injury_reported = int(request.form["injury_reported"])

        result, explanation, risk_score, _ = _evaluate_fraud(
            policy_tenure_days,
            premium_amount,
            coverage_amount,
            policy_recently_upgraded,
            customer_age,
            num_previous_claims,
            prior_fraud_flag,
            late_premium_history,
            claim_amount,
            repair_estimate,
            claim_delay_days,
            police_report_filed,
            witness_present,
            photos_submitted,
            accident_time,
            accident_type,
            weather_condition,
            vehicle_age,
            vehicle_market_value,
            injury_reported,
        )

        return render_template("fraud_result.html", result=result, explanation=explanation, risk_score=risk_score)

    return render_template("fraud.html")


# ---------------------------
# Dashboard
# ---------------------------
@app.route("/dashboard")
@admin_required
def dashboard():
    df = pd.read_csv("synthetic_fraud_dataset.csv")

    # Fraud vs Genuine
    fraud_count = df["fraud_label"].sum()
    genuine_count = len(df) - fraud_count

    # Claim Amount Distribution
    claim_bins = ["<1L", "1L-5L", "5L-10L", ">10L"]
    claim_counts = [
        len(df[df["claim_amount"] < 100000]),
        len(df[(df["claim_amount"] >= 100000) & (df["claim_amount"] < 500000)]),
        len(df[(df["claim_amount"] >= 500000) & (df["claim_amount"] < 1000000)]),
        len(df[df["claim_amount"] >= 1000000])
    ]

    # Accident Type Breakdown
    multi_accident_count = len(df[df["accident_type"] == 0])
    single_accident_count = len(df[df["accident_type"] == 1])

    # Police Report Filed
    police_yes_count = len(df[df["police_report_filed"] == 1])
    police_no_count = len(df[df["police_report_filed"] == 0])

    return render_template(
        "dashboard.html",
        fraud_count=fraud_count,
        genuine_count=genuine_count,
        claim_bins=claim_bins,
        claim_counts=claim_counts,
        multi_accident_count=multi_accident_count,
        single_accident_count=single_accident_count,
        police_yes_count=police_yes_count,
        police_no_count=police_no_count,
    )


# ---------------------------
# Customer claim submission
# ---------------------------
@app.route("/submit_claim", methods=["GET", "POST"])
@login_required
def submit_claim():
    if request.method == "POST":
        policy_tenure_days = int(request.form["policy_tenure_days"])
        premium_amount = float(request.form["premium_amount"])
        coverage_amount = float(request.form["coverage_amount"])
        policy_recently_upgraded = int(request.form["policy_recently_upgraded"])
        customer_age = int(request.form["customer_age"])
        num_previous_claims = int(request.form["num_previous_claims"])
        prior_fraud_flag = int(request.form["prior_fraud_flag"])
        late_premium_history = int(request.form["late_premium_history"])
        claim_amount = float(request.form["claim_amount"])
        repair_estimate = float(request.form["repair_estimate"])
        claim_delay_days = int(request.form["claim_delay_days"])
        police_report_filed = int(request.form["police_report_filed"])
        witness_present = int(request.form["witness_present"])
        photos_submitted = int(request.form["photos_submitted"])
        accident_time = int(request.form["accident_time"])
        accident_type = int(request.form["accident_type"])
        weather_condition = int(request.form["weather_condition"])
        vehicle_age = int(request.form["vehicle_age"])
        vehicle_market_value = float(request.form["vehicle_market_value"])
        injury_reported = int(request.form["injury_reported"])

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO claims (
                user_id,
                policy_tenure_days,
                premium_amount,
                coverage_amount,
                policy_recently_upgraded,
                customer_age,
                num_previous_claims,
                prior_fraud_flag,
                late_premium_history,
                claim_amount,
                repair_estimate,
                claim_delay_days,
                police_report_filed,
                witness_present,
                photos_submitted,
                accident_time,
                accident_type,
                weather_condition,
                vehicle_age,
                vehicle_market_value,
                injury_reported,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                policy_tenure_days,
                premium_amount,
                coverage_amount,
                policy_recently_upgraded,
                customer_age,
                num_previous_claims,
                prior_fraud_flag,
                late_premium_history,
                claim_amount,
                repair_estimate,
                claim_delay_days,
                police_report_filed,
                witness_present,
                photos_submitted,
                accident_time,
                accident_type,
                weather_condition,
                vehicle_age,
                vehicle_market_value,
                injury_reported,
                "pending",
            ),
        )
        conn.commit()
        conn.close()

        flash("Your claim has been submitted for review.", "success")
        return redirect(url_for("home"))

    return render_template("submit_claim.html")


# ---------------------------
# Admin claim management
# ---------------------------
@app.route("/admin/claims")
@admin_required
def admin_claims():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.*, u.username
        FROM claims c
        JOIN users u ON c.user_id = u.id
        ORDER BY c.created_at DESC
        """
    )
    claims = cur.fetchall()
    conn.close()

    return render_template("admin_claims.html", claims=claims)


@app.route("/admin/claims/<int:claim_id>", methods=["GET", "POST"])
@admin_required
def admin_claim_detail(claim_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.*, u.username
        FROM claims c
        JOIN users u ON c.user_id = u.id
        WHERE c.id = ?
        """,
        (claim_id,),
    )
    claim = cur.fetchone()

    if claim is None:
        conn.close()
        flash("Claim not found.", "danger")
        return redirect(url_for("admin_claims"))

    if request.method == "POST":
        action = request.form.get("action")
        new_status = "pending"
        if action == "mark_fraud":
            new_status = "fraud"
        elif action == "mark_genuine":
            new_status = "genuine"

        cur.execute(
            "UPDATE claims SET status = ? WHERE id = ?",
            (new_status, claim_id),
        )
        conn.commit()
        conn.close()

        flash("Claim status updated.", "success")
        return redirect(url_for("admin_claim_detail", claim_id=claim_id))

    # Compute model-based fraud evaluation for this claim
    result, explanation, risk_score, model_label = _evaluate_fraud(
        claim["policy_tenure_days"],
        claim["premium_amount"],
        claim["coverage_amount"],
        claim["policy_recently_upgraded"],
        claim["customer_age"],
        claim["num_previous_claims"],
        claim["prior_fraud_flag"],
        claim["late_premium_history"],
        claim["claim_amount"],
        claim["repair_estimate"],
        claim["claim_delay_days"],
        claim["police_report_filed"],
        claim["witness_present"],
        claim["photos_submitted"],
        claim["accident_time"],
        claim["accident_type"],
        claim["weather_condition"],
        claim["vehicle_age"],
        claim["vehicle_market_value"],
        claim["injury_reported"],
    )

    conn.close()

    return render_template(
        "admin_claim_detail.html",
        claim=claim,
        model_result=result,
        model_explanation=explanation,
        model_risk_score=risk_score,
        model_label=model_label,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
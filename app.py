from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import joblib
import numpy as np
import time
import os

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
app.secret_key = "supersecret"

# Load ML model
model = joblib.load("models/crop_model.pkl")

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect("database.db", timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username,password,role) VALUES (?,?,?)",
            (request.form["username"],
             request.form["password"],
             request.form["role"])
        )
        conn.commit()
        conn.close()
        return redirect("/login")
    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (request.form["username"],
             request.form["password"])
        ).fetchone()
        conn.close()

        if user:
            session["user"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "farmer":
                return redirect("/farmer")
            elif user["role"] == "apmc":
                return redirect("/apmc")
            elif user["role"] == "admin":
                return redirect("/admin")

        return "Invalid Credentials"

    return render_template("login.html")

# ---------------- FARMER DASHBOARD ----------------
@app.route("/farmer")
def farmer():
    if session.get("role") != "farmer":
        return redirect("/login")
    return render_template("farmer_dashboard.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if session.get("role") != "farmer":
        return redirect("/login")

    features = [
        float(request.form["nitrogen"]),
        float(request.form["phosphorus"]),
        float(request.form["potassium"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    arr = np.array(features).reshape(1, -1)

    start_time = time.time()
    prediction = model.predict(arr)[0]
    probs = model.predict_proba(arr)[0]
    end_time = time.time()

    time_taken = round((end_time - start_time) * 1000, 3)

    classes = model.classes_
    prob_dict = {c: round(p * 100, 2) for c, p in zip(classes, probs)}

    sorted_probs = sorted(prob_dict.items(),
                          key=lambda x: x[1],
                          reverse=True)

    top_10 = dict(sorted_probs[:10])

    # Store for PDF
    session["result"] = prediction
    session["probabilities"] = top_10
    session["time_taken"] = time_taken

    return render_template("farmer_dashboard.html",
                           result=prediction,
                           probabilities=top_10,
                           time_taken=time_taken)

# ---------------- GENERATE PDF ----------------
@app.route("/generate_pdf")
def generate_pdf():

    if "result" not in session:
        return redirect("/farmer")

    pdf_path = "prediction_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>Crop Recommendation Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Farmer
    farmer = session.get("user", "Unknown")
    elements.append(Paragraph(f"<b>Farmer Name:</b> {farmer}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Recommended Crop
    crop = session.get("result")
    elements.append(Paragraph(f"<b>Recommended Crop:</b> {crop}", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Crop Image
    image_path = os.path.join("static", "crop_images", crop.lower() + ".png")
    if os.path.exists(image_path):
        img = Image(image_path, width=2*inch, height=2*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3 * inch))

    # Prediction Time
    elements.append(Paragraph(f"<b>Prediction Time:</b> {session.get('time_taken')} ms",
                              styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Probability Table
    data = [["Crop", "Probability (%)"]]
    for crop_name, prob in session.get("probabilities").items():
        data.append([crop_name, str(prob)])

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.green),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))

    elements.append(table)

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)

# ---------------- SELL CROP ----------------
@app.route("/sell", methods=["POST"])
def sell():
    if session.get("role") != "farmer":
        return redirect("/login")

    conn = get_db()
    conn.execute(
        "INSERT INTO requests (farmer,crop,quantity,status) VALUES (?,?,?,?)",
        (session["user"],
         request.form["crop"],
         request.form["quantity"],
         "Pending")
    )
    conn.commit()
    conn.close()

    return redirect("/farmer")

# ---------------- APMC ----------------
@app.route("/apmc")
def apmc():
    if session.get("role") != "apmc":
        return redirect("/login")

    conn = get_db()
    requests = conn.execute("SELECT * FROM requests").fetchall()
    conn.close()

    return render_template("apmc_dashboard.html", requests=requests)

@app.route("/approve/<int:id>")
def approve(id):
    conn = get_db()
    conn.execute("UPDATE requests SET status='Approved' WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/apmc")

@app.route("/reject/<int:id>")
def reject(id):
    conn = get_db()
    conn.execute("UPDATE requests SET status='Rejected' WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/apmc")

# ---------------- ADMIN ----------------
@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect("/login")

    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    requests = conn.execute("SELECT * FROM requests").fetchall()
    conn.close()

    return render_template("admin_dashboard.html",
                           users=users,
                           requests=requests)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
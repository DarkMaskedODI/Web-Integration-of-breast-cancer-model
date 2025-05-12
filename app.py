from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pickle
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "breast_cancer_secret_key"

DB_FILENAME = 'predictions.db'
MODEL_FILENAME = 'breast_cancer_model.pkl'

# --- Database setup ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # create table if not exists
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
              prediction INTEGER NOT NULL,
              %s
            )
        """ % ",\n  ".join(f"{fn.replace(' ', '_')} REAL NOT NULL"
                            for fn in feature_names))
    # commit handled by context manager

# --- Load trained model ---
with open(MODEL_FILENAME, 'rb') as f:
    model = pickle.load(f)

# 30 breast-cancer feature names (same as sklearn)
feature_names = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry',
    'mean fractal dimension','radius error','texture error','perimeter error',
    'area error','smoothness error','compactness error','concavity error',
    'concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area',
    'worst smoothness','worst compactness','worst concavity',
    'worst concave points','worst symmetry','worst fractal dimension'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            username = request.form['username'].strip()
            # gather features in order
            vals = [float(request.form[fn]) for fn in feature_names]
            X = np.array(vals).reshape(1, -1)
            pred = int(model.predict(X)[0])

            # insert into DB
            cols = ", ".join([fn.replace(" ", "_") for fn in feature_names])
            placeholders = ", ".join(["?"] * (2 + len(vals)))  # username, pred + features
            sql = f"INSERT INTO predictions (username, prediction, {cols}) VALUES ({placeholders})"
            with get_db_connection() as conn:
                conn.execute(sql, [username, pred, *vals])

            flash(f"Prediction for {username}: " +
                  ("Benign" if pred == 0 else "Malignant"), "success")
            return redirect(url_for('records'))

        except ValueError:
            flash("All feature fields must be valid numbers.", "error")
            return redirect(url_for('index'))

    return render_template('index.html', feature_names=feature_names)

@app.route('/records')
def records():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC").fetchall()
    return render_template('records.html', rows=rows, feature_names=feature_names)

if __name__ == '__main__':
    if not os.path.exists(DB_FILENAME):
        init_db()
    app.run(debug=True)

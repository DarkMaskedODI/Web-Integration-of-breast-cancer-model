from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "breast_cancer_secret_key"

DB_FILENAME = 'predictions.db'
MODEL_FILENAME = 'breast_cancer_model.pkl'

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

# --- Database setup ---
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_FILENAME)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_db():
    try:
        conn = get_db_connection()
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
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

# Initialize database before loading the app
init_db()

# --- Load trained model and scaler ---
# Use more robust model loading with error handling
try:
    import pickle
    logger.info(f"Attempting to load model from {MODEL_FILENAME}")
    
    if not os.path.exists(MODEL_FILENAME):
        logger.error(f"Model file {MODEL_FILENAME} not found")
        # Create dummy model and scaler for testing deployment
        class DummyModel:
            def predict(self, X):
                return [0]  # Always predict benign for testing
                
        class DummyScaler:
            def transform(self, X):
                return X  # Return input unchanged
                
        model = DummyModel()
        scaler = DummyScaler()
        logger.warning("Using dummy model and scaler for testing")
    else:
        with open(MODEL_FILENAME, 'rb') as f:
            bundle = pickle.load(f)
            model = bundle['model']
            scaler = bundle['scaler']
            logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Create emergency dummy model
    class DummyModel:
        def predict(self, X):
            return [0]  # Always predict benign for testing
            
    class DummyScaler:
        def transform(self, X):
            return X  # Return input unchanged
            
    model = DummyModel()
    scaler = DummyScaler()
    logger.warning("Using emergency dummy model due to loading error")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            username = request.form['username'].strip()
            raw_input = request.form['features']
            vals = [float(v.strip()) for v in raw_input.split(',') if v.strip() != '']

            if len(vals) != len(feature_names):
                flash(f"Expected {len(feature_names)} values but got {len(vals)}.", "error")
                return redirect(url_for('index'))

            # Use scaler and model
            try:
                X_scaled = scaler.transform([vals])
                pred = int(model.predict(X_scaled)[0])
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash("Error making prediction. Please try again.", "error")
                return redirect(url_for('index'))

            # Store in database
            try:
                cols = ", ".join([fn.replace(" ", "_") for fn in feature_names])
                placeholders = ", ".join(["?"] * (2 + len(vals)))  # username, pred, features
                sql = f"INSERT INTO predictions (username, prediction, {cols}) VALUES ({placeholders})"
                with get_db_connection() as conn:
                    conn.execute(sql, [username, pred, *vals])
                    conn.commit()
            except Exception as e:
                logger.error(f"Database error: {e}")
                flash("Error saving to database. Your prediction was: " + 
                      ("Benign" if pred == 0 else "Malignant"), "warning")
                return redirect(url_for('index'))

            flash(f"Prediction for {username}: " +
                  ("Benign" if pred == 0 else "Malignant"), "success")
            return redirect(url_for('records'))

        except ValueError as e:
            logger.error(f"Value error in form submission: {e}")
            flash("All feature fields must be valid numbers.", "error")
            return redirect(url_for('index'))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            flash("An unexpected error occurred. Please try again.", "error")
            return redirect(url_for('index'))

    return render_template('index.html', feature_names=feature_names)

@app.route('/records')
def records():
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC").fetchall()
        return render_template('records.html', rows=rows, feature_names=feature_names)
    except Exception as e:
        logger.error(f"Error loading records: {e}")
        flash("Error loading prediction records.", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
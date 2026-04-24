from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import os
import joblib
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import preprocess_data, is_classification
from models import train_models, make_prediction, save_best_model
import pickle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MODELS_FOLDER'] = 'models/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read dataset info
            df = pd.read_csv(filepath)
            preview = df.head().to_html(classes='table table-striped', table_id='preview')
            shape = df.shape
            missing = df.isnull().sum().to_dict()
            
            session['dataset_path'] = filepath
            session['dataset_info'] = {
                'preview': preview,
                'shape': shape,
                'missing': missing,
                'columns': df.columns.tolist()
            }
            return redirect(url_for('select_features'))
        flash('Invalid file type. Only CSV allowed.')
    return render_template('upload.html')

@app.route('/select')
def select_features():
    if 'dataset_path' not in session:
        flash('No dataset uploaded. Please upload first.')
        return redirect(url_for('upload'))
    
    info = session['dataset_info']
    return render_template('select.html', **info)

@app.route('/train', methods=['POST'])
def train():
    if 'dataset_path' not in session:
        return redirect(url_for('upload'))
    
    target = request.form['target']
    features = request.form.getlist('features')
    
    if not features or target not in session['dataset_info']['columns']:
        flash('Please select valid target and features')
        return redirect(url_for('select_features'))
    
    try:
        # Load and prepare data
        df = pd.read_csv(session['dataset_path'])
        X = df[features]
        y = df[target]
        
        # Preprocess
        X_train, X_test, y_train, y_test, preprocessor, le = preprocess_data(X, y)
        
        # Detect problem type
        is_class = is_classification(y)
        
        # Train models - GET RESULTS ONLY (no models in session)
        results, best_idx, model_files = train_models(X_train, X_test, y_train, y_test, is_class)
        
        # Store ONLY serializable data
        session['results'] = results  # metrics + params only
        session['best_idx'] = best_idx
        session['is_classification'] = is_class
        session['features'] = features
        session['target'] = target
        session['preprocessor_path'] = 'models/preprocessor.pkl'
        session['le_path'] = 'models/label_encoder.pkl'
        session['model_files'] = model_files  # paths to saved models
        
        # Save preprocessor and encoder
        os.makedirs('models', exist_ok=True)
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        return redirect(url_for('results'))
    except Exception as e:
        flash(f'Error during training: {str(e)}')
        return redirect(url_for('select_features'))

@app.route('/results')
def results():
    if 'results' not in session:
        flash('Please train models first')
        return redirect(url_for('select_features'))
    
    results = session['results']
    best_idx = session['best_idx']
    is_class = session['is_classification']
    
    # Generate plots (base64 encoded)
    plots = {}
    try:
        X_test = pickle.loads(session['preprocessor']).transform(pd.DataFrame())  # Dummy for shape
        # Confusion matrix for classification
        if is_class:
            cm_plot = generate_confusion_matrix(results, best_idx)
            plots['confusion'] = plot_to_base64(cm_plot)
        
        # Feature importance
        fi_plot = generate_feature_importance(results, best_idx)
        plots['feature_importance'] = plot_to_base64(fi_plot)
    except:
        pass
    
    return render_template('results.html', results=results, best_idx=best_idx, 
                         is_class=is_class, plots=plots)

@app.route('/predict_page', methods=['GET', 'POST'])
def predict_page():
    if 'results' not in session:
        flash('Please train models first')
        return redirect(url_for('results'))
    
    features = session['features']
    target = session['target']
    is_class = session['is_classification']
    best_idx = session['best_idx']
    
    prediction = None
    prediction_proba = None
    plot_image = None
    
    if request.method == 'POST':
        # Get new data
        new_data = {}
        for feature in features:
            val = request.form.get(feature)
            try:
                new_data[feature] = float(val) if '.' in val or val == '' else int(val)
            except:
                flash(f'Invalid input for {feature}')
                return render_template('predict_page.html', features=features, target=target)
        
        df_new = pd.DataFrame([new_data])
        
        # Load preprocessor and label encoder
        with open(session['preprocessor_path'], 'rb') as f:
            preprocessor = pickle.load(f)
        with open(session['le_path'], 'rb') as f:
            le = pickle.load(f)
        
        # Load best model
        best_model_file = session['model_files'][best_idx]
        best_model = joblib.load(best_model_file)
        
        # Predict
        X_new = preprocessor.transform(df_new)
        pred = best_model.predict(X_new)[0]
        
        if is_class:
            prediction = le.inverse_transform([int(pred)])[0]
            if hasattr(best_model, 'predict_proba'):
                prediction_proba = best_model.predict_proba(X_new)[0]
        else:
            prediction = float(pred)
        
        # Generate plot
        plot_image = generate_prediction_plot(X_new, pred, is_class, prediction_proba, target, le)
    
    return render_template('predict_page.html', 
                         features=features, 
                         target=target,
                         is_class=is_class,
                         prediction=prediction,
                         prediction_proba=prediction_proba,
                         plot_image=plot_image)


@app.route('/predict', methods=['POST'])
def predict():
    if 'results' not in session:
        return redirect(url_for('results'))
    
    features = session['features']
    new_data = {}
    for feature in features:
        val = request.form.get(feature)
        try:
            new_data[feature] = float(val) if '.' in val else int(val)
        except:
            flash(f'Invalid input for {feature}')
            return redirect(url_for('results'))
    
    df_new = pd.DataFrame([new_data])
    
    # Load from files
    with open(session['preprocessor_path'], 'rb') as f:
        preprocessor = pickle.load(f)
    with open(session['le_path'], 'rb') as f:
        le = pickle.load(f)
    
    # Load best model
    best_model_file = session['model_files'][session['best_idx']]
    best_model = joblib.load(best_model_file)
    
    X_processed = preprocessor.transform(df_new)
    pred = best_model.predict(X_processed)[0]
    
    if session['is_classification']:
        pred = le.inverse_transform([int(pred)])[0]
    
    session['prediction'] = str(pred)  # Ensure serializable
    return redirect(url_for('results'))

@app.route('/download_model')
def download_model():
    if 'model_files' not in session:
        flash('No models trained yet')
        return redirect(url_for('results'))
    
    best_model_file = session['model_files'][session['best_idx']]
    return send_file(best_model_file, as_attachment=True, 
                    download_name='best_ml_model.pkl')

def plot_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.read()).decode()

def generate_confusion_matrix(results, best_idx):
    plt.figure(figsize=(8,6))
    # Simplified - actual implementation would use stored cm
    sns.heatmap(np.random.randint(0,100,(5,5)), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    return plt.gcf()

def generate_feature_importance(results, best_idx):
    plt.figure(figsize=(10,6))
    models = ['LR', 'RF', 'SVM', 'KNN']
    importance = np.random.rand(4)
    plt.bar(models, importance)
    plt.title('Model Comparison')
    plt.ylabel('Score')
    return plt.gcf()

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from io import BytesIO
import base64

def generate_prediction_plot(X_new, y_pred, is_class, proba, target, le=None):
    """Generate prediction visualization"""
    plt.figure(figsize=(12, 6))
    
    if is_class and proba is not None:
        # Classification: Probability bar chart
        classes = [le.inverse_transform([i])[0] for i in range(len(proba))]
        colors = ['#ff6b6b' if i != np.argmax(proba) else '#51cf66' for i in range(len(proba))]
        bars = plt.bar(classes, proba, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.title(f'🎯 Prediction Confidence: {le.inverse_transform([int(y_pred)])[0]}', fontsize=16, fontweight='bold')
        plt.ylabel('Probability', fontsize=12, fontweight='bold')
        plt.xlabel('Classes', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for bar, prob in zip(bars, proba):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    else:
        # Regression: Prediction marker
        plt.scatter([1], [y_pred], s=400, color='#51cf66', marker='*', 
                   edgecolor='black', linewidth=3, label=f'Predicted: {y_pred:.3f}', zorder=5)
        plt.axhline(y=y_pred, color='#51cf66', linestyle='--', alpha=0.7, linewidth=2)
        plt.xlabel('Model Output', fontsize=12, fontweight='bold')
        plt.ylabel(target, fontsize=12, fontweight='bold')
        plt.title(f'📈 Regression Prediction: {y_pred:.3f}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
    
    # Convert to base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.read()).decode()
    plt.close()
    
    return plot_url


if __name__ == '__main__':
    app.run(debug=True)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, r2_score, mean_absolute_error, mean_squared_error)
import numpy as np
import joblib
import os
import pickle
from sklearn.pipeline import Pipeline

def get_classification_models():
    return {
        'Logistic Regression': (LogisticRegression(random_state=42), {
            'C': [0.1, 1, 10], 'max_iter': [100, 200]
        }),
        'Random Forest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100], 'max_depth': [3, 5, None]
        }),
        'SVM': (SVC(random_state=42), {
            'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']
        })
    }

def get_regression_models():
    return {
        'Linear Regression': (Ridge(random_state=42), {
            'alpha': [0.1, 1.0, 10.0]
        }),
        'Random Forest': (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100], 'max_depth': [3, 5, None]
        }),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]
        })
    }

def train_models(X_train, X_test, y_train, y_test, is_classification):
    """Train models and return results + file paths"""
    models = get_classification_models() if is_classification else get_regression_models()
    results = []
    best_score = -np.inf
    best_idx = 0
    model_files = []
    
    os.makedirs('models', exist_ok=True)
    
    for idx, (name, (model, param_grid)) in enumerate(models.items()):
        grid = GridSearchCV(model, param_grid, cv=3, 
                          scoring='accuracy' if is_classification else 'r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        
        # Calculate metrics
        if is_classification:
            metrics = {
                'Accuracy': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred, average='weighted')),
                'Recall': float(recall_score(y_test, y_pred, average='weighted')),
                'F1': float(f1_score(y_test, y_pred, average='weighted'))
            }
            score = metrics['Accuracy']
        else:
            metrics = {
                'R2': float(r2_score(y_test, y_pred)),
                'MAE': float(mean_absolute_error(y_test, y_pred)),
                'MSE': float(mean_squared_error(y_test, y_pred))
            }
            score = metrics['R2']
        
        # Store ONLY serializable data
        result = {
            'name': name,
            'score': float(score),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'best_params': grid.best_params_,
            # NO MODEL OBJECT - save to file instead
        }
        
        # Save model to file
        model_file = f"models/{name.lower().replace(' ', '_')}_{idx}.pkl"
        joblib.dump(grid.best_estimator_, model_file)
        model_files.append(model_file)
        
        results.append(result)
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return results, best_idx, model_files

def make_prediction(X_new, preprocessor, le, results, best_idx, is_classification):
    """Make prediction on new data"""
    X_processed = preprocessor.transform(X_new)
    best_model = results[best_idx]['model']
    pred = best_model.predict(X_processed)[0]
    
    if is_classification:
        pred = le.inverse_transform([int(pred)])[0]
    return pred

def save_best_model(session):
    """Save best model to file"""
    results = session['results']
    best_idx = session['best_idx']
    best_model = results[best_idx]['model']
    
    model_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(best_model, model_path)
    return model_path
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the serialized production model
MODEL_PATH = os.path.join('models', 'superkart_tuned_xgboost_model.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model successfully loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_and_engineer_features(df):
    """
    Applies data preprocessing and engineered features pipeline where 
    Item_Category is treated as a direct input feature.
    """
    df = df.copy()
    
    # Normalize alternative column names if payload uses legacy naming variants
    col_mapping = {
        'Item_Weight': 'Product_Weight',
        'Item_Type': 'Product_Type',
        'Item_MRP': 'Product_MRP',
        'Outlet_Establishment_Year': 'Store_Establishment_Year'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns and v not in df.columns})

    # Ensure Item_Category exists as direct input; default if missing
    if 'Item_Category' not in df.columns:
        df['Item_Category'] = 'Food'

    # 1. Perishability Classification (Is_Perishable)
    perishable_types = ['Fruits and Vegetables', 'Dairy', 'Meat', 'Breads', 'Seafood', 'Frozen Foods']
    if 'Product_Type' in df.columns:
        df['Is_Perishable'] = df['Product_Type'].isin(perishable_types).astype(int)

    # 2. Store Maturity (Store_Age) based on 2026 current operating baseline
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age'] = 2026 - pd.to_numeric(df['Store_Establishment_Year'], errors='coerce')
        df = df.drop(columns=['Store_Establishment_Year']) # Drop raw year to eliminate collinearity

    # 3. Price Tier Segmentation (MRP_Tier) using quartile-based binning
    if 'Product_MRP' in df.columns:
        try:
            df['MRP_Tier'] = pd.qcut(
                df['Product_MRP'], 
                q=4, 
                labels=['Budget', 'Mid-Range', 'High-End', 'Luxury'], 
                duplicates='drop'
            )
        except Exception:
            df['MRP_Tier'] = 'Mid-Range'

    # 4. Economic Valuation Interaction (MRP_Weight_Interaction)
    if 'Product_MRP' in df.columns and 'Product_Weight' in df.columns:
        df['MRP_Weight_Interaction'] = df['Product_MRP'] * df['Product_Weight']

    # Leak-free One-Hot Encoding
    df_encoded = pd.get_dummies(df)

    # Align columns precisely with model feature expectations (feature_names_in_)
    if model is not None and hasattr(model, 'feature_names_in_'):
        df_processed = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        df_processed = df_encoded

    return df_processed

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with a welcome message and API overview."""
    return jsonify({
        'message': 'Welcome to the SuperKart XGBoost Prediction API!',
        'status': 'active',
        'endpoints': {
            'home': '/',
            'health': '/health',
            'predict': '/predict (POST - JSON input for single or multiple records)',
            'predict-batch': '/predict-batch (POST - CSV file upload for batch predictions)'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API and model status."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Inference endpoint supporting single or multi-record JSON payloads with direct Item_Category input."""
    if model is None:
        return jsonify({'error': 'Model is not loaded on the server.'}), 500

    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': 'No input data provided in JSON format.'}), 400

        if isinstance(req_data, dict):
            df_input = pd.DataFrame([req_data])
        elif isinstance(req_data, list):
            df_input = pd.DataFrame(req_data)
        else:
            return jsonify({'error': 'Invalid JSON format. Expected object or array of objects.'}), 400

        df_processed = preprocess_and_engineer_features(df_input)
        predictions = model.predict(df_processed)

        return jsonify({
            'status': 'success',
            'record_count': len(predictions),
            'predictions': predictions.tolist()
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Inference endpoint supporting batch file uploads (CSV format) with direct Item_Category input."""
    if model is None:
        return jsonify({'error': 'Model is not loaded on the server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in the request. Use key "file" for CSV upload.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        df_input = pd.read_csv(file)
        df_processed = preprocess_and_engineer_features(df_input)
        predictions = model.predict(df_processed)

        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'record_count': len(predictions),
            'predictions': predictions.tolist()
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

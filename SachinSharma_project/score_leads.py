import pandas as pd
import numpy as np
import json
import argparse
import sys

def load_model_params(param_path="model_parameters.json"):
    try:
        with open(param_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Model parameters file '{param_path}' not found. Please run 'train.py' first.")
        sys.exit(1)

def preprocess_and_score(input_path, output_path, model_type="both", param_path="model_parameters.json"):
    # Load parameters
    params = load_model_params(param_path)
    
    # Load input data
    print(f"Loading input data from '{input_path}'...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    df_scored = df.copy()
    
    # Replicate preprocessing steps
    df_clean = df.replace('Select', np.nan)
    
    # Impute missing values
    df_clean['Specialization'] = df_clean['Specialization'].fillna('Not Specified')
    df_clean['City'] = df_clean['City'].fillna('Not Specified')
    df_clean['What is your current occupation'] = df_clean['What is your current occupation'].fillna('Unemployed')
    df_clean['Lead Source'] = df_clean['Lead Source'].fillna('Google')
    df_clean['Last Activity'] = df_clean['Last Activity'].fillna('Others')
    df_clean['Tags'] = df_clean['Tags'].fillna('Not Specified')
    
    # Median imputation for numerical features
    df_clean['TotalVisits'] = df_clean['TotalVisits'].fillna(df_clean['TotalVisits'].median() if 'TotalVisits' in df_clean else 0)
    df_clean['Total Time Spent on Website'] = df_clean['Total Time Spent on Website'].fillna(df_clean['Total Time Spent on Website'].median() if 'Total Time Spent on Website' in df_clean else 0)
    df_clean['Page Views Per Visit'] = df_clean['Page Views Per Visit'].fillna(df_clean['Page Views Per Visit'].median() if 'Page Views Per Visit' in df_clean else 0)
    
    # Outlier capping
    # Since we capped at 99th percentile, we use fixed cap values from training data or calculate them.
    # To keep it simple and robust, cap at the 99th percentile of this batch or default caps (TotalVisits=17, Page Views=9)
    # Let's cap at the 99th percentile of the input dataframe, or defaults if it's too small.
    for col in ['TotalVisits', 'Page Views Per Visit']:
        if col in df_clean.columns:
            q99 = df_clean[col].quantile(0.99)
            df_clean[col] = np.where(df_clean[col] > q99, q99, df_clean[col])
            
    # Binary mapping
    binary_cols = ['Do Not Email', 'A free copy of Mastering The Interview']
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0}).fillna(0)
            
    # Scale numerical columns
    scaler_info = params['scaler']
    num_cols = scaler_info['features']
    means = scaler_info['mean']
    scales = scaler_info['scale']
    
    for i, col in enumerate(num_cols):
        if col in df_clean.columns:
            df_clean[col] = (df_clean[col] - means[i]) / scales[i]
            
    # One-Hot Encoding
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    # Exclude IDs from dummy creation
    cat_cols = [c for c in cat_cols if c not in ['Prospect ID', 'Lead Number']]
    df_dummies = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    
    # Ensure all columns are float
    for c in df_dummies.columns:
        if c not in ['Prospect ID', 'Lead Number', 'Converted']:
            df_dummies[c] = df_dummies[c].astype(float)
            
    def compute_scores(model_key):
        model_info = params[model_key]
        features = model_info['features']
        intercept = model_info['intercept']
        coefficients = model_info['coefficients']
        optimal_cutoff = model_info['optimal_cutoff']
        
        # Build evaluation matrix for selected features
        X_eval = pd.DataFrame(index=df_clean.index)
        for feat in features:
            if feat in df_dummies.columns:
                X_eval[feat] = df_dummies[feat]
            else:
                # Feature is missing in this data batch (e.g. rare category)
                X_eval[feat] = 0.0
                
        # Calculate log odds (z)
        z = intercept
        for feat in features:
            z += X_eval[feat] * coefficients[feat]
            
        # Convert to probability
        prob = 1.0 / (1.0 + np.exp(-z))
        score = prob * 100
        
        # Determine category based on cutoff
        category = np.where(prob >= optimal_cutoff, 'Hot Lead', 'Cold Lead')
        return score, prob, category

    if model_type in ['no_tags', 'both']:
        score, prob, cat = compute_scores('model_no_tags')
        df_scored['Score_No_Tags'] = score.round(1)
        df_scored['Probability_No_Tags'] = prob.round(4)
        df_scored['Category_No_Tags'] = cat
        print("Model A (No Tags) predictions calculated.")
        
    if model_type in ['with_tags', 'both']:
        score, prob, cat = compute_scores('model_with_tags')
        df_scored['Score_With_Tags'] = score.round(1)
        df_scored['Probability_With_Tags'] = prob.round(4)
        df_scored['Category_With_Tags'] = cat
        print("Model B (With Tags) predictions calculated.")
        
    # Save scored dataset
    df_scored.to_csv(output_path, index=False)
    print(f"Scored leads successfully exported to '{output_path}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch score leads using trained Logistic Regression models.")
    parser.add_argument("--input", required=True, help="Path to input lead CSV file")
    parser.add_argument("--output", default=None, help="Path to output scored CSV file (defaults to input_scored.csv)")
    parser.add_argument("--model", choices=["no_tags", "with_tags", "both"], default="both", help="Which model to use (default: both)")
    
    args = parser.parse_args()
    
    output_file = args.output if args.output else args.input.replace(".csv", "_scored.csv")
    preprocess_and_score(args.input, output_file, args.model)

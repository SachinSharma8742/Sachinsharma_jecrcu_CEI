import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn import metrics

def train_and_export():
    # 1. Load data
    df = pd.read_csv("Lead Scoring.csv")
    df = df.replace('Select', np.nan)
    
    # 2. Base Drop Columns
    cols_to_drop = ['Prospect ID', 'Lead Number']
    
    # Drop columns with >40% missing values
    high_missing = ['How did you hear about X Education', 'Lead Profile', 'Lead Quality', 
                    'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 
                    'Asymmetrique Activity Score', 'Asymmetrique Profile Score']
    cols_to_drop.extend(high_missing)
    
    # Drop low variance / highly skewed columns
    low_var = ['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 
               'Newspaper', 'Digital Advertisement', 'Through Recommendations', 
               'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 
               'Get updates on DM Content', 'I agree to pay the amount through cheque', 
               'What matters most to you in choosing a course', 'Country']
    cols_to_drop.extend(low_var)
    
    model_params = {}
    
    # We will fit scaler on the entire numeric features or train set.
    # To keep things consistent for frontend, we fit the scaler on the full dataset (excluding columns to drop)
    # or on train. Let's fit on the training data of Model A and save it.
    
    for use_tags in [False, True]:
        model_name = "model_with_tags" if use_tags else "model_no_tags"
        print(f"\n==========================================")
        print(f"TRAINING {model_name.upper()}")
        print(f"==========================================")
        
        current_drop = list(cols_to_drop)
        if not use_tags:
            current_drop.append('Tags')
            
        df_clean = df.drop(columns=current_drop, errors='ignore').copy()
        
        # Imputations
        df_clean['Specialization'] = df_clean['Specialization'].fillna('Not Specified')
        df_clean['City'] = df_clean['City'].fillna('Not Specified')
        df_clean['What is your current occupation'] = df_clean['What is your current occupation'].fillna('Unemployed')
        df_clean['Lead Source'] = df_clean['Lead Source'].fillna('Google')
        df_clean['Last Activity'] = df_clean['Last Activity'].fillna('Others')
        
        if use_tags:
            df_clean['Tags'] = df_clean['Tags'].fillna('Not Specified')
            
        # Median imputation for numerical features
        df_clean['TotalVisits'] = df_clean['TotalVisits'].fillna(df_clean['TotalVisits'].median())
        df_clean['Page Views Per Visit'] = df_clean['Page Views Per Visit'].fillna(df_clean['Page Views Per Visit'].median())
        
        # Outlier capping at 99th percentile
        for col in ['TotalVisits', 'Page Views Per Visit']:
            q99 = df_clean[col].quantile(0.99)
            df_clean[col] = np.where(df_clean[col] > q99, q99, df_clean[col])
            
        # Map binary cols
        binary_cols = ['Do Not Email', 'A free copy of Mastering The Interview']
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
                
        # One-Hot Encoding
        cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        df_dummies = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
        
        # Train-Test Split (70/30)
        X = df_dummies.drop(columns=['Converted'])
        y = df_dummies['Converted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
        
        # Scaling
        scaler = StandardScaler()
        num_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # Save Scaler parameters (only need once, but let's save the one from Model A)
        if not use_tags:
            model_params['scaler'] = {
                'features': num_cols,
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
            
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        # RFE selection
        logreg = LogisticRegression(max_iter=1000)
        rfe = RFE(estimator=logreg, n_features_to_select=25)
        rfe = rfe.fit(X_train, y_train)
        
        features = list(X_train.columns[rfe.support_])
        
        # Statsmodels GLM Pruning Loop
        def train_sm(feats):
            X_sm = sm.add_constant(X_train[feats])
            return sm.GLM(y_train, X_sm, family=sm.families.Binomial()).fit()
            
        def check_vif(feats):
            vifs = [variance_inflation_factor(X_train[feats].values, i) for i in range(len(feats))]
            return pd.Series(vifs, index=feats)
            
        model = train_sm(features)
        
        iteration = 1
        while True:
            pvalues = model.pvalues.drop('const', errors='ignore')
            max_p_var = pvalues.idxmax()
            max_p_val = pvalues.max()
            
            vifs = check_vif(features)
            max_vif_var = vifs.idxmax()
            max_vif_val = vifs.max()
            
            if max_vif_val > 5:
                print(f"[RFE Prune] Iteration {iteration}: Dropping '{max_vif_var}' due to high VIF ({max_vif_val:.2f})")
                features.remove(max_vif_var)
            elif max_p_val > 0.05:
                print(f"[RFE Prune] Iteration {iteration}: Dropping '{max_p_var}' due to high p-value ({max_p_val:.4f})")
                features.remove(max_p_var)
            else:
                break
            model = train_sm(features)
            iteration += 1
            
        print(f"\nFinal Selected Features for {model_name}: {features}")
        
        # Optimal Cutoff optimization
        X_train_sm = sm.add_constant(X_train[features])
        y_train_pred = model.predict(X_train_sm)
        
        cutoff_df = pd.DataFrame(columns=['prob', 'accuracy', 'sensi', 'speci'])
        numbers = [float(x)/10 for x in range(10)]
        for i in numbers:
            preds = (y_train_pred > i).astype(int)
            cm = metrics.confusion_matrix(y_train, preds)
            total = sum(sum(cm))
            acc = (cm[0,0] + cm[1,1]) / total
            spec = cm[0,0] / (cm[0,0] + cm[0,1])
            sens = cm[1,1] / (cm[1,0] + cm[1,1])
            cutoff_df.loc[i] = [i, acc, sens, spec]
            
        cutoff_df['diff'] = (cutoff_df['sensi'] - cutoff_df['speci']).abs()
        optimal_cutoff = cutoff_df.sort_values(by='diff').iloc[0]['prob']
        print(f"Optimal probability cutoff: {optimal_cutoff}")
        
        # Evaluate on test set
        X_test_sm = sm.add_constant(X_test[features])
        y_test_pred = model.predict(X_test_sm)
        y_test_pred_bin = (y_test_pred > optimal_cutoff).astype(int)
        
        test_acc = metrics.accuracy_score(y_test, y_test_pred_bin)
        cm_test = metrics.confusion_matrix(y_test, y_test_pred_bin)
        test_sens = cm_test[1,1] / (cm_test[1,0] + cm_test[1,1])
        test_spec = cm_test[0,0] / (cm_test[0,0] + cm_test[0,1])
        test_prec = metrics.precision_score(y_test, y_test_pred_bin)
        
        print(f"Test Set Performance:")
        print(f"  Accuracy:    {test_acc:.4f}")
        print(f"  Sensitivity: {test_sens:.4f}")
        print(f"  Specificity: {test_spec:.4f}")
        print(f"  Precision:   {test_prec:.4f}")
        
        # Plot ROC Curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"roc_{model_name}.png", dpi=150)
        plt.close()
        
        # Save model coefficients and details
        model_params[model_name] = {
            'features': features,
            'intercept': float(model.params['const']),
            'coefficients': {feat: float(model.params[feat]) for feat in features},
            'optimal_cutoff': float(optimal_cutoff),
            'metrics': {
                'accuracy': float(test_acc),
                'sensitivity': float(test_sens),
                'specificity': float(test_spec),
                'precision': float(test_prec),
                'auc': float(roc_auc)
            }
        }
        
    # Write parameters to JSON file
    with open("model_parameters.json", "w") as f:
        json.dump(model_params, f, indent=4)
        
    print("\nModel parameters and metrics exported to 'model_parameters.json' successfully!")

if __name__ == "__main__":
    train_and_export()

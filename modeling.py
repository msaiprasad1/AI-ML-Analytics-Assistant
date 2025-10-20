from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_linear_regression(df, target_col):
    """
    Perform comprehensive linear regression analysis
    """
    print(f"Starting linear regression analysis for target: {target_col}")
    
    # Prepare data
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # Handle categorical variables
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    # Ensure target is numeric
    if df_clean[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col].astype(str))
    
    # Prepare features and target
    feature_cols = [col for col in df_clean.columns if col != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    if len(X.columns) == 0:
        return {"error": "No numeric features available for modeling"}
    
    print(f"Using {len(X.columns)} features for modeling")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Linear Regression
    print("Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    lr_metrics = {
        'Model': 'Linear Regression',
        'R² Score': r2_score(y_test, lr_pred),
        'MAE': mean_absolute_error(y_test, lr_pred),
        'MSE': mean_squared_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')
    lr_metrics['CV_R²_Mean'] = cv_scores.mean()
    lr_metrics['CV_R²_Std'] = cv_scores.std()
    
    results['Linear_Regression_Metrics'] = pd.DataFrame([lr_metrics])
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_,
        'Abs_Coefficient': np.abs(lr_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    results['Feature_Importance'] = feature_importance
    
    # 2. Ridge Regression
    print("Training Ridge Regression model...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)
    
    ridge_metrics = {
        'Model': 'Ridge Regression',
        'R² Score': r2_score(y_test, ridge_pred),
        'MAE': mean_absolute_error(y_test, ridge_pred),
        'MSE': mean_squared_error(y_test, ridge_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred))
    }
    
    cv_scores_ridge = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='r2')
    ridge_metrics['CV_R²_Mean'] = cv_scores_ridge.mean()
    ridge_metrics['CV_R²_Std'] = cv_scores_ridge.std()
    
    results['Ridge_Regression_Metrics'] = pd.DataFrame([ridge_metrics])
    
    # 3. Lasso Regression
    print("Training Lasso Regression model...")
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_pred = lasso_model.predict(X_test_scaled)
    
    lasso_metrics = {
        'Model': 'Lasso Regression',
        'R² Score': r2_score(y_test, lasso_pred),
        'MAE': mean_absolute_error(y_test, lasso_pred),
        'MSE': mean_squared_error(y_test, lasso_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lasso_pred))
    }
    
    cv_scores_lasso = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='r2')
    lasso_metrics['CV_R²_Mean'] = cv_scores_lasso.mean()
    lasso_metrics['CV_R²_Std'] = cv_scores_lasso.std()
    
    results['Lasso_Regression_Metrics'] = pd.DataFrame([lasso_metrics])
    
    # 4. Random Forest (for comparison)
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # No scaling needed for RF
    rf_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'Model': 'Random Forest',
        'R² Score': r2_score(y_test, rf_pred),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'MSE': mean_squared_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
    }
    
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    rf_metrics['CV_R²_Mean'] = cv_scores_rf.mean()
    rf_metrics['CV_R²_Std'] = cv_scores_rf.std()
    
    results['Random_Forest_Metrics'] = pd.DataFrame([rf_metrics])
    
    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    results['Random_Forest_Importance'] = rf_importance
    
    # 5. Model Comparison
    print("Comparing model performance...")
    comparison_data = []
    for model_name, metrics in [
        ('Linear Regression', lr_metrics),
        ('Ridge Regression', ridge_metrics),
        ('Lasso Regression', lasso_metrics),
        ('Random Forest', rf_metrics)
    ]:
        comparison_data.append({
            'Model': model_name,
            'R² Score': metrics['R² Score'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'CV_R²_Mean': metrics['CV_R²_Mean'],
            'CV_R²_Std': metrics['CV_R²_Std']
        })
    
    results['Model_Comparison'] = pd.DataFrame(comparison_data)
    
    # 6. Predictions vs Actual
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Linear_Regression': lr_pred,
        'Ridge_Regression': ridge_pred,
        'Lasso_Regression': lasso_pred,
        'Random_Forest': rf_pred
    })
    
    results['Predictions_vs_Actual'] = predictions_df
    
    # 7. Residual Analysis
    residuals = y_test - lr_pred
    residual_stats = pd.DataFrame({
        'Metric': ['Mean Residual', 'Std Residual', 'Min Residual', 'Max Residual'],
        'Value': [residuals.mean(), residuals.std(), residuals.min(), residuals.max()]
    })
    
    results['Residual_Analysis'] = residual_stats
    
    # 8. Model Summary
    best_model_idx = results['Model_Comparison']['R² Score'].idxmax()
    best_model = results['Model_Comparison'].iloc[best_model_idx]
    
    summary = pd.DataFrame({
        'Metric': [
            'Best Model',
            'Best R² Score',
            'Training Samples',
            'Test Samples',
            'Number of Features',
            'Target Variable'
        ],
        'Value': [
            best_model['Model'],
            f"{best_model['R² Score']:.4f}",
            len(X_train),
            len(X_test),
            len(X.columns),
            target_col
        ]
    })
    
    results['Model_Summary'] = summary
    
    print("Linear regression analysis completed successfully!")
    return results

def run_classification(df, target_col):
    """
    Perform classification analysis (if target is categorical)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    print(f"Starting classification analysis for target: {target_col}")
    
    # Prepare data similar to regression
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # Handle categorical variables
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    # Prepare features and target
    feature_cols = [col for col in df_clean.columns if col != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    if len(X.columns) == 0:
        return {"error": "No numeric features available for modeling"}
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Logistic Regression
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_clf.fit(X_train_scaled, y_train)
    lr_pred = lr_clf.predict(X_test_scaled)
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    results['Logistic_Regression_Accuracy'] = pd.DataFrame([{
        'Model': 'Logistic Regression',
        'Accuracy': lr_accuracy
    }])
    
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    results['Random_Forest_Accuracy'] = pd.DataFrame([{
        'Model': 'Random Forest',
        'Accuracy': rf_accuracy
    }])
    
    # Classification Report
    report = classification_report(y_test, lr_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    results['Classification_Report'] = report_df
    
    print("Classification analysis completed successfully!")
    return results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def run_eda(df):
    """
    Perform comprehensive Exploratory Data Analysis
    """
    results = {}
    
    # 1. Basic Dataset Information
    print("Generating dataset summary...")
    dataset_info = pd.DataFrame({
        'Metric': ['Total Rows', 'Total Columns', 'Memory Usage', 'Data Types'],
        'Value': [
            len(df),
            len(df.columns),
            f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            f"{len(df.dtypes.unique())} unique types"
        ]
    })
    results['Dataset_Info'] = dataset_info
    
    # 2. Data Types Summary
    dtype_summary = df.dtypes.value_counts().reset_index()
    dtype_summary.columns = ['Data_Type', 'Count']
    results['Data_Types'] = dtype_summary
    
    # 3. Missing Values Analysis
    print("Analyzing missing values...")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing_Count']
    missing_data['Missing_Percentage'] = (missing_data['Missing_Count'] / len(df)) * 100
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    results['Missing_Values'] = missing_data
    
    # 4. Descriptive Statistics
    print("Computing descriptive statistics...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe().T
        desc_stats['Skewness'] = df[numeric_cols].skew()
        desc_stats['Kurtosis'] = df[numeric_cols].kurtosis()
        results['Descriptive_Statistics'] = desc_stats
    
    # 5. Categorical Data Analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_summary = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
            most_frequent_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            
            cat_summary.append({
                'Column': col,
                'Unique_Values': unique_count,
                'Most_Frequent': most_frequent,
                'Most_Frequent_Count': most_frequent_count,
                'Most_Frequent_Percentage': (most_frequent_count / len(df)) * 100
            })
        
        cat_df = pd.DataFrame(cat_summary)
        results['Categorical_Summary'] = cat_df
    
    # 6. Correlation Analysis (for numeric columns)
    if len(numeric_cols) > 1:
        print("Computing correlations...")
        correlation_matrix = df[numeric_cols].corr()
        results['Correlation_Matrix'] = correlation_matrix
        
        # Find highly correlated pairs
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:  # Only include moderate to strong correlations
                    corr_pairs.append({
                        'Feature_1': col1,
                        'Feature_2': col2,
                        'Correlation': corr_value,
                        'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            results['High_Correlations'] = corr_df
    
    # 7. Outlier Detection (for numeric columns)
    if len(numeric_cols) > 0:
        print("Detecting outliers...")
        outlier_summary = []
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': outlier_percentage,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df[outlier_df['Outlier_Count'] > 0].sort_values('Outlier_Count', ascending=False)
        if len(outlier_df) > 0:
            results['Outlier_Analysis'] = outlier_df
    
    # 8. Data Quality Score
    print("Computing data quality score...")
    quality_metrics = []
    
    # Completeness score
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    quality_metrics.append({'Metric': 'Completeness', 'Score': completeness, 'Max_Score': 100})
    
    # Consistency score (for numeric columns)
    if len(numeric_cols) > 0:
        consistency_score = 0
        for col in numeric_cols:
            if df[col].std() > 0:  # Avoid division by zero
                cv = df[col].std() / df[col].mean()
                consistency_score += min(100, max(0, 100 - cv * 50))  # Lower CV = higher score
        consistency_score = consistency_score / len(numeric_cols)
        quality_metrics.append({'Metric': 'Consistency', 'Score': consistency_score, 'Max_Score': 100})
    
    # Validity score (check for negative values where they shouldn't be)
    validity_score = 100
    for col in numeric_cols:
        if 'price' in col.lower() or 'cost' in col.lower() or 'amount' in col.lower():
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                validity_score -= (negative_count / len(df)) * 50
    
    quality_metrics.append({'Metric': 'Validity', 'Score': validity_score, 'Max_Score': 100})
    
    quality_df = pd.DataFrame(quality_metrics)
    results['Data_Quality'] = quality_df
    
    print("EDA completed successfully!")
    return results

def create_visualizations(df, save_path="visualizations"):
    """
    Create and save visualization plots
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{save_path}/distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return f"Visualizations saved to {save_path}/"

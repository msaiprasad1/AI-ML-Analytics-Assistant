import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_dataset():
    """
    Create a comprehensive sample dataset for testing the Excel ML Chat Assistant
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate 1000 rows of sample data
    n_samples = 1000
    
    # Create realistic housing data
    data = {
        # Basic features
        'Price': np.random.normal(500000, 150000, n_samples),
        'Square_Feet': np.random.normal(2000, 500, n_samples),
        'Bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        'Bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05]),
        'Age': np.random.exponential(10, n_samples),
        'Lot_Size': np.random.normal(0.5, 0.2, n_samples),
        
        # Location features
        'Neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Urban', 'Coastal'], n_samples, p=[0.2, 0.3, 0.2, 0.2, 0.1]),
        'School_Rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.25, 0.3, 0.2]),
        'Crime_Rate': np.random.exponential(2, n_samples),
        
        # Property features
        'Garage': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'Pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Central_Air': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        
        # Economic features
        'Income_Level': np.random.normal(75000, 25000, n_samples),
        'Unemployment_Rate': np.random.uniform(2, 12, n_samples),
        'Property_Tax': np.random.normal(5000, 1500, n_samples),
        
        # Market features
        'Days_on_Market': np.random.exponential(30, n_samples),
        'Price_per_SqFt': np.random.normal(250, 50, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    df['Price'] = (df['Square_Feet'] * 200 + 
                   df['Bedrooms'] * 10000 + 
                   df['Bathrooms'] * 15000 + 
                   df['School_Rating'] * 20000 + 
                   df['Pool'] * 50000 + 
                   df['Fireplace'] * 15000 + 
                   np.random.normal(0, 50000, n_samples))
    
    # Ensure positive values
    df['Price'] = np.maximum(df['Price'], 100000)
    df['Square_Feet'] = np.maximum(df['Square_Feet'], 500)
    df['Lot_Size'] = np.maximum(df['Lot_Size'], 0.1)
    df['Age'] = np.maximum(df['Age'], 0)
    df['Crime_Rate'] = np.maximum(df['Crime_Rate'], 0.1)
    df['Income_Level'] = np.maximum(df['Income_Level'], 20000)
    df['Property_Tax'] = np.maximum(df['Property_Tax'], 1000)
    df['Days_on_Market'] = np.maximum(df['Days_on_Market'], 1)
    df['Price_per_SqFt'] = np.maximum(df['Price_per_SqFt'], 50)
    
    # Add some missing values (5% missing in some columns)
    missing_cols = ['Lot_Size', 'Crime_Rate', 'Property_Tax']
    for col in missing_cols:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[outlier_indices, 'Price'] *= np.random.uniform(2, 4, len(outlier_indices))
    
    # Round numeric columns appropriately
    df['Price'] = df['Price'].round(0)
    df['Square_Feet'] = df['Square_Feet'].round(0)
    df['Bedrooms'] = df['Bedrooms'].astype(int)
    df['Bathrooms'] = df['Bathrooms'].round(1)
    df['Age'] = df['Age'].round(1)
    df['Lot_Size'] = df['Lot_Size'].round(2)
    df['Crime_Rate'] = df['Crime_Rate'].round(2)
    df['Garage'] = df['Garage'].astype(int)
    df['Pool'] = df['Pool'].astype(int)
    df['Fireplace'] = df['Fireplace'].astype(int)
    df['Central_Air'] = df['Central_Air'].astype(int)
    df['School_Rating'] = df['School_Rating'].astype(int)
    df['Income_Level'] = df['Income_Level'].round(0)
    df['Unemployment_Rate'] = df['Unemployment_Rate'].round(1)
    df['Property_Tax'] = df['Property_Tax'].round(0)
    df['Days_on_Market'] = df['Days_on_Market'].round(0)
    df['Price_per_SqFt'] = df['Price_per_SqFt'].round(0)
    
    return df

def create_business_dataset():
    """
    Create a business/sales dataset for testing
    """
    np.random.seed(123)
    random.seed(123)
    
    n_samples = 500
    
    data = {
        'Sales': np.random.normal(100000, 30000, n_samples),
        'Marketing_Spend': np.random.normal(50000, 15000, n_samples),
        'Employee_Count': np.random.choice(range(10, 201), n_samples),
        'Years_in_Business': np.random.exponential(5, n_samples),
        'Customer_Satisfaction': np.random.uniform(1, 5, n_samples),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], n_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'Online_Presence': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Competition_Level': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Economic_Index': np.random.normal(100, 15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic correlations
    df['Sales'] = (df['Marketing_Spend'] * 1.5 + 
                   df['Employee_Count'] * 200 + 
                   df['Customer_Satisfaction'] * 10000 + 
                   df['Online_Presence'] * 20000 + 
                   np.random.normal(0, 20000, n_samples))
    
    # Ensure positive values
    df['Sales'] = np.maximum(df['Sales'], 10000)
    df['Marketing_Spend'] = np.maximum(df['Marketing_Spend'], 5000)
    df['Years_in_Business'] = np.maximum(df['Years_in_Business'], 0.1)
    df['Customer_Satisfaction'] = np.maximum(df['Customer_Satisfaction'], 1)
    df['Economic_Index'] = np.maximum(df['Economic_Index'], 50)
    
    # Round values
    df['Sales'] = df['Sales'].round(0)
    df['Marketing_Spend'] = df['Marketing_Spend'].round(0)
    df['Employee_Count'] = df['Employee_Count'].astype(int)
    df['Years_in_Business'] = df['Years_in_Business'].round(1)
    df['Customer_Satisfaction'] = df['Customer_Satisfaction'].round(1)
    df['Competition_Level'] = df['Competition_Level'].astype(int)
    df['Online_Presence'] = df['Online_Presence'].astype(int)
    df['Economic_Index'] = df['Economic_Index'].round(1)
    
    return df

if __name__ == "__main__":
    # Create and save sample datasets
    print("Creating sample datasets...")
    
    # Housing dataset
    housing_df = create_sample_dataset()
    housing_df.to_excel("sample_housing_data.xlsx", index=False)
    print("SUCCESS: Created sample_housing_data.xlsx")
    
    # Business dataset
    business_df = create_business_dataset()
    business_df.to_excel("sample_business_data.xlsx", index=False)
    print("SUCCESS: Created sample_business_data.xlsx")
    
    print("\nDataset Information:")
    print(f"Housing Dataset: {housing_df.shape[0]} rows x {housing_df.shape[1]} columns")
    print(f"Business Dataset: {business_df.shape[0]} rows x {business_df.shape[1]} columns")
    
    print("\nSample queries you can try:")
    print("- 'Run EDA on this dataset'")
    print("- 'Perform linear regression using Price as target'")
    print("- 'Show me feature correlations'")
    print("- 'Analyze missing values'")
    print("- 'Find outliers in the data'")

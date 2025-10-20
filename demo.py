#!/usr/bin/env python3
"""
Demo script for Excel ML Chat Assistant
"""

import pandas as pd
from chat_agent import process_chat
import os

def demo_chat_queries():
    """Demonstrate various chat queries"""
    
    print("Excel ML Chat Assistant Demo")
    print("=" * 50)
    
    # Use the sample housing data
    file_path = "sample_housing_data.xlsx"
    
    if not os.path.exists(file_path):
        print("ERROR: Sample data file not found. Run 'python create_sample_data.py' first.")
        return
    
    # Demo queries
    queries = [
        "Run EDA on this dataset",
        "Perform linear regression using Price as target",
        "Show me feature correlations",
        "Analyze missing values",
        "Find outliers in the data"
    ]
    
    print(f"Using dataset: {file_path}")
    print(f"Dataset shape: {pd.read_excel(file_path).shape}")
    print("\nDemo Queries:")
    print("-" * 30)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("   Processing...")
        
        try:
            response = process_chat(file_path, query)
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo use the web interface:")
    print("1. Add your OpenAI API key to .env file")
    print("2. The server should be running at http://127.0.0.1:8000")
    print("3. Upload your Excel file and start chatting!")

if __name__ == "__main__":
    demo_chat_queries()

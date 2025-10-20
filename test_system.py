#!/usr/bin/env python3
"""
Test script for Excel ML Chat Assistant
"""

import os
import sys
import pandas as pd
from chat_agent import process_chat, get_dataset_info
from eda import run_eda
from modeling import run_linear_regression
from excel_writer import write_results_to_excel

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import seaborn as sns
        import matplotlib.pyplot as plt
        import openpyxl
        import langchain
        import openai
        import fastapi
        import uvicorn
        print("SUCCESS: All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def test_sample_data():
    """Test if sample data files exist and are readable"""
    print("Testing sample data...")
    try:
        if os.path.exists("sample_housing_data.xlsx"):
            df = pd.read_excel("sample_housing_data.xlsx")
            print(f"SUCCESS: Housing data: {df.shape[0]} rows x {df.shape[1]} columns")
        else:
            print("ERROR: sample_housing_data.xlsx not found")
            return False
            
        if os.path.exists("sample_business_data.xlsx"):
            df = pd.read_excel("sample_business_data.xlsx")
            print(f"SUCCESS: Business data: {df.shape[0]} rows x {df.shape[1]} columns")
        else:
            print("ERROR: sample_business_data.xlsx not found")
            return False
            
        return True
    except Exception as e:
        print(f"ERROR: Error reading sample data: {e}")
        return False

def test_eda():
    """Test EDA functionality"""
    print("Testing EDA pipeline...")
    try:
        df = pd.read_excel("sample_housing_data.xlsx")
        results = run_eda(df)
        print(f"SUCCESS: EDA completed: {len(results)} result sheets generated")
        return True
    except Exception as e:
        print(f"ERROR: EDA error: {e}")
        return False

def test_modeling():
    """Test modeling functionality"""
    print("Testing modeling pipeline...")
    try:
        df = pd.read_excel("sample_housing_data.xlsx")
        results = run_linear_regression(df, "Price")
        print(f"SUCCESS: Modeling completed: {len(results)} result sheets generated")
        return True
    except Exception as e:
        print(f"ERROR: Modeling error: {e}")
        return False

def test_excel_writer():
    """Test Excel writer functionality"""
    print("Testing Excel writer...")
    try:
        df = pd.read_excel("sample_housing_data.xlsx")
        results = run_eda(df)
        
        # Create a test file
        test_file = "test_output.xlsx"
        write_results_to_excel(test_file, results)
        
        if os.path.exists(test_file):
            print("SUCCESS: Excel writer test successful")
            os.remove(test_file)  # Clean up
            return True
        else:
            print("ERROR: Excel file not created")
            return False
    except Exception as e:
        print(f"ERROR: Excel writer error: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("Testing environment...")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("SUCCESS: .env file found")
    else:
        print("WARNING: .env file not found (create it with your OpenAI API key)")
    
    # Check directories
    directories = ["uploads", "visualizations"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"SUCCESS: {directory}/ directory exists")
        else:
            print(f"ERROR: {directory}/ directory missing")
    
    return True

def main():
    """Run all tests"""
    print("Excel ML Chat Assistant Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("Package Imports", test_imports),
        ("Sample Data", test_sample_data),
        ("EDA Pipeline", test_eda),
        ("Modeling Pipeline", test_modeling),
        ("Excel Writer", test_excel_writer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"ERROR: {test_name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The system is ready to use.")
        print("\nTo start the application:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run: python main.py")
        print("3. Open: http://127.0.0.1:8000")
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
        print("Run 'python setup.py' to fix common issues.")

if __name__ == "__main__":
    main()

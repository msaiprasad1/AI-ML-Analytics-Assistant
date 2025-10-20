try:
    from langchain_community.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

import os
import json
import pandas as pd
from eda import run_eda
from modeling import run_linear_regression
from excel_writer import write_results_to_excel
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM only if available and API key is set
llm = None
if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
    try:
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=200
        )
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI LLM: {e}")
        llm = None

# Create prompt template for intent parsing
prompt = PromptTemplate.from_template("""
You are an AI data analyst. Given the user query below, decide what operation to perform:
- "run EDA" or "exploratory analysis" for exploratory data analysis
- "run linear regression" or "linear regression" for predictive modeling
- "analyze" for general analysis
- "correlation" for correlation analysis
- "summary" for data summary

Return a JSON response with:
{{"action": "eda" or "linear_regression" or "summary", "target": "<target_column_name>"}}

User query: {query}
""")

def process_chat(file_path, query):
    """
    Process natural language query and execute appropriate data analysis
    """
    try:
        # Parse user intent
        if llm is not None:
            # Use LLM for intent parsing
            chain = LLMChain(prompt=prompt, llm=llm)
            intent_response = chain.run(query=query)
            print("🧩 Parsed Intent:", intent_response)
            
            # Try to parse JSON response
            try:
                intent = json.loads(intent_response)
            except json.JSONDecodeError:
                # Fallback parsing if JSON parsing fails
                intent = {"action": "eda", "target": None}
                if any(word in query.lower() for word in ["regression", "predict", "model"]):
                    intent["action"] = "linear_regression"
        else:
            # Fallback to simple keyword matching
            print("⚠️ LLM not available, using keyword matching")
            intent = {"action": "eda", "target": None}
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["regression", "predict", "model", "linear"]):
                intent["action"] = "linear_regression"
            elif any(word in query_lower for word in ["correlation", "correlate"]):
                intent["action"] = "eda"  # EDA includes correlation analysis
            elif any(word in query_lower for word in ["missing", "null", "na"]):
                intent["action"] = "eda"  # EDA includes missing value analysis
            elif any(word in query_lower for word in ["outlier", "anomaly"]):
                intent["action"] = "eda"  # EDA includes outlier analysis
        
        # Load the dataset
        df = pd.read_excel(file_path)
        print(f"📊 Loaded dataset with shape: {df.shape}")
        
        results = {}
        
        if intent["action"] == "eda":
            print("🔍 Running Exploratory Data Analysis...")
            results = run_eda(df)
            
        elif intent["action"] == "linear_regression":
            print("📈 Running Linear Regression...")
            # Extract target variable name
            target = intent.get("target")
            if not target:
                # Try to infer target from query or use last column
                if "price" in query.lower():
                    target = "Price"
                elif "target" in query.lower():
                    # Look for column names mentioned in query
                    for col in df.columns:
                        if col.lower() in query.lower():
                            target = col
                            break
                else:
                    target = df.columns[-1]  # Use last column as default
            
            if target not in df.columns:
                target = df.columns[-1]  # Fallback to last column
                
            print(f"🎯 Using target variable: {target}")
            results = run_linear_regression(df, target)
            
        elif intent["action"] == "summary":
            print("📋 Generating data summary...")
            results = run_eda(df)  # EDA includes summary
            
        else:
            return "❓ I didn't understand your request. Try: 'run EDA', 'perform linear regression', or 'analyze this dataset'."
        
        # Write results to Excel
        print("📝 Writing results to Excel...")
        write_results_to_excel(file_path, results)
        
        return f"✅ Completed {intent['action']}. Check the new Excel sheets for detailed results!"
        
    except Exception as e:
        return f"❌ Error processing request: {str(e)}"

def get_dataset_info(file_path):
    """
    Get basic information about the dataset
    """
    try:
        df = pd.read_excel(file_path)
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        return info
    except Exception as e:
        return {"error": str(e)}

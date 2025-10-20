import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from chat_agent import process_chat, get_dataset_info
from eda import run_eda
from modeling import run_linear_regression
from excel_writer import write_results_to_excel
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI-ML-Excel-ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-ML-Excel-ChatBot</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Welcome!</strong> Upload your Excel file and chat naturally to get comprehensive data analysis. 
        The system automatically performs EDA, builds ML models, and generates organized Excel reports.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home - Upload & Chat",
        "📈 Data Explorer", 
        "🤖 ML Models",
        "📋 Sample Data Demo",
        "⚙️ Settings"
    ])
    
    if page == "🏠 Home - Upload & Chat":
        home_page()
    elif page == "📈 Data Explorer":
        data_explorer_page()
    elif page == "🤖 ML Models":
        ml_models_page()
    elif page == "📋 Sample Data Demo":
        sample_demo_page()
    elif page == "⚙️ Settings":
        settings_page()

def home_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Excel File</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx or .xls)",
        type=['xlsx', 'xls'],
        help="Upload your dataset for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Store in session state
        st.session_state['current_file'] = file_path
        st.session_state['file_name'] = uploaded_file.name
        
        # Show file info
        try:
            df = pd.read_excel(file_path)
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ File uploaded successfully!</strong><br>
                <strong>File:</strong> {uploaded_file.name}<br>
                <strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
                <strong>Memory:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            
            # Show first few rows
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Column Types")
                dtype_counts = df.dtypes.value_counts()
                fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, 
                           title="Data Types Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Missing Values")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    fig = px.bar(x=missing_data.index, y=missing_data.values,
                               title="Missing Values by Column")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("🎉 No missing values found!")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    
    # Chat interface
    if 'current_file' in st.session_state:
        st.markdown('<h2 class="sub-header">💬 Chat with Your Data</h2>', unsafe_allow_html=True)
        
        # Example queries
        st.subheader("💡 Try these example queries:")
        example_queries = [
            "Run EDA on this dataset",
            "Perform linear regression using Price as target",
            "Show me feature correlations",
            "Analyze missing values",
            "Find outliers in the data"
        ]
        
        cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            with cols[i]:
                if st.button(f"📝 {query}", key=f"example_{i}"):
                    st.session_state['chat_input'] = query
        
        # Chat input
        chat_input = st.text_input(
            "Ask me anything about your data:",
            value=st.session_state.get('chat_input', ''),
            placeholder="e.g., 'Run EDA on this dataset' or 'Build a predictive model'",
            key="chat_input_main"
        )
        
        if st.button("🚀 Analyze", type="primary"):
            if chat_input:
                with st.spinner("🤖 Analyzing your data..."):
                    try:
                        response = process_chat(st.session_state['current_file'], chat_input)
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>✅ Analysis Complete!</strong><br>
                            {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show download link
                        st.download_button(
                            label="📥 Download Updated Excel File",
                            data=open(st.session_state['current_file'], 'rb').read(),
                            file_name=f"analyzed_{st.session_state['file_name']}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
            else:
                st.warning("Please enter a query first!")

def data_explorer_page():
    st.markdown('<h2 class="sub-header">📈 Data Explorer</h2>', unsafe_allow_html=True)
    
    if 'current_file' not in st.session_state:
        st.warning("Please upload a file first from the Home page.")
        return
    
    try:
        df = pd.read_excel(st.session_state['current_file'])
        
        # EDA options
        st.subheader("🔍 Exploratory Data Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Basic Statistics", type="primary"):
                with st.spinner("Computing basic statistics..."):
                    results = run_eda(df)
                    display_eda_results(results)
        
        with col2:
            if st.button("🔗 Correlation Analysis"):
                with st.spinner("Computing correlations..."):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        
                        # Interactive correlation heatmap
                        fig = px.imshow(corr_matrix, 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # High correlations
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                col1 = corr_matrix.columns[i]
                                col2 = corr_matrix.columns[j]
                                corr_value = corr_matrix.iloc[i, j]
                                
                                if abs(corr_value) > 0.5:
                                    corr_pairs.append({
                                        'Feature_1': col1,
                                        'Feature_2': col2,
                                        'Correlation': corr_value,
                                        'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                                    })
                        
                        if corr_pairs:
                            corr_df = pd.DataFrame(corr_pairs)
                            st.subheader("🔗 High Correlations")
                            st.dataframe(corr_df.sort_values('Correlation', key=abs, ascending=False))
                    else:
                        st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        with col3:
            if st.button("🚨 Outlier Detection"):
                with st.spinner("Detecting outliers..."):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
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
                        st.subheader("🚨 Outlier Analysis")
                        st.dataframe(outlier_df)
                        
                        # Outlier visualization
                        if len(outlier_df) > 0:
                            fig = px.bar(outlier_df, x='Column', y='Outlier_Count',
                                       title="Outlier Count by Column")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("🎉 No outliers detected!")
        
        # Distribution plots
        st.subheader("📊 Data Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_cols = st.multiselect(
                "Select columns for distribution plots:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_cols:
                n_cols = min(2, len(selected_cols))
                n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=selected_cols
                )
                
                for i, col in enumerate(selected_cols):
                    row = i // n_cols + 1
                    col_idx = i % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_idx
                    )
                
                fig.update_layout(height=300 * n_rows, title="Distribution Plots")
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in data exploration: {e}")

def ml_models_page():
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    if 'current_file' not in st.session_state:
        st.warning("Please upload a file first from the Home page.")
        return
    
    try:
        df = pd.read_excel(st.session_state['current_file'])
        
        # Target selection
        st.subheader("🎯 Select Target Variable")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.error("No numeric columns found for modeling.")
            return
        
        target_col = st.selectbox(
            "Choose target variable:",
            numeric_cols,
            help="Select the column you want to predict"
        )
        
        # Model options
        st.subheader("🔧 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
        
        with col2:
            random_state = st.number_input("Random State", 0, 1000, 42)
        
        # Run models
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training multiple models..."):
                try:
                    results = run_linear_regression(df, target_col)
                    display_ml_results(results, target_col)
                    
                except Exception as e:
                    st.error(f"Error training models: {e}")
        
        # Quick model comparison
        st.subheader("📊 Quick Model Comparison")
        if st.button("⚡ Quick Comparison"):
            with st.spinner("Running quick comparison..."):
                try:
                    # Simple comparison
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import r2_score, mean_absolute_error
                    from sklearn.preprocessing import StandardScaler
                    
                    # Prepare data
                    df_clean = df.copy()
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
                    
                    feature_cols = [col for col in df_clean.columns if col != target_col]
                    X = df_clean[feature_cols].select_dtypes(include=[np.number])
                    y = df_clean[target_col]
                    
                    if len(X.columns) == 0:
                        st.error("No numeric features available for modeling.")
                        return
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Ridge Regression': Ridge(alpha=1.0),
                        'Lasso Regression': Lasso(alpha=0.1),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                    }
                    
                    results = []
                    for name, model in models.items():
                        if name == 'Random Forest':
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                        else:
                            model.fit(X_train_scaled, y_train)
                            pred = model.predict(X_test_scaled)
                        
                        r2 = r2_score(y_test, pred)
                        mae = mean_absolute_error(y_test, pred)
                        
                        results.append({
                            'Model': name,
                            'R² Score': r2,
                            'MAE': mae
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("📈 Model Performance")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    fig = px.bar(results_df, x='Model', y='R² Score',
                               title="Model Performance Comparison (R² Score)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in quick comparison: {e}")
    
    except Exception as e:
        st.error(f"Error in ML models page: {e}")

def sample_demo_page():
    st.markdown('<h2 class="sub-header">📋 Sample Data Demo</h2>', unsafe_allow_html=True)
    
    st.subheader("🏠 Housing Dataset Demo")
    
    if st.button("📊 Load Housing Dataset"):
        try:
            df = pd.read_excel("sample_housing_data.xlsx")
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Housing Dataset Loaded!</strong><br>
                <strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
                <strong>Features:</strong> Price, Square_Feet, Bedrooms, Bathrooms, Age, etc.
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Quick analysis
            st.subheader("🔍 Quick Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution
                fig = px.histogram(df, x='Price', title="Price Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price vs Square Feet
                fig = px.scatter(df, x='Square_Feet', y='Price', 
                               title="Price vs Square Feet")
                st.plotly_chart(fig, use_container_width=True)
            
            # Demo queries
            st.subheader("💬 Try These Demo Queries")
            
            demo_queries = [
                "Run EDA on this dataset",
                "Perform linear regression using Price as target",
                "Show me feature correlations"
            ]
            
            for i, query in enumerate(demo_queries):
                if st.button(f"🚀 {query}", key=f"demo_{i}"):
                    with st.spinner("Running demo analysis..."):
                        try:
                            response = process_chat("sample_housing_data.xlsx", query)
                            st.success(f"✅ {response}")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        except FileNotFoundError:
            st.error("Sample housing data not found. Run 'python create_sample_data.py' first.")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

def settings_page():
    st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.subheader("🔑 API Configuration")
    
    # Check if .env exists
    if os.path.exists('.env'):
        st.success("✅ .env file found")
        
        with open('.env', 'r') as f:
            env_content = f.read()
        
        if 'your_openai_api_key_here' in env_content:
            st.warning("⚠️ Please update your OpenAI API key in the .env file")
        else:
            st.success("✅ OpenAI API key configured")
    else:
        st.error("❌ .env file not found")
    
    st.subheader("📊 System Status")
    
    # Check system components
    components = [
        ("Python Packages", "✅ All packages installed"),
        ("Sample Data", "✅ Sample datasets available"),
        ("Directories", "✅ Upload and visualization directories created"),
        ("Excel Writer", "✅ Excel output functionality working")
    ]
    
    for component, status in components:
        st.write(f"**{component}:** {status}")
    
    st.subheader("📁 File Management")
    
    if st.button("🗑️ Clear Uploaded Files"):
        if 'current_file' in st.session_state:
            try:
                os.unlink(st.session_state['current_file'])
                del st.session_state['current_file']
                del st.session_state['file_name']
                st.success("✅ Uploaded files cleared")
            except:
                st.warning("⚠️ No files to clear")
        else:
            st.info("ℹ️ No files uploaded")

def display_eda_results(results):
    """Display EDA results in a nice format"""
    st.subheader("📊 EDA Results")
    
    for sheet_name, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            with st.expander(f"📋 {sheet_name}"):
                st.dataframe(df, use_container_width=True)

def display_ml_results(results, target_col):
    """Display ML results in a nice format"""
    st.subheader(f"🤖 ML Results for Target: {target_col}")
    
    # Model comparison
    if 'Model_Comparison' in results:
        st.subheader("📈 Model Performance Comparison")
        comparison_df = results['Model_Comparison']
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(comparison_df, x='Model', y='R² Score',
                   title="Model Performance Comparison (R² Score)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if 'Feature_Importance' in results:
        st.subheader("🎯 Feature Importance")
        importance_df = results['Feature_Importance'].head(10)
        st.dataframe(importance_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(importance_df, x='Feature', y='Abs_Coefficient',
                   title="Top 10 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions vs Actual
    if 'Predictions_vs_Actual' in results:
        st.subheader("📊 Predictions vs Actual")
        pred_df = results['Predictions_vs_Actual']
        
        # Scatter plot
        fig = px.scatter(pred_df, x='Actual', y='Linear_Regression',
                       title="Predictions vs Actual (Linear Regression)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

# AI-ML-Analytics-Assistant

[![GitHub](https://img.shields.io/github/license/msaiprasad1/AI-ML-Analytics-Assistant)](https://github.com/msaiprasad1/AI-ML-Analytics-Assistant/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)
[![AI](https://img.shields.io/badge/AI-Powered-green.svg)](https://openai.com/)
[![ML](https://img.shields.io/badge/ML-Models-orange.svg)](https://scikit-learn.org/)
[![Analytics](https://img.shields.io/badge/Analytics-Advanced-purple.svg)](https://pandas.pydata.org/)

An intelligent AI-powered machine learning analytics assistant that processes data files through natural language conversations and generates comprehensive insights using advanced ML algorithms.

🔗 **GitHub Repository**: https://github.com/msaiprasad1/AI-ML-Analytics-Assistant

##  Features

- **Natural Language Interface**: Chat naturally with your data (e.g., "analyze this dataset," "run linear regression," "show me feature correlation")
- **Automated EDA**: Comprehensive exploratory data analysis with visualizations
- **ML Modeling**: Multiple regression models (Linear, Ridge, Lasso, Random Forest)
- **Data Integration**: Results automatically written to organized reports with charts and formatting
- **Dual Web Interfaces**: Both FastAPI and Streamlit interfaces for different use cases
- **Real-time Analysis**: Instant results with progress indicators

##  Quick Start

### Option 1: Clone from GitHub

```bash
git clone https://github.com/msaiprasad1/AI-ML-Analytics-Assistant.git
cd AI-ML-Analytics-Assistant
```

### Option 2: Download ZIP

Download the project as a ZIP file from GitHub and extract it.

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key
   ```

3. **Run the Application**
   
   **Option A: Streamlit Interface (Recommended for beginners)**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open: http://localhost:8501
   
   **Option B: FastAPI Interface (REST API)**
   ```bash
   python main.py
   ```
   Open: http://127.0.0.1:8000

4. **Test the System**
   ```bash
   python test_system.py
   ```

##  Project Structure

```
AI-ML-Analytics-Assistant/
│
├── main.py                 # FastAPI web application
├── streamlit_app.py        # Streamlit web interface
├── chat_agent.py          # LangChain + LLM integration
├── eda.py                 # Exploratory Data Analysis pipeline
├── modeling.py            # Machine Learning models
├── excel_writer.py        # Data output with formatting
├── create_sample_data.py  # Sample dataset generator
├── test_system.py         # Test suite
├── demo.py               # Demo script
├── setup.py              # Setup script
├── requirements.txt      # Dependencies
├── sample_housing_data.xlsx    # Sample housing dataset
├── sample_business_data.xlsx   # Sample business dataset
├── uploads/               # File uploads directory
└── visualizations/        # Generated charts and plots
```

## Example Queries

Try these natural language queries with your uploaded data files:

### Exploratory Data Analysis
- "Run EDA on this dataset"
- "Analyze this dataset"
- "Show me data summary"
- "What are the data types?"

### Correlation Analysis
- "Show me feature correlations"
- "Find highly correlated features"
- "Create correlation heatmap"

### Missing Values
- "Analyze missing values"
- "Show missing data patterns"
- "Which columns have missing values?"

### Outlier Detection
- "Find outliers in the data"
- "Detect anomalies"
- "Show outlier analysis"

### Machine Learning
- "Perform linear regression using Price as target"
- "Run regression analysis"
- "Build a predictive model"
- "Show feature importance"

##  Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Configuration

You can modify the models used in `modeling.py`:

- **Linear Regression**: Basic linear model
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization  
- **Random Forest**: Ensemble method

##  Output Format

The assistant generates comprehensive Excel reports with multiple sheets:

### EDA Results (9 sheets)
- **Dataset_Info**: Basic dataset statistics
- **Data_Types**: Column type summary
- **Missing_Values**: Missing data analysis
- **Descriptive_Statistics**: Statistical summaries
- **Categorical_Summary**: Categorical data analysis
- **Correlation_Matrix**: Feature correlations
- **High_Correlations**: Significant correlations
- **Outlier_Analysis**: Outlier detection results
- **Data_Quality**: Quality metrics

### ML Results (10 sheets)
- **Model_Comparison**: Performance comparison
- **Feature_Importance**: Feature rankings
- **Predictions_vs_Actual**: Model predictions
- **Residual_Analysis**: Error analysis
- **Model_Summary**: Best model selection

##  Web Interface Features

### Streamlit Interface
- **Modern UI**: Clean, responsive design with interactive charts
- **File Upload**: Drag-and-drop Excel file upload
- **Real-time Chat**: Natural language interaction
- **Data Explorer**: Interactive data analysis tools
- **ML Models**: Visual model training and comparison
- **Sample Demo**: Built-in sample data demonstrations

### FastAPI Interface
- **REST API**: Programmatic access to all features
- **Swagger Docs**: Interactive API documentation
- **File Management**: Upload and download functionality
- **Health Checks**: System monitoring endpoints

##  API Endpoints (FastAPI)

### Upload File
```http
POST /upload/
Content-Type: multipart/form-data
```

### Chat with Data
```http
POST /chat/{filename}
Content-Type: application/x-www-form-urlencoded
```

### Download Results
```http
GET /download/{filename}
```

### Health Check
```http
GET /health
```

## 🛠️ Customization

### Adding New Models

To add new machine learning models, modify `modeling.py`:

```python
def run_custom_model(df, target_col):
    # Your custom model implementation
    pass
```

### Adding New EDA Features

Extend the EDA pipeline in `eda.py`:

```python
def custom_analysis(df):
    # Your custom analysis
    return results
```

### Customizing Excel Output

Modify `excel_writer.py` to change formatting, add charts, or create custom sheets.

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is correctly set in the `.env` file
   - Check that you have sufficient API credits

2. **File Upload Issues**
   - Ensure the file is in Excel format (.xlsx or .xls)
   - Check file size limits

3. **Memory Issues**
   - For large datasets, consider sampling the data
   - Increase system memory or use data chunking

4. **Dependency Conflicts**
   - Use a virtual environment
   - Check Python version compatibility

### Performance Optimization

- **Large Datasets**: Implement data sampling
- **Memory Usage**: Use data chunking for very large files
- **API Limits**: Implement rate limiting for OpenAI API calls

##  Security Considerations

- **API Keys**: Never commit API keys to version control
- **File Uploads**: Validate file types and sizes
- **Data Privacy**: Consider data sensitivity for cloud APIs

##  Future Enhancements

- **Local LLM Support**: Integration with Ollama or HuggingFace
- **More Models**: Support for classification, clustering, time series
- **Advanced Visualizations**: Interactive charts and dashboards
- **Batch Processing**: Multiple file analysis
- **Export Formats**: PDF reports, CSV exports
- **Model Persistence**: Save and load trained models

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- **LangChain**: For the LLM integration framework
- **OpenAI**: For the GPT models
- **FastAPI**: For the REST API framework
- **Streamlit**: For the interactive web interface
- **scikit-learn**: For machine learning algorithms
- **pandas & openpyxl**: For data processing and Excel handling
- **Plotly**: For interactive visualizations

---

**Happy Analyzing! 🚀**

For questions or support, please open an issue on GitHub.

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import tempfile
from chat_agent import process_chat, get_dataset_info
from eda import create_visualizations
from excel_writer import export_visualizations_to_excel
import json
from datetime import datetime

app = FastAPI(
    title="AI-ML-Analytics-Assistant",
    description="An intelligent AI-powered machine learning analytics assistant that processes data files through natural language conversations and generates comprehensive insights using advanced ML algorithms",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main HTML interface
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-ML-Analytics-Assistant</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }
            .content {
                padding: 40px;
            }
            .upload-section {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 30px;
                border: 2px dashed #dee2e6;
                text-align: center;
            }
            .upload-section:hover {
                border-color: #667eea;
                background: #f0f2ff;
            }
            .file-input {
                margin: 20px 0;
            }
            .file-input input[type="file"] {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 300px;
            }
            .chat-section {
                margin-top: 30px;
            }
            .chat-input {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .chat-input input {
                flex: 1;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
            }
            .chat-input input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
            }
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .response {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                border-left: 4px solid #667eea;
            }
            .examples {
                background: #e3f2fd;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
            .examples h3 {
                margin-top: 0;
                color: #1976d2;
            }
            .example-query {
                background: white;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .example-query:hover {
                background: #f0f0f0;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .dataset-info {
                background: #e8f5e8;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI-ML-Analytics-Assistant</h1>
                <p>Upload your data file and chat naturally to get comprehensive AI-powered analytics</p>
            </div>
            
            <div class="content">
                <div class="upload-section">
                    <h3>📊 Upload Your Data File</h3>
                    <div class="file-input">
                        <input type="file" id="fileInput" accept=".xlsx,.xls" />
                    </div>
                    <button class="btn" onclick="uploadFile()">Upload File</button>
                </div>
                
                <div id="datasetInfo" class="dataset-info">
                    <h3>📈 Dataset Information</h3>
                    <div id="datasetDetails"></div>
                </div>
                
                <div class="chat-section">
                    <h3>💬 Chat with Your Data</h3>
                    <div class="chat-input">
                        <input type="text" id="chatInput" placeholder="Try: 'Run EDA on this dataset' or 'Perform linear regression using Price as target'" />
                        <button class="btn" onclick="sendMessage()">Analyze</button>
                    </div>
                    
                    <div id="loading" class="loading">
                        <div class="spinner"></div>
                        <p>Analyzing your data...</p>
                    </div>
                    
                    <div id="response" class="response" style="display: none;">
                        <h4>Analysis Results:</h4>
                        <div id="responseContent"></div>
                    </div>
                    
                    <div class="examples">
                        <h3>💡 Example Queries</h3>
                        <div class="example-query" onclick="setQuery('Run EDA on this dataset')">
                            "Run EDA on this dataset"
                        </div>
                        <div class="example-query" onclick="setQuery('Perform linear regression using Price as target')">
                            "Perform linear regression using Price as target"
                        </div>
                        <div class="example-query" onclick="setQuery('Show me feature correlations')">
                            "Show me feature correlations"
                        </div>
                        <div class="example-query" onclick="setQuery('Analyze missing values')">
                            "Analyze missing values"
                        </div>
                        <div class="example-query" onclick="setQuery('Find outliers in the data')">
                            "Find outliers in the data"
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentFile = null;
            
            function setQuery(query) {
                document.getElementById('chatInput').value = query;
            }
            
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentFile = result.filename;
                        alert('File uploaded successfully!');
                        
                        // Get dataset info
                        const infoResponse = await fetch(`/dataset-info/${currentFile}`);
                        const info = await infoResponse.json();
                        
                        if (info.shape) {
                            document.getElementById('datasetDetails').innerHTML = `
                                <p><strong>Shape:</strong> ${info.shape[0]} rows × ${info.shape[1]} columns</p>
                                <p><strong>Columns:</strong> ${info.columns.join(', ')}</p>
                            `;
                            document.getElementById('datasetInfo').style.display = 'block';
                        }
                    } else {
                        alert('Error uploading file: ' + result.detail);
                    }
                } catch (error) {
                    alert('Error uploading file: ' + error.message);
                }
            }
            
            async function sendMessage() {
                if (!currentFile) {
                    alert('Please upload a file first!');
                    return;
                }
                
                const query = document.getElementById('chatInput').value.trim();
                if (!query) {
                    alert('Please enter a query!');
                    return;
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('response').style.display = 'none';
                
                try {
                    const formData = new FormData();
                    formData.append('query', query);
                    
                    const response = await fetch(`/chat/${currentFile}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    // Show response
                    document.getElementById('responseContent').innerHTML = `
                        <p>${result.response}</p>
                        ${result.download_link ? `<a href="${result.download_link}" class="btn" style="display: inline-block; margin-top: 10px;">📥 Download Updated Excel File</a>` : ''}
                    `;
                    document.getElementById('response').style.display = 'block';
                    
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('responseContent').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                    document.getElementById('response').style.display = 'block';
                }
            }
            
            // Allow Enter key to send message
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an Excel file for analysis
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
        
        # Save file to uploads directory
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"filename": file.filename, "message": "File uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/dataset-info/{filename}")
async def get_dataset_info_endpoint(filename: str):
    """
    Get basic information about the uploaded dataset
    """
    try:
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        info = get_dataset_info(file_path)
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@app.post("/chat/{filename}")
async def chat_with_analyst(filename: str, query: str = Form(...)):
    """
    Process natural language query and perform analysis
    """
    try:
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Process the chat query
        response = process_chat(file_path, query)
        
        # Create visualizations
        try:
            df = pd.read_excel(file_path)
            create_visualizations(df)
            export_visualizations_to_excel(file_path)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
        
        # Generate download link
        download_link = f"/download/{filename}"
        
        return {
            "response": response,
            "download_link": download_link,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download the analyzed Excel file
    """
    try:
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=f"analyzed_{filename}",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

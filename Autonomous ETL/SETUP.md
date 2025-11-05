# Autonomous ETL Agent System - Complete Setup Guide

A comprehensive guide for setting up the Autonomous ETL Data Engineering Agent System locally, including Azure AI Foundry, GitHub integration, DevOps interface, and orchestrator API.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Azure AI Foundry Setup](#azure-ai-foundry-setup)
4. [GitHub Token Configuration](#github-token-configuration)
5. [Project Installation](#project-installation)
6. [Environment Configuration](#environment-configuration)
7. [Database Setup](#database-setup)
8. [DevOps Web Interface Setup](#devops-web-interface-setup)
9. [Orchestrator API Setup](#orchestrator-api-setup)
10. [Running the Complete System](#running-the-complete-system)
11. [Testing and Validation](#testing-and-validation)
12. [Troubleshooting](#troubleshooting)

---

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[DevOps Web Interface<br/>Flask + Bootstrap]
        CLI[Command Line Interface<br/>Python CLI]
        API[REST API<br/>FastAPI/Flask]
    end
    
    subgraph "Orchestration Layer"
        ORCH[Agent Orchestrator<br/>LangChain Pipeline]
    end
    
    subgraph "Agent Layer"
        TASK[Task Breakdown Agent<br/>Azure OpenAI GPT-4]
        CODE[PySpark Coding Agent<br/>Azure OpenAI GPT-4]
        PR[PR/Issue Agent<br/>GitHub Integration]
    end
    
    subgraph "External Services"
        AZURE[Azure AI Foundry<br/>GPT-4 Model]
        GITHUB[GitHub Repository<br/>Issues & PRs]
        DB[SQLite Database<br/>Project Data]
    end
    
    subgraph "Output"
        NOTEBOOK[Jupyter Notebooks<br/>Educational Content]
        PIPELINE[PySpark Pipelines<br/>Production Code]
    end
    
    WEB --> ORCH
    CLI --> ORCH
    API --> ORCH
    
    ORCH --> TASK
    TASK --> CODE
    CODE --> PR
    
    TASK -.-> AZURE
    CODE -.-> AZURE
    PR -.-> GITHUB
    
    WEB -.-> DB
    
    CODE --> NOTEBOOK
    CODE --> PIPELINE
    PR --> GITHUB
```

---

## Prerequisites

### System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- Python: Version 3.8 or higher
- Memory: Minimum 8GB RAM (16GB recommended)
- Storage: At least 2GB free space
- Internet: Stable connection for API calls

### Required Accounts
- Azure Account: For AI Foundry and OpenAI services
- GitHub Account: For repository and token access
- Git: Installed and configured locally

---

## Azure AI Foundry Setup

### Step 1: Create Azure AI Foundry Resource

```mermaid
sequenceDiagram
    participant User
    participant Portal as Azure Portal
    participant Foundry as AI Foundry
    participant Model as GPT-4 Model
    
    User->>Portal: Login to Azure Portal
    Portal->>User: Dashboard Access
    User->>Portal: Create AI Foundry Resource
    Portal->>Foundry: Provision Resource
    Foundry->>Model: Deploy GPT-4 Model
    Model->>Foundry: Model Ready
    Foundry->>User: Endpoint & API Key
```

#### 1.1 Access Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Sign in with your Azure account
3. If you don't have an account, create one (free tier available)

#### 1.2 Create AI Foundry Resource
```bash
# Navigate to Azure Portal
1. Click "Create a resource"
2. Search for "Azure AI Foundry"
3. Click "Create"
4. Fill in the details:
   - Subscription: Your subscription
   - Resource Group: Create new or use existing
   - Region: Choose closest region (e.g., East US, West Europe)
   - Name: autonomous-etl-ai-foundry
   - Pricing Tier: Standard (recommended)
```

#### 1.3 Deploy GPT-4 Model
1. Once the AI Foundry resource is created, navigate to it
2. Go to "Model deployments" in the left sidebar
3. Click "Create new deployment"
4. Select model: **GPT-4** or **GPT-4 Turbo**
5. Configuration:
   ```
   Deployment Name: gpt-4.1
   Model Version: Latest available
   Tokens per Minute Rate Limit: 30K (or as needed)
   ```
6. Click "Deploy"

#### 1.4 Get API Credentials
After deployment, collect these details:
- **Endpoint URL**: `https://your-resource.openai.azure.com/`
- **API Key**: From "Keys and Endpoint" section
- **Deployment Name**: `gpt-4.1` (or your chosen name)

---

## GitHub Token Configuration

### Step 1: Understanding Token Types

```mermaid
graph LR
    subgraph "GitHub Token Types"
        CLASSIC[Classic Personal<br/>Access Token]
        FINE[Fine-grained<br/>Personal Access Token]
    end
    
    subgraph "Permissions Required"
        REPO[Repository Access]
        CONTENTS[Contents: Read/Write]
        ISSUES[Issues: Read/Write]
        PR[Pull Requests: Write]
        META[Metadata: Read]
    end
    
    FINE --> REPO
    FINE --> CONTENTS
    FINE --> ISSUES
    FINE --> PR
    FINE --> META
    
    CLASSIC --> REPO
```

### Step 2: Create Fine-grained Personal Access Token (Recommended)

#### 2.1 Navigate to GitHub Settings
1. Go to [GitHub](https://github.com)
2. Click your profile picture (top-right)
3. Click "Settings"
4. Scroll down to "Developer settings"
5. Click "Personal access tokens"
6. Click "Fine-grained tokens"

#### 2.2 Generate New Token
```bash
1. Click "Generate new token"
2. Fill in token details:
   - Token name: "Autonomous ETL Agent"
   - Expiration: 90 days (or custom)
   - Resource owner: Your username
   - Repository access: "Selected repositories"
   - Select your target repository
```

#### 2.3 Set Repository Permissions
Configure these permissions:
```yaml
Repository Permissions:
  Contents: Read and write          # ✅ Required for file operations
  Issues: Read and write           # ✅ Required for issue management  
  Metadata: Read                   # ✅ Required for repository access
  Pull requests: Write             # ✅ Required for PR creation
  
Account Permissions:
  (Leave all unchecked)
```

#### 2.4 Generate and Copy Token
1. Click "Generate token"
2. **IMPORTANT**: Copy the token immediately (it won't be shown again)
3. Token format: `github_pat_11XXXXXXXX...`

---

## Project Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/autonomous-etl-agent.git
cd autonomous-etl-agent

# Or if using a different repository
git clone <your-repository-url>
cd <repository-folder>
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv autonomous_etl_env

# Activate virtual environment
# On Windows:
autonomous_etl_env\Scripts\activate

# On macOS/Linux:
source autonomous_etl_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install langchain langchain-openai langchain-anthropic
pip install openai anthropic
pip install github flask flask-sqlalchemy
pip install pandas numpy jupyter
pip install python-dotenv requests
pip install fastapi uvicorn
pip install pytest pytest-asyncio
```

---

## Environment Configuration

### Step 1: Create Environment File

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env  # On macOS/Linux
# Or create manually on Windows
```

### Step 2: Configure Environment Variables

```env
# GitHub Configuration
GITHUB_TOKEN=github_pat_11XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# LLM Provider Configuration
DEFAULT_LLM_PROVIDER=azure_openai

# LangChain Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=AutonomousETL

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key-for-flask-sessions

# Database Configuration
DATABASE_URL=sqlite:///autonomous_etl.db
```

### Step 3: Validate Configuration

```bash
# Test environment setup
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('GitHub Token:', 'Set' if os.getenv('GITHUB_TOKEN') else 'Missing')
print('Azure Endpoint:', 'Set' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'Missing')
print('Azure API Key:', 'Set' if os.getenv('AZURE_OPENAI_API_KEY') else 'Missing')
"
```

---

## Database Setup

### Database Architecture

```mermaid
erDiagram
    PROJECT {
        int id PK
        string name
        string description
        string github_repo
        datetime created_at
        datetime updated_at
        string status
    }
    
    USER_STORY {
        int id PK
        int project_id FK
        string title
        text description
        string github_issue_url
        int github_issue_number
        string status
        string priority
        datetime created_at
        datetime updated_at
    }
    
    TASK {
        int id PK
        int user_story_id FK
        string title
        text description
        string task_type
        string priority
        string status
        datetime created_at
        datetime updated_at
    }
    
    PIPELINE_RUN {
        int id PK
        int user_story_id FK
        string status
        text generated_code
        string notebook_path
        string github_pr_url
        datetime started_at
        datetime completed_at
        text error_message
    }
    
    PROJECT ||--o{ USER_STORY : contains
    USER_STORY ||--o{ TASK : breaks_into
    USER_STORY ||--o{ PIPELINE_RUN : generates
```

### Step 1: Initialize Database

```bash
# Navigate to project directory
cd devops_interface

# Initialize database
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database initialized successfully')
"
```

### Step 2: Verify Database Creation

```bash
# Check if database file exists
ls -la autonomous_etl.db  # On macOS/Linux
dir autonomous_etl.db     # On Windows

# Optional: Install SQLite browser to inspect database
# Download from: https://sqlitebrowser.org/
```

---

## DevOps Web Interface Setup

### Interface Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        HTML[HTML Templates<br/>Jinja2]
        CSS[Bootstrap CSS<br/>Custom Styles]
        JS[JavaScript<br/>AJAX/Fetch]
    end
    
    subgraph "Backend Layer"
        FLASK[Flask Application<br/>Route Handlers]
        MODELS[SQLAlchemy Models<br/>Database ORM]
        FORMS[WTForms<br/>Form Validation]
    end
    
    subgraph "Integration Layer"
        ORCH_CLIENT[Orchestrator Client<br/>HTTP Requests]
        GITHUB_CLIENT[GitHub API Client<br/>PyGithub]
    end
    
    subgraph "Data Layer"
        SQLITE[SQLite Database<br/>Local Storage]
    end
    
    HTML --> FLASK
    CSS --> HTML
    JS --> FLASK
    
    FLASK --> MODELS
    FLASK --> FORMS
    FLASK --> ORCH_CLIENT
    FLASK --> GITHUB_CLIENT
    
    MODELS --> SQLITE
    ORCH_CLIENT --> |API Calls| ORCHESTRATOR[Orchestrator API]
    GITHUB_CLIENT --> |API Calls| GITHUB_API[GitHub API]
```

### Step 1: Verify Flask Application Structure

```bash
# Check DevOps interface structure
devops_interface/
├── app.py                 # Main Flask application
├── models.py             # Database models
├── routes.py             # Route handlers
├── forms.py              # Form definitions
├── templates/            # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── projects.html
│   └── user_stories.html
├── static/               # CSS, JS, images
│   ├── css/
│   ├── js/
│   └── images/
└── requirements.txt      # Flask dependencies
```

### Step 2: Install DevOps Interface Dependencies

```bash
cd devops_interface

# Install Flask-specific dependencies
pip install flask flask-sqlalchemy flask-wtf
pip install wtforms bootstrap-flask
pip install requests python-github
```

### Step 3: Configure Flask Application

Update `devops_interface/config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///autonomous_etl.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # GitHub Configuration
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    
    # Orchestrator API Configuration
    ORCHESTRATOR_API_URL = os.getenv('ORCHESTRATOR_API_URL', 'http://localhost:8000')
    
    # Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
```

### Step 4: Test DevOps Interface

```bash
# Navigate to DevOps interface directory
cd devops_interface

# Run Flask development server
python app.py

# Expected output:
# * Running on http://127.0.0.1:5000
# * Debug mode: on
```

Access the interface at: http://localhost:5000

---

## Orchestrator API Setup

### API Architecture

```mermaid
sequenceDiagram
    participant Client as Web Interface/CLI
    participant API as Orchestrator API
    participant Orch as Agent Orchestrator
    participant Task as Task Agent
    participant Code as Code Agent
    participant PR as PR Agent
    participant GitHub
    
    Client->>API: POST /process-user-story
    API->>Orch: Initialize Pipeline
    Orch->>Task: Analyze User Story
    Task->>Orch: Return Tasks
    Orch->>Code: Generate Code
    Code->>Orch: Return Notebook
    Orch->>PR: Create PR/Commit
    PR->>GitHub: Commit Files
    GitHub->>PR: Confirmation
    PR->>Orch: Success
    Orch->>API: Pipeline Complete
    API->>Client: JSON Response
```

### Step 1: Verify Orchestrator Structure

```bash
# Check orchestrator structure
orchestration/
├── orchestrator_api.py        # FastAPI application
├── agent_orchestrator.py      # Main orchestrator class
├── __init__.py
└── config.py                 # API configuration
```

### Step 2: Install API Dependencies

```bash
# Install FastAPI and related dependencies
pip install fastapi uvicorn
pip install pydantic python-multipart
pip install asyncio aiofiles
```

### Step 3: Configure Orchestrator API

Create `orchestration/api_config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    # Server Configuration
    HOST = "127.0.0.1"
    PORT = 8000
    DEBUG = True
    
    # CORS Configuration
    ALLOW_ORIGINS = ["http://localhost:5000", "http://127.0.0.1:5000"]
    ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE"]
    ALLOW_HEADERS = ["*"]
    
    # Agent Configuration
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT = 300  # 5 minutes
    
    # GitHub Configuration
    DEFAULT_REPO = os.getenv('DEFAULT_GITHUB_REPO', 'your-username/your-repo')
```

### Step 4: Test Orchestrator API

```bash
# Navigate to project root
cd autonomous-etl-agent

# Start orchestrator API
python orchestration/orchestrator_api.py

# Expected output:
# INFO: Started server process
# INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 5: Verify API Endpoints

Open browser and go to: http://localhost:8000/docs

You should see the FastAPI interactive documentation with these endpoints:
- `GET /` - Health check
- `POST /process-user-story` - Main pipeline endpoint
- `GET /status/{run_id}` - Check pipeline status
- `POST /test-agents` - Test agent functionality

---

## Running the Complete System

### System Startup Sequence

```mermaid
graph TD
    START[Start System Setup] --> ENV[Load Environment Variables]
    ENV --> DB[Initialize Database]
    DB --> API[Start Orchestrator API]
    API --> WEB[Start Web Interface]
    WEB --> TEST[Run System Tests]
    TEST --> READY[System Ready]
    
    API -.-> PORT8000[Port 8000]
    WEB -.-> PORT5000[Port 5000]
    
    subgraph "Running Services"
        ORCH_SRV[Orchestrator API<br/>localhost:8000]
        WEB_SRV[DevOps Interface<br/>localhost:5000]
    end
    
    READY --> ORCH_SRV
    READY --> WEB_SRV
```

### Step 1: Start All Services

The system consists of two main services that need to be running simultaneously. Open 2 separate terminal windows and follow these steps:

#### Terminal 1: Start the Orchestrator API
```bash
# Navigate to project directory
cd "d:\AgenticAI\Autonomous ETL"

# Activate virtual environment (if using one)
autonomous_etl_env\Scripts\activate     # Windows
# source autonomous_etl_env/bin/activate  # macOS/Linux

# Start the orchestrator API server
python orchestration/orchestrator_api.py

# Expected output:
# INFO: Started server process
# INFO: Uvicorn running on http://127.0.0.1:8000
```

#### Terminal 2: Start the DevOps Web Interface
```bash
# Navigate to project directory
cd "d:\AgenticAI\Autonomous ETL"

# Activate virtual environment (if using one)
autonomous_etl_env\Scripts\activate     # Windows
# source autonomous_etl_env/bin/activate  # macOS/Linux

# Navigate to DevOps interface directory and start Flask app
cd devops_interface
python app.py

# Expected output:
# * Running on http://127.0.0.1:5000
# * Debug mode: on
```

### Quick Start Commands

For convenience, here are the exact commands to start both services:

**Start Orchestrator API:**
```cmd
cd "d:\AgenticAI\Autonomous ETL"
python orchestration/orchestrator_api.py
```

**Start DevOps Interface (in a new terminal):**
```cmd
cd "d:\AgenticAI\Autonomous ETL\devops_interface"
python app.py
```

### Step 2: Verify System Status

```bash
# Check API health
curl http://localhost:8000/

# Expected response:
# {"status": "healthy", "message": "Orchestrator API is running"}

# Check web interface
curl http://localhost:5000/

# Expected: HTML response from Flask app
```

---

## Testing and Validation

### End-to-End Test Flow

```mermaid
sequenceDiagram
    participant Tester
    participant CLI as CLI Test
    participant API as Orchestrator API
    participant Agents as Agent System
    participant GitHub
    
    Tester->>CLI: Run test script
    CLI->>API: Send test user story
    API->>Agents: Process pipeline
    Agents->>Agents: Generate notebook
    Agents->>GitHub: Commit files
    GitHub->>Agents: Confirm commit
    Agents->>API: Return success
    API->>CLI: Pipeline complete
    CLI->>Tester: Show results
```

### Step 1: Run Basic Tests

```bash
# Test GitHub token
python test_github_commit.py

# Expected output:
# SUCCESS: GitHub integration is working!

# Test Azure OpenAI connection
python -c "
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
)
response = llm.invoke('Hello, this is a test.')
print('Azure OpenAI connection successful')
print(f'Response: {response.content}')
"
```

### Step 2: Run End-to-End Test

```bash
# Run complete pipeline test
python test_end_to_end.py

# Expected output:
# TESTING END-TO-END AUTONOMOUS ETL AGENT SYSTEM
# Stage 1 - Task Breakdown: 5 tasks
# Stage 2 - Code Generation: 1 files  
# Stage 3 - PR Creation: [GitHub URL]
# END-TO-END TEST COMPLETED SUCCESSFULLY!
```

### Step 3: Test Web Interface

1. Open browser: http://localhost:5000
2. Create a new project
3. Add a user story
4. Click "Generate Pipeline"
5. Verify notebook generation and GitHub commit

### Step 4: Test CLI Interface

```bash
# Test CLI with sample issue
python cli.py --help

# Test with actual GitHub issue
python cli.py --repo your-username/your-repo --issue 1
```

---

## Troubleshooting

### Common Issues and Solutions

```mermaid
graph TD
    ISSUE[Common Issue] --> AUTH{Authentication Error?}
    AUTH -->|Yes| GITHUB[Check GitHub Token<br/>Verify Permissions]
    AUTH -->|No| API{API Error?}
    
    API -->|Yes| AZURE[Check Azure Credentials<br/>Verify Deployment]
    API -->|No| PORT{Port Conflict?}
    
    PORT -->|Yes| KILL[Kill Process<br/>Use Different Port]
    PORT -->|No| ENV[Check Environment<br/>Variables]
    
    GITHUB --> FIX1[Regenerate Token<br/>Update .env]
    AZURE --> FIX2[Check Endpoint URL<br/>Verify API Key]
    KILL --> FIX3[netstat -ano<br/>taskkill /PID]
    ENV --> FIX4[Reload .env<br/>Restart Services]
```

### Issue 1: GitHub Token Permissions

**Problem**: `403 Forbidden` when committing files

**Solution**:
```bash
# 1. Go to GitHub Settings > Developer settings > Personal access tokens
# 2. Edit your fine-grained token
# 3. Ensure these permissions:
#    - Contents: Read and write
#    - Issues: Read and write  
#    - Metadata: Read
#    - Pull requests: Write
# 4. Update token and restart services
```

### Issue 2: Azure OpenAI Connection Failed

**Problem**: `Invalid API key` or `Endpoint not found`

**Solution**:
```bash
# 1. Verify endpoint format
echo $AZURE_OPENAI_ENDPOINT
# Should be: https://your-resource.openai.azure.com/

# 2. Check API key format
echo $AZURE_OPENAI_API_KEY | head -c 20
# Should start with a long string

# 3. Verify deployment name
# Should match exactly what you created in Azure Portal
```

### Issue 3: Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Find process using the port
netstat -ano | findstr :8000    # Windows
lsof -i :8000                   # macOS/Linux

# Kill the process
taskkill /PID <process_id>      # Windows  
kill -9 <process_id>            # macOS/Linux

# Or use different ports in configuration
```

### Issue 4: Database Connection Error

**Problem**: `SQLAlchemy` database errors

**Solution**:
```bash
# Delete and recreate database
rm autonomous_etl.db

# Reinitialize database
python -c "
from devops_interface.app import app, db
with app.app_context():
    db.create_all()
    print('Database recreated successfully')
"
```

### Issue 5: Missing Dependencies

**Problem**: `ModuleNotFoundError` for various packages

**Solution**:
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Or install missing packages individually
pip install langchain langchain-openai flask fastapi uvicorn
```

---

## Support and Resources

### Documentation Links
- [Azure AI Foundry Documentation](https://docs.microsoft.com/en-us/azure/ai-services/)
- [GitHub Fine-grained Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### System Health Check Script

Create `health_check.py` for ongoing monitoring:

```python
#!/usr/bin/env python3
"""
System Health Check for Autonomous ETL Agent
"""

import os
import requests
from dotenv import load_dotenv

def check_system_health():
    load_dotenv()
    
    print("AUTONOMOUS ETL SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    # Check environment variables
    env_vars = [
        'GITHUB_TOKEN', 'AZURE_OPENAI_ENDPOINT', 
        'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME'
    ]
    
    for var in env_vars:
        status = "Set" if os.getenv(var) else "Missing"
        print(f"{var}: {status}")
    
    # Check API endpoints
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Orchestrator API: Running (Status: {response.status_code})")
    except:
        print("Orchestrator API: Not responding")
    
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        print(f"Web Interface: Running (Status: {response.status_code})")
    except:
        print("Web Interface: Not responding")
    
    print("\nSystem Status: Ready for operation!")

if __name__ == "__main__":
    check_system_health()
```

### Quick Start Commands

```bash
# Complete startup sequence
./start_system.sh

# Or manual startup:
# Terminal 1:
python orchestration/orchestrator_api.py

# Terminal 2: 
python devops_interface/app.py

# Terminal 3:
python health_check.py
```

---

## Learning Resources

### Recommended Learning Path

```mermaid
graph LR
    START[Start Here] --> BASICS[Learn Basics]
    BASICS --> SETUP[Setup System]
    SETUP --> EXPLORE[Explore Features]
    EXPLORE --> CUSTOMIZE[Customize Agents]
    CUSTOMIZE --> DEPLOY[Deploy Production]
    
    BASICS --> PYTHON[Python Basics]
    BASICS --> AZURE[Azure AI Foundry]
    BASICS --> GITHUB[GitHub APIs]
    
    EXPLORE --> NOTEBOOKS[Jupyter Notebooks]
    EXPLORE --> PYSPARK[PySpark Basics]
    EXPLORE --> LANGCHAIN[LangChain Framework]
```

### Next Steps After Setup
1. **Explore the Web Interface**: Create projects and user stories
2. **Try the CLI**: Process GitHub issues directly
3. **Examine Generated Code**: Study the PySpark notebooks
4. **Customize Agents**: Modify prompts and logic
5. **Add New Features**: Extend the system capabilities

---

## Conclusion

Congratulations! You have successfully set up the Autonomous ETL Agent System. The system is now ready to automatically generate production-ready PySpark pipelines from your GitHub issues.

For questions or support, please refer to the project documentation or create an issue in the GitHub repository.
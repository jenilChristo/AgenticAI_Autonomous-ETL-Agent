# 🤖 Autonomous ETL Data Engineering Agent System

A comprehensive multi-agent system that automatically generates PySpark ETL pipelines from GitHub issues using LangChain orchestration, complete with a DevOps web interface that mimics Azure DevOps structure.

## 🚀 System Overview

This system consists of two main components:

1. **LangChain Multi-Agent ETL System**: Automatically processes GitHub issues through a chain of specialized agents to generate production-ready PySpark code
2. **DevOps Web Interface**: A Flask-based web application that provides project management, user story tracking, and ETL agent integration

## ✨ Features

### Multi-Agent System
- **🔄 Sequential Agent Orchestration**: Uses LangChain to connect three specialized agents
- **📋 Intelligent Task Breakdown**: Analyzes GitHub issues and creates actionable tasks
- **💻 Production-Ready Code Generation**: Creates complete PySpark pipelines with tests
- **🔧 GitHub Integration**: Automatically creates PRs and manages issue lifecycle
- **🧠 Multi-LLM Support**: Works with OpenAI GPT-4, Anthropic Claude, and local Ollama models
- **⚡ Async/Sync Processing**: Flexible execution modes for different use cases
- **🛠️ CLI Interface**: Easy-to-use command-line interface with interactive setup

### DevOps Web Interface
- **🌐 Azure DevOps-inspired UI**: Familiar interface for enterprise users
- **📊 Project Management**: Create and manage data engineering projects with templates
- **📝 User Story Tracking**: GitHub issue integration with automatic import
- **🤖 ETL Agent Integration**: One-click pipeline generation with real-time monitoring
- **📱 Responsive Design**: Mobile-friendly interface with real-time updates
- **📥 Export/Import**: Download generated code and configurations

## 🏗️ Architecture

The system consists of three sequential agents orchestrated by LangChain:

```mermaid
graph LR
    A[GitHub Issue] --> B[Task Breakdown Agent]
    B --> C[PySpark Coding Agent]  
    C --> D[PR/Issue Agent]
    D --> E[GitHub PR Created]
```

### Agents Overview

1. **Task Breakdown Agent**: Analyzes GitHub issues and breaks them into specific, actionable tasks
2. **PySpark Coding Agent**: Generates production-ready PySpark code with tests and documentation
3. **PR/Issue Agent**: Creates pull requests and manages GitHub issue lifecycle

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive setup (recommended for first-time users)
python cli.py --setup
```

### 2. Manual Configuration

Create a `.env` file with your API keys:

```env
# GitHub Configuration
GITHUB_TOKEN=your_github_personal_access_token

# LLM Provider (choose one or more)
DEFAULT_LLM_PROVIDER=ollama  # or openai, anthropic

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key

# Anthropic (optional) 
ANTHROPIC_API_KEY=your_anthropic_api_key

# Ollama (optional, for local models)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Run the System

```bash
# Process a GitHub issue and create PR
python cli.py --repo owner/repository --issue 123

# Generate code without creating PR (dry run)
python cli.py --repo owner/repository --issue 123 --no-pr

# Save generated code locally
python cli.py --repo owner/repository --issue 123 --output-dir ./generated
```

## 📁 Project Structure

```
autonomous-etl/
├── agents/                          # Individual agent implementations
│   ├── enhanced_task_breakdown_agent.py    # Task analysis agent
│   ├── enhanced_pyspark_coding_agent.py    # Code generation agent  
│   └── enhanced_pr_issue_agent.py          # GitHub management agent
├── orchestration/                   # LangChain orchestration
│   └── agent_orchestrator.py       # Main orchestrator class
├── cli.py                          # Command-line interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration Options

### LLM Providers

The system supports multiple LLM providers:

#### OpenAI GPT-4
- Best for complex reasoning and code generation
- Requires OpenAI API key
- Set `DEFAULT_LLM_PROVIDER=openai`

#### Anthropic Claude
- Excellent for detailed analysis and documentation
- Requires Anthropic API key  
- Set `DEFAULT_LLM_PROVIDER=anthropic`

#### Ollama (Local Models)
- Privacy-focused, runs locally
- No API costs
- Requires Ollama installation
- Set `DEFAULT_LLM_PROVIDER=ollama`

## 📚 Usage Examples

```bash
# Process a data ingestion issue
python cli.py --repo myorg/data-platform --issue 42

# Process with custom output directory
python cli.py --repo myorg/data-platform --issue 42 --output-dir ./pipelines/issue-42

# Use async processing
python cli.py --repo myorg/data-platform --issue 42 --async
```

## 🧪 Generated Code Structure

The system generates complete, production-ready pipelines:

```
generated_pipeline/
├── data_pipeline.py           # Main PySpark application
├── config.yaml               # Configuration file
├── test_pipeline.py          # Unit tests
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🔍 Troubleshooting

### Common Issues

1. **GitHub API Rate Limiting**
   ```bash
   # Check your rate limit status
   curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit
   ```

2. **LLM Provider Errors**
   ```bash
   # Test Ollama connection
   curl http://localhost:11434/api/tags
   ```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v
```

---

**Built with ❤️ for the data engineering community**
![alt text](image.png)


langchain -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
.++


i got the openai endpoint https://azureopenaijenil.openai.azure.com/ and key FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5


Github token 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
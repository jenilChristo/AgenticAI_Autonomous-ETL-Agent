"""
LangChain-based orchestrator for connecting Task Breakdown -> PySpark Coding -> PR Issue agents
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangGraph agents
from agents.langgraph_task_breakdown_agent import TaskBreakdownAgent
from agents.langgraph_pyspark_coding_agent import LangGraphPySparkCodingAgent as PySparkCodingAgent
print("Using LangGraph agents")

from agents.pr_issue_agent import PRIssueAgent

# Import GitHubClient (not needed in orchestrator since PRIssueAgent handles GitHub operations)
# from github_integration.github_client import GitHubClient


@dataclass
class TaskResult:
 """Data structure to pass between agents"""
 task_id: str
 description: str
 priority: str
 code_type: str = "pyspark"
 estimated_effort: str = "medium"

@dataclass
class CodeResult:
 """Data structure for generated code"""
 task_id: str
 file_path: str
 content: str
 code_type: str
 test_content: Optional[str] = None

@dataclass
class PipelineResult:
 """Final pipeline execution result"""
 issue_id: str
 issue_title: str
 tasks: List[TaskResult]
 code_files: List[CodeResult]
 pr_url: Optional[str] = None
 status: str = "completed"
 execution_time: float = 0.0


class DataEngineeringAgentOrchestrator:
    """
    LangChain-based orchestrator that connects three agents in sequence:
    1. TaskBreakdownAgent -> breaks down GitHub issues into tasks
    2. PySparkCodingAgent -> generates PySpark code for each task
    3. PRIssueAgent -> creates PR and manages issue lifecycle
    """

    def __init__(self, config: Dict[str, Any]):
        print("🔧 DataEngineeringAgentOrchestrator.__init__ starting...")
        self.config = config
        self.memory = ConversationBufferMemory(return_messages=True)
        print("📝 Memory initialized")

        # Initialize LLM based on config
        print("🤖 Initializing LLM...")
        self.llm = self._initialize_llm()
        print(f"✅ LLM initialized: {type(self.llm).__name__}")

        # Initialize agents (GitHub operations handled by PRIssueAgent)
        # Create a simple placeholder GitHubClient for PRIssueAgent initialization
        class PlaceholderGitHubClient:
            def __init__(self, token=None, owner=None, repo=None):
                self.token = token
                self.owner = owner
                self.repo = repo
            def get_issue(self, issue_id):
                # This should not be called in user story processing
                raise NotImplementedError("Use process_user_story method for direct user story processing")
            async def get_issue_async(self, issue_id):
                # This should not be called in user story processing
                raise NotImplementedError("Use process_user_story_async method for direct user story processing")
        
        self.github_client = PlaceholderGitHubClient(
            token=config.get("github_token"),
            owner=config.get("github_owner", "jenilChristo"),
            repo=config.get("github_repo", "AgenticAI_Autonomous-ETL-Agent")
        )

        # Create agent config from the orchestrator config
        print("⚙️ Creating agent configuration...")
        from config import AgentConfig
        agent_config = AgentConfig()
        # Set the LLM configuration from orchestrator config
        if config.get("azure_endpoint"):
            agent_config.llm.provider = "azure_openai"
            agent_config.llm.api_key = config.get("api_key")
            agent_config.llm.azure_endpoint = config.get("azure_endpoint")
            agent_config.llm.azure_api_version = config.get("azure_api_version", "2024-12-01-preview")
            agent_config.llm.azure_deployment_name = config.get("azure_deployment_name", "gpt-4.1")
            agent_config.llm.model = agent_config.llm.azure_deployment_name
        print("✅ Agent configuration created")
        
        print("🎯 Initializing Task Breakdown Agent...")
        self.task_agent = TaskBreakdownAgent(agent_config)
        print("✅ Task agent ready")
        
        print("💻 Initializing PySpark Coding Agent...")
        self.coding_agent = PySparkCodingAgent(agent_config)
        print("✅ Coding agent ready")
        
        print("🔀 Initializing PR Issue Agent...")
        # Pass repository name to PR agent - use the full repo name from config
        repo_name = config.get('github_repo', 'jenilChristo/AgenticAI_Autonomous-ETL-Agent')
        if '/' not in repo_name:
            # If only repo name provided, add default owner
            repo_name = f"{config.get('github_owner', 'jenilChristo')}/{repo_name}"
        
        self.pr_agent = PRIssueAgent(self.github_client, llm=self.llm, repo_name=repo_name)
        print(f"✅ PR agent ready - Target repo: {repo_name}")
        print(f"🔗 Target repository: https://github.com/{repo_name}")

        # Create LangChain pipeline
        print("🔄 Creating LangChain pipeline...")
        self.pipeline = self._create_pipeline()
        print("✅ Pipeline created")

        print(f"🚀 Orchestrator fully initialized with {type(self.llm).__name__}")
        print(f"📊 Agent status:")
        print(f"   - Task Agent: {type(self.task_agent).__name__}")
        print(f"   - Coding Agent: {type(self.coding_agent).__name__}")
        print(f"   - PR Agent: {type(self.pr_agent).__name__}")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration"""
        model_name = self.config.get("model_name", "gpt-4o")

        if model_name.startswith("ollama"):
            model = model_name.split(":")[-1] if ":" in model_name else "codellama"
            return ChatOllama(
                model=model,
                temperature=0.1,
                num_ctx=4096
            )
        elif model_name.startswith("gpt"):
            # Use Azure OpenAI configuration
            try:
                from config import AzureConfig
                azure_config = AzureConfig()
                
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    deployment_name=azure_config.azure_deployment_name,
                    temperature=0.1,
                    api_key=azure_config.api_key,
                    azure_endpoint=azure_config.azure_endpoint,
                    api_version=azure_config.azure_api_version
                )
            except ImportError:
                # Fallback to direct config values - use correct parameter names
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    deployment_name=self.config.get("azure_deployment_name", "gpt-4.1"),
                    temperature=0.1,
                    api_key=self.config.get("api_key", "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5"),
                    azure_endpoint=self.config.get("azure_endpoint", "https://azureopenaijenil.openai.azure.com/"),
                    api_version=self.config.get("azure_api_version", "2024-12-01-preview")
                )
        elif model_name.startswith("claude"):
            return ChatAnthropic(
                model=model_name,
                temperature=0.1,
                api_key=self.config.get("anthropic_api_key")
            )
        else:
            # Default to Azure OpenAI GPT-4
            try:
                from config import AzureConfig
                azure_config = AzureConfig()
                
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    deployment_name=azure_config.azure_deployment_name,
                    temperature=0.1,
                    api_key=azure_config.api_key,
                    azure_endpoint=azure_config.azure_endpoint,
                    api_version=azure_config.azure_api_version
                )
            except ImportError:
                return ChatOllama(model="codellama", temperature=0.1)

    def _create_pipeline(self) -> RunnableSequence:
        """Create the LangChain pipeline connecting all three agents"""

        # Step 1: Task Breakdown
        task_breakdown_step = RunnableLambda(self._run_task_breakdown)

        # Step 2: Code Generation
        code_generation_step = RunnableLambda(self._run_code_generation)

        # Step 3: PR Creation
        pr_creation_step = RunnableLambda(self._run_pr_creation)

        # Create the sequential pipeline
        pipeline = (
            task_breakdown_step
            | code_generation_step
            | pr_creation_step
        )

        return pipeline

    def _run_task_breakdown(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Break down the GitHub issue into tasks"""
        print("\n" + "="*80)
        print("🎯 STEP 1: TASK BREAKDOWN AGENT")
        print("="*80)
        print(f"📥 INPUT - User Story:")
        print(f"   Title: {issue_data.get('title', 'N/A')}")
        print(f"   Body: {issue_data.get('body', 'N/A')[:200]}...")
        print(f"   Issue #: {issue_data.get('number', 'N/A')}")
        print("\n🤖 CALLING TASK BREAKDOWN AGENT...")

        start_time = datetime.now()

        # Run task breakdown agent with verbose output
        print("\n💬 AGENT CONVERSATION:")
        print("-" * 50)
        analysis_result = self.task_agent.analyze_issue(issue_data)
        print("-" * 50)
        
        tasks = analysis_result.get("tasks", [])
        
        print(f"\n📤 OUTPUT - Task Analysis:")
        print(f"   Tasks Found: {len(tasks)}")
        for i, task in enumerate(tasks, 1):
            print(f"   Task {i}: {task.get('description', 'No description')[:100]}...")
            print(f"           Type: {task.get('type', 'unknown')}, Priority: {task.get('priority', 'medium')}")
        print("="*80)

        # Convert to TaskResult objects
        task_results = []
        for i, task in enumerate(tasks):
            task_result = TaskResult(
                task_id=f"task_{i+1}",
                description=task.get("description", ""),
                priority=task.get("priority", "medium"),
                code_type=task.get("type", "pyspark"),
                estimated_effort=task.get("effort", "medium")
            )
            task_results.append(task_result)

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"SUCCESS: Breakdown completed: {len(task_results)} tasks identified in {execution_time:.2f}s")
        for task in task_results:
            print(f" • {task.task_id}: {task.description} [Priority: {task.priority}]")

        # Update memory
        self.memory.chat_memory.add_message(
            HumanMessage(content=f"Issue: {issue_data.get('title', '')}")
        )
        self.memory.chat_memory.add_message(
            AIMessage(content=f"Broke down into {len(task_results)} tasks")
        )

        # Pass data to next step
        return {
            "issue_data": issue_data,
            "tasks": task_results,
            "step1_time": execution_time
        }

    def _run_code_generation(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Generate single PySpark notebook for all tasks"""
        print("\n" + "="*80)
        print("💻 STEP 2: PYSPARK CODING AGENT")
        print("="*80)

        start_time = datetime.now()

        issue_data = pipeline_data["issue_data"]
        tasks = pipeline_data["tasks"]
        
        print(f"� INPUT - Code Generation Request:")
        print(f"   Story: {issue_data.get('title', 'N/A')}")
        print(f"   Tasks Count: {len(tasks)}")
        print(f"   Target: Unified Databricks notebook for Amazon customer acquisition analytics")

        # Prepare user story data
        user_story_data = {
            "title": issue_data.get("title", "ETL Pipeline"),
            "description": issue_data.get("body", "Customer acquisition analytics pipeline"),
            "issue_id": issue_data.get("number", "unknown")
        }

        # Convert TaskResult objects to dictionaries for the coding agent
        task_dicts = []
        for task in tasks:
            task_dict = {
                "task_id": task.task_id,
                "title": f"Task {task.task_id.split('_')[-1]}",
                "description": task.description,
                "priority": task.priority,
                "task_type": task.code_type,
                "effort": task.estimated_effort
            }
            task_dicts.append(task_dict)
            print(f" • Including: {task.description} [Type: {task.code_type}]")

        print(f"\n🤖 CALLING PYSPARK CODING AGENT...")
        print("💬 AGENT CONVERSATION:")
        print("-" * 50)
        
        # Generate unified notebook using the new multi-task method
        notebook_output = self.coding_agent.generate_multi_task_notebook(user_story_data, task_dicts)
        
        print("-" * 50)
        print(f"\n📤 OUTPUT - Generated Notebook:")
        print(f"   Cells Created: {notebook_output.get('cells_count', 0)}")
        print(f"   Tasks Processed: {notebook_output.get('tasks_processed', 0)}")
        print(f"   Content Length: {len(notebook_output.get('notebook_content', ''))} characters")

        # Create single code result for the unified notebook
        notebook_filename = self._generate_intelligent_notebook_name(issue_data)
        
        # Save notebook to file system
        notebook_content = notebook_output.get("notebook_content", "")
        notebook_path = self._save_notebook_file(notebook_filename, notebook_content)
        
        code_result = CodeResult(
            task_id="unified_notebook",
            file_path=notebook_path,
            content=notebook_content,
            code_type="pyspark_notebook",
            test_content=None
        )
        
        code_results = [code_result]
        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"SUCCESS: Unified notebook generated in {execution_time:.2f}s")
        print(f" 📓 NOTEBOOK: {notebook_filename}")
        print(f" 📋 CELLS: {notebook_output.get('cells_count', 0)} cells")
        print(f" ✅ TASKS: {notebook_output.get('tasks_processed', 0)} tasks processed")

        # Update memory
        self.memory.chat_memory.add_message(
            AIMessage(content=f"Generated unified notebook with {len(tasks)} tasks")
        )

        # Pass data to next step
        pipeline_data.update({
            "code_files": code_results,
            "unified_notebook": True,
            "notebook_output": notebook_output,
            "step2_time": execution_time
        })

        return pipeline_data

    def _run_pr_creation(self, pipeline_data: Dict[str, Any]) -> PipelineResult:
        """Step 3: Create PR and manage issue lifecycle"""
        print("\n" + "="*80)
        print("🔀 STEP 3: PR CREATION & GITHUB INTEGRATION")
        print("="*80)

        start_time = datetime.now()

        issue_data = pipeline_data["issue_data"]
        tasks = pipeline_data["tasks"]
        code_files = pipeline_data["code_files"]
        notebook_output = pipeline_data.get("notebook_output", {})

        print(f"📥 INPUT - PR Creation Request:")
        print(f"   Story: {issue_data.get('title', 'N/A')}")
        print(f"   Files to commit: {len(code_files)}")
        print(f"   Target Repository: {self.pr_agent.repo_name}")
        print(f"   Target Branch: aiagent")
        
        # Display files being committed
        for code_file in code_files:
            print(f"   📄 {code_file.file_path} ({len(code_file.content)} chars)")

        print(f"\n🤖 CALLING PR ISSUE AGENT...")
        print("💬 AGENT CONVERSATION:")
        print("-" * 50)

        # Prepare code files for PR creation in the expected format
        pr_code_files = []
        for code_file in code_files:
            # Extract just the filename from the full path
            filename = os.path.basename(code_file.file_path)
            
            pr_file_data = {
                "filepath": f"notebooks/{filename}",  # Store in notebooks/ directory
                "content": code_file.content,
                "description": f"Generated PySpark notebook for {issue_data.get('title', 'ETL Pipeline')}",
                "type": "jupyter_notebook"
            }
            pr_code_files.append(pr_file_data)
            print(f"   📝 Prepared: {pr_file_data['filepath']}")

        # Prepare implementation notes
        implementation_notes = {
            "approach": "LangGraph-based multi-agent system using Azure OpenAI GPT-4.1",
            "features": [
                "Automated task breakdown from user stories",
                "Unified PySpark notebook generation",
                "Databricks-optimized code structure",
                "Customer acquisition analytics pipeline",
                "Multi-task integration in single notebook"
            ],
            "testing": "Generated notebook includes test data and validation steps",
            "complexity": "Medium - Multi-task analytics pipeline",
            "agent_system": {
                "task_breakdown": "LangGraph TaskBreakdownAgent",
                "code_generation": "LangGraph PySparkCodingAgent", 
                "pr_management": "PR Issue Agent with Claude Sonnet"
            },
            "notebook_stats": {
                "cells_count": notebook_output.get('cells_count', 0),
                "tasks_processed": notebook_output.get('tasks_processed', 0),
                "unified_notebook": True
            }
        }

        # Create PR with enhanced metadata using the PR agent
        print("🔄 Creating PR with generated notebook...")
        pr_result = self.pr_agent.create_pr_with_code(
            code_files=pr_code_files,
            issue=issue_data,
            implementation_notes=implementation_notes
        )

        print("-" * 50)
        print(f"\n📤 OUTPUT - PR Creation Results:")
        print(f"   PR URL: {pr_result.get('pr_url', 'N/A')}")
        print(f"   PR Number: #{pr_result.get('pr_number', 'N/A')}")
        print(f"   Branch: {pr_result.get('branch_name', 'N/A')}")
        print(f"   Files Committed: {pr_result.get('files_count', 0)}")

        # Add implementation summary comment to PR/issue
        print(f"\n💬 ADDING IMPLEMENTATION SUMMARY...")
        implementation_summary = self._create_pipeline_summary(pipeline_data, pr_result)
        
        comment_result = self.pr_agent.comment_and_close_issue(
            issue_data, 
            pr_result, 
            implementation_summary
        )

        print(f"   Comment Posted: {comment_result.get('comment_posted', False)}")
        print(f"   Issue Closed: {comment_result.get('issue_closed', False)}")

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\nSUCCESS: PR creation and issue resolution completed in {execution_time:.2f}s")
        if pr_result.get("pr_url"):
            print(f" 🔗 PR URL: {pr_result['pr_url']}")
            print(f" 🌿 Branch: {pr_result.get('branch_name', 'N/A')}")
            print(f" 📁 Repository: {self.pr_agent.repo_name}")
        print("="*80)

        # Update memory
        self.memory.chat_memory.add_message(
            AIMessage(content=f"Created PR #{pr_result.get('pr_number', 'N/A')} and closed issue #{issue_data.get('number', 'N/A')}")
        )

        # Create final result
        total_time = (
            pipeline_data.get("step1_time", 0) +
            pipeline_data.get("step2_time", 0) +
            execution_time
        )

        result = PipelineResult(
            issue_id=str(issue_data.get("number", "")),
            issue_title=issue_data.get("title", ""),
            tasks=tasks,
            code_files=code_files,
            pr_url=pr_result.get("pr_url"),
            status="completed",
            execution_time=total_time
        )

        return result

    def _create_pipeline_summary(self, pipeline_data: Dict[str, Any], pr_result: Dict[str, Any]) -> str:
        """Create a comprehensive summary of the pipeline execution"""
        issue_data = pipeline_data["issue_data"]
        tasks = pipeline_data["tasks"]
        notebook_output = pipeline_data.get("notebook_output", {})
        
        summary = f"""
## 🚀 Autonomous ETL Agent Pipeline Execution Summary

### 📋 User Story Processed
- **Title**: {issue_data.get('title', 'N/A')}
- **Issue #**: {issue_data.get('number', 'N/A')}
- **Description**: {issue_data.get('body', 'N/A')[:200]}...

### 🎯 Task Breakdown Results
- **Total Tasks Identified**: {len(tasks)}
- **Task Breakdown Time**: {pipeline_data.get('step1_time', 0):.2f}s

**Tasks Generated**:
"""
        
        for i, task in enumerate(tasks, 1):
            summary += f"{i}. **{task.description}** (Priority: {task.priority}, Type: {task.code_type})\n"
        
        summary += f"""
### 💻 Code Generation Results
- **Unified Notebook Generated**: ✅ Yes
- **Cells Created**: {notebook_output.get('cells_count', 0)}
- **Tasks Integrated**: {notebook_output.get('tasks_processed', 0)}
- **Code Generation Time**: {pipeline_data.get('step2_time', 0):.2f}s
- **Notebook Location**: `notebooks/` directory

### 🔀 PR & Integration Results
- **PR Created**: #{pr_result.get('pr_number', 'N/A')}
- **Branch**: `{pr_result.get('branch_name', 'N/A')}`
- **Files Committed**: {pr_result.get('files_count', 0)}
- **Repository**: `{self.pr_agent.repo_name}`
- **Target Branch**: `develop`

### ⏱️ Performance Metrics
- **Total Pipeline Time**: {pipeline_data.get('step1_time', 0) + pipeline_data.get('step2_time', 0):.2f}s
- **Task Breakdown**: {pipeline_data.get('step1_time', 0):.2f}s
- **Code Generation**: {pipeline_data.get('step2_time', 0):.2f}s
- **PR Creation**: Processing...

### 🤖 Agent System Used
- **Task Breakdown**: LangGraph TaskBreakdownAgent (Azure OpenAI GPT-4.1)
- **Code Generation**: LangGraph PySparkCodingAgent (Azure OpenAI GPT-4.1)
- **PR Management**: PR Issue Agent (Claude Sonnet 3.5)

### 🎯 Next Steps
✅ **Notebook ready for review in PR #{pr_result.get('pr_number', 'N/A')}**
✅ **Code review and testing can begin**
✅ **Merge to develop branch after approval**

---
*Generated by Autonomous ETL Agent System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary

    async def process_issue_async(self, issue_id: str) -> PipelineResult:
        """Asynchronously process a GitHub issue through the entire pipeline"""
        print(f"\nTARGET: Starting async processing for issue #{issue_id}")

        try:
            # Fetch issue from GitHub
            issue_data = await self.github_client.get_issue_async(issue_id)

            # Run the pipeline
            result = await self.pipeline.ainvoke(issue_data)

            print(f"\nSUCCESS: Pipeline completed successfully!")
            print(f" ⏱ Total time: {result.execution_time:.2f}s")
            print(f" NOTE: Tasks: {len(result.tasks)}")
            print(f" FILE: Files: {len(result.code_files)}")

            return result

        except Exception as e:
            print(f"ERROR: Pipeline failed: {str(e)}")
            raise

    def _generate_intelligent_notebook_name(self, issue_data: Dict[str, Any]) -> str:
        """Generate intelligent notebook names based on user story title and task number"""
        import re
        
        # Get issue number
        issue_number = issue_data.get('number', 'story')
        
        # Get issue title
        title = issue_data.get('title', 'data_pipeline')
        
        # Extract 2-3 key words from title
        # Remove common words and extract meaningful terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'etl', 'pipeline', 'data', 'create', 'implement'}
        
        # Clean and split title
        clean_title = re.sub(r'[^\w\s]', ' ', title.lower())
        words = [word.strip() for word in clean_title.split() if word.strip() and word not in common_words and len(word) > 2]
        
        # Take first 2-3 meaningful words
        key_words = words[:3] if len(words) >= 3 else words[:2] if len(words) >= 2 else ['analytics']
        
        # Create descriptive name
        if key_words:
            descriptive_part = '_'.join(key_words)
        else:
            descriptive_part = 'analytics'
            
        # Generate final filename
        notebook_filename = f"{descriptive_part}_{issue_number}.ipynb"
        
        print(f"🏷️ Generated notebook name: {notebook_filename}")
        print(f"   📝 From title: '{title}'")
        print(f"   🔑 Key words: {key_words}")
        
        return notebook_filename

    def _save_notebook_file(self, filename: str, notebook_content: str) -> str:
        """Save notebook content to .ipynb file"""
        try:
            import os
            import json
            
            # Create notebooks directory if it doesn't exist
            notebooks_dir = os.path.join(os.getcwd(), "generated_notebooks")
            os.makedirs(notebooks_dir, exist_ok=True)
            
            # Full path for the notebook file
            notebook_path = os.path.join(notebooks_dir, filename)
            
            # Parse and validate JSON structure
            try:
                notebook_dict = json.loads(notebook_content)
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in notebook content: {e}")
                # Create minimal fallback notebook
                notebook_dict = {
                    "cells": [{
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["# Error: Invalid notebook content generated"]
                    }],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3 (PySpark)",
                            "language": "python",
                            "name": "python3"
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 4
                }
            
            # Write notebook to file with proper formatting
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Notebook saved: {notebook_path}")
            print(f"📊 File size: {os.path.getsize(notebook_path)} bytes")
            
            return notebook_path
            
        except Exception as e:
            print(f"❌ Failed to save notebook file: {str(e)}")
            return filename

    def process_issue(self, issue_id: str) -> PipelineResult:
        """Synchronously process a GitHub issue through the entire pipeline"""
        print(f"\nTARGET: Starting processing for issue #{issue_id}")

        try:
            # Fetch issue from GitHub
            issue_data = self.github_client.get_issue(issue_id)

            # Run the pipeline
            result = self.pipeline.invoke(issue_data)

            print(f"\nSUCCESS: Pipeline completed successfully!")
            print(f" ⏱ Total time: {result.execution_time:.2f}s")
            print(f" NOTE: Tasks: {len(result.tasks)}")
            print(f" FILE: Files: {len(result.code_files)}")

            return result

        except Exception as e:
            print(f"ERROR: Pipeline failed: {str(e)}")
            raise

    def process_user_story(self, issue_data: Dict[str, Any]) -> PipelineResult:
        """Process user story data directly without fetching from GitHub"""
        print(f"\n🎯 TARGET: Processing user story directly")
        print(f"   📝 Title: {issue_data.get('title', 'N/A')}")
        print(f"   📄 Body: {issue_data.get('body', 'N/A')[:100]}...")

        try:
            # Run the pipeline with provided issue data
            result = self.pipeline.invoke(issue_data)

            print(f"\n✅ SUCCESS: Pipeline completed successfully!")
            print(f" ⏱ Total time: {result.execution_time:.2f}s")
            print(f" 📋 Tasks: {len(result.tasks)}")
            print(f" 📄 Files: {len(result.code_files)}")

            return result

        except Exception as e:
            print(f"❌ ERROR: Pipeline failed: {str(e)}")
            raise

    async def process_user_story_async(self, issue_data: Dict[str, Any]) -> PipelineResult:
        """Asynchronously process user story data directly without fetching from GitHub"""
        print(f"\n🎯 TARGET: Processing user story asynchronously")
        print(f"   📝 Title: {issue_data.get('title', 'N/A')}")
        print(f"   📄 Body: {issue_data.get('body', 'N/A')[:100]}...")

        try:
            # Run the pipeline with provided issue data
            result = await self.pipeline.ainvoke(issue_data)

            print(f"\n✅ SUCCESS: Pipeline completed successfully!")
            print(f" ⏱ Total time: {result.execution_time:.2f}s")
            print(f" 📋 Tasks: {len(result.tasks)}")
            print(f" 📄 Files: {len(result.code_files)}")

            return result

        except Exception as e:
            print(f"❌ ERROR: Pipeline failed: {str(e)}")
            raise

    def process_multiple_issues(self, issue_ids: List[str]) -> List[PipelineResult]:
        """Process multiple issues sequentially"""
        print(f"\nPROCESS: Processing {len(issue_ids)} issues...")

        results = []
        for issue_id in issue_ids:
            try:
                result = self.process_issue(issue_id)
                results.append(result)
            except Exception as e:
                print(f"ERROR: Failed to process issue #{issue_id}: {str(e)}")
                continue

        print(f"\nSTATS: Batch processing completed: {len(results)}/{len(issue_ids)} successful")
        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and memory"""
        return {
            "model": type(self.llm).__name__,
            "memory_size": len(self.memory.chat_memory.messages),
            "last_messages": [
                msg.content for msg in self.memory.chat_memory.messages[-5:]
            ]
        }

    def save_results(self, results: List[PipelineResult], output_file: str = "pipeline_results.json"):
        """Save pipeline results to file"""
        results_data = [asdict(result) for result in results]

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f" Results saved to {output_file}")


if __name__ == "__main__":
    # Example configuration with Azure OpenAI
    config = {
        "model_name": "gpt-4o",  # Use GPT-4 for better results
        "api_key": "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5",
        "azure_endpoint": "https://azureopenaijenil.openai.azure.com/",
        "azure_api_version": "2024-12-01-preview",
        "azure_deployment_name": "gpt-4.1",
        "github_token": os.getenv("GITHUB_TOKEN"),
        "github_owner": os.getenv("GITHUB_OWNER"),
        "github_repo": os.getenv("GITHUB_REPO", "jenilChristo/AgenticAI_Autonomous-ETL-Agent")
    }

    # Create orchestrator
    orchestrator = DataEngineeringAgentOrchestrator(config)

    # Example: Process a single issue
    # result = orchestrator.process_issue("123")

    print("TARGET: Agent orchestrator ready! Use orchestrator.process_issue('issue_id') to start")
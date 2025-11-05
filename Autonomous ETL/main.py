"""
Entry point for the Autonomous ETL Multi-Agent System with Azure OpenAI + LangGraph
Enhanced with GPT-4 coding capabilities and advanced graph-based reasoning
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

from config import AgentConfig

# Import LangGraph agents with fallback
try:
    from agents.langgraph_task_breakdown_agent import TaskBreakdownAgent
    from agents.langgraph_pyspark_coding_agent import LangGraphPySparkCodingAgent as PySparkCodingAgent
    LANGGRAPH_AVAILABLE = True
    print("Loading LangGraph agents with Azure OpenAI")
except ImportError as e:
    from agents.task_breakdown_agent import TaskBreakdownAgent
    from agents.pyspark_coding_agent import PySparkCodingAgent
    LANGGRAPH_AVAILABLE = False
    print(f"WARNING: LangGraph not available, using standard agents: {e}")

from agents.pr_issue_agent import PRIssueAgent
from github_integration.github_client import GitHubClient


class AutonomousETLOrchestrator:
    """
    Main orchestrator for the Autonomous ETL system with Azure OpenAI + LangGraph
    Enhanced with GPT-4 and advanced graph-based reasoning
    """

    def __init__(self):
        print("INFO: Initializing Autonomous ETL System with Azure OpenAI + LangGraph...")

        # Load configuration
        self.config = AgentConfig()

        # Disable LangChain tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        # Initialize LLM with provider-specific client
        from config import create_llm_client, get_provider_info
        self.llm = create_llm_client(self.config.llm)

        provider_info = get_provider_info(self.config.llm)
        print(f"BOT: LLM Provider: {provider_info}")

        if LANGGRAPH_AVAILABLE:
            print("INFO: Using LangGraph for advanced reasoning workflows")

        # Initialize agents
        self._initialize_agents()

        print("SUCCESS: System initialization complete!")

    def _initialize_agents(self):
        """Initialize all agents with Azure OpenAI LLM"""

        print("BOT: Initializing agents...")

        # Initialize GitHub client
        self.github_client = GitHubClient()

        # Initialize agents with configuration (LangGraph agents take config, not llm directly)
        if LANGGRAPH_AVAILABLE:
            self.task_agent = TaskBreakdownAgent(config=self.config)
            self.coding_agent = PySparkCodingAgent(config=self.config)
            print("INFO: LangGraph agents initialized")
        else:
            self.task_agent = TaskBreakdownAgent(llm=self.llm)
            self.coding_agent = PySparkCodingAgent(llm=self.llm)
            print("PROCESS: Standard agents initialized")

        self.pr_agent = PRIssueAgent(self.github_client, llm=self.llm)

        print(f"SUCCESS: All agents initialized with {self.config.llm.provider} ({self.config.llm.model})")

    def process_single_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single GitHub issue through the full pipeline"""

        print(f"\n{'='*60}")
        print(f"LIST: Processing Issue: {issue.get('title', 'Unknown')}")
        print(f"{'='*60}")

        results = {
            "issue": issue,
            "timestamp": datetime.now().isoformat(),
            "pipeline_results": {}
        }

        try:
            # Step 1: Task breakdown using Azure OpenAI + LangGraph
            print("\nSEARCH: Step 1: Analyzing issue and breaking down tasks...")

            if LANGGRAPH_AVAILABLE:
                # Use new LangGraph agent
                task_result = self.task_agent.analyze_issue(issue)
                tasks = task_result.get("tasks", [])
                task_stats = task_result.get("summary", {})
            else:
                # Use original agent
                tasks = self.task_agent.breakdown_issue(issue)
                task_stats = self.task_agent.get_breakdown_statistics(tasks)

            results["pipeline_results"]["tasks"] = tasks
            results["pipeline_results"]["task_count"] = len(tasks)

            print(f"INFO: Task Breakdown Complete: {task_stats}")

            # Step 2: Generate PySpark notebooks for each task
            print("\n Step 2: Generating PySpark notebooks...")
            notebooks = []

            for i, task in enumerate(tasks):
                print(f" Generating notebook {i+1}/{len(tasks)}: {task.get('title', 'Unknown')}")

                try:
                    if LANGGRAPH_AVAILABLE:
                        # Use new LangGraph agent
                        notebook_result = self.coding_agent.generate_notebook(task)
                    else:
                        # Use original agent
                        notebook_result = self.coding_agent.generate_pyspark_code(task)

                    notebooks.append(notebook_result)

                except Exception as e:
                    print(f" ERROR: Error generating notebook for task {i+1}: {str(e)}")
                    notebooks.append({
                        "error": str(e),
                        "task_id": task.get("id", f"task_{i+1}"),
                        "status": "failed"
                    })

            results["pipeline_results"]["notebooks"] = notebooks
            results["pipeline_results"]["notebook_count"] = len([n for n in notebooks if "error" not in n])

            # Get code generation statistics
            successful_notebooks = len([n for n in notebooks if "error" not in n])
            print(f"INFO: Code Generation Complete: {successful_notebooks}/{len(notebooks)} successful")

            # Step 3: Create comprehensive PR with all notebooks
            print("\nPROCESS: Step 3: Creating pull request...")

            # Prepare code files for PR
            code_files = []
            for notebook in notebooks:
                if "error" not in notebook:
                    code_files.append({
                        "filepath": notebook.get("file_path", "unknown.ipynb"),
                        "content": notebook.get("content", ""),
                        "description": f"Production-ready PySpark ETL notebook: {notebook.get('task_id', 'Unknown')}"
                    })

            # Create PR with implementation notes
            implementation_notes = {
                "approach": "Claude Sonnet AI-powered generation with comprehensive ETL pipeline",
                "features": [
                    "Production-ready Jupyter notebooks",
                    "Data validation and quality checks",
                    "Performance monitoring",
                    "Error handling and logging",
                    "Unit testing framework"
                ],
                "testing": "Comprehensive unit tests included in notebooks"
            }

            pr_result = self.pr_agent.create_pr_with_code(
                code_files,
                issue,
                implementation_notes
            )
            results["pipeline_results"]["pr_creation"] = pr_result

            # Step 4: Comment and close issue
            print("\nCOMMENT: Step 4: Adding resolution comment and closing issue...")

            implementation_summary = f"""
**Implementation Summary:**
- SEARCH: **Task Analysis**: {len(tasks)} tasks identified using Claude Sonnet
- **Notebooks Generated**: {len(code_files)} production-ready Jupyter notebooks
- TEST: **Features**: Data validation, monitoring, error handling, unit tests
- **Technology**: PySpark ETL pipelines with comprehensive data processing

**Generated Artifacts:**
{self._format_artifacts_summary(notebooks)}
"""

            comment_result = self.pr_agent.comment_and_close_issue(
                issue,
                pr_result,
                implementation_summary
            )
            results["pipeline_results"]["issue_resolution"] = comment_result

            # Calculate overall success
            results["success"] = all([
                len(tasks) > 0,
                len(code_files) > 0,
                pr_result.get("pr_url"),
                comment_result.get("comment_posted", False)
            ])

            print(f"\nSUCCESS: Pipeline Complete! Success: {results['success']}")
            return results

        except Exception as e:
            print(f"\nERROR: Pipeline Error: {str(e)}")
            results["success"] = False
            results["error"] = str(e)
            return results

    def _format_artifacts_summary(self, notebooks: List[Dict[str, Any]]) -> str:
        """Format a summary of generated artifacts"""
        summary_lines = []

        for i, notebook in enumerate(notebooks, 1):
            if "error" not in notebook:
                filename = notebook.get("filename", f"notebook_{i}.ipynb")
                description = notebook.get("description", "Jupyter notebook")
                summary_lines.append(f"- **{filename}**: {description}")
            else:
                summary_lines.append(f"- **Task {i}**: Failed to generate (Error: {notebook.get('error', 'Unknown')})")

        return "\n".join(summary_lines)

    def process_multiple_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple GitHub issues"""

        print(f"\nINFO: Processing {len(issues)} issues...")

        results = []

        for i, issue in enumerate(issues, 1):
            print(f"\nLIST: Processing issue {i}/{len(issues)}")

            try:
                result = self.process_single_issue(issue)
                results.append(result)

                # Brief pause between issues
                import time
                time.sleep(1)

            except Exception as e:
                print(f"ERROR: Failed to process issue {i}: {str(e)}")
                results.append({
                    "issue": issue,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Print overall statistics
        self._print_batch_statistics(results)

        return results

    def _print_batch_statistics(self, results: List[Dict[str, Any]]) -> None:
        """Print statistics for batch processing"""

        total_issues = len(results)
        successful = len([r for r in results if r.get("success", False)])
        failed = total_issues - successful

        total_tasks = sum(r.get("pipeline_results", {}).get("task_count", 0) for r in results)
        total_notebooks = sum(r.get("pipeline_results", {}).get("notebook_count", 0) for r in results)

        print(f"\nINFO: Batch Processing Statistics:")
        print(f" LIST: Issues Processed: {total_issues}")
        print(f" SUCCESS: Successful: {successful}")
        print(f" ERROR: Failed: {failed}")
        print(f" GROWTH: Success Rate: {(successful/total_issues*100):.1f}%")
        print(f" CONFIG: Total Tasks: {total_tasks}")
        print(f" Total Notebooks: {total_notebooks}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system configuration"""
        return {
            "config": {
                "llm_model": self.config.llm.model,
                "llm_temperature": self.config.llm.temperature,
                "tracing_enabled": False,
                "tracing_note": "LangSmith integration disabled"
            },
            "agents": {
                "task_breakdown": "TaskBreakdownAgent with Claude Sonnet",
                "pyspark_coding": "PySparkCodingAgent with notebook generation",
                "pr_management": "PRIssueAgent with intelligent automation"
            },
            "capabilities": [
                "Intelligent issue analysis",
                "Production-ready notebook generation",
                "Comprehensive data validation",
                "Performance monitoring",
                "Automated PR management",
                "No tracing (LangSmith disabled)"
            ]
        }


def main():
    """Main entry point with enhanced capabilities"""

    # Initialize the orchestrator
    orchestrator = AutonomousETLOrchestrator()

    # Print system information
    system_info = orchestrator.get_system_info()
    print(f"\nTARGET: System Information:")
    print(f" BOT: LLM Model: {system_info['config']['llm_model']}")
    print(f" Tracing: {system_info['config']['tracing_note']}")
    print(f" CONFIG: Capabilities: {len(system_info['capabilities'])} features")

    # Get demo issue (or from GitHub API)
    demo_issue = get_demo_issue()

    # Process the issue
    result = orchestrator.process_single_issue(demo_issue)

    # Print final results
    print(f"\nTARGET: Final Results:")
    print(f" SUCCESS: {result['success']}")
    if result.get("pipeline_results"):
        pipeline = result["pipeline_results"]
        print(f" LIST: Tasks Generated: {pipeline.get('task_count', 0)}")
        print(f" Notebooks Created: {pipeline.get('notebook_count', 0)}")
        print(f" PROCESS: PR Created: {'Yes' if pipeline.get('pr_creation', {}).get('pr_url') else 'No'}")

    return result


def get_demo_issue() -> Dict[str, Any]:
    """Get a demo issue for testing (replace with real GitHub API call)"""
    return {
        "number": 123,
        "title": "Build ETL Pipeline for Customer Transaction Data",
        "body": """We need to build a comprehensive ETL pipeline to process customer transaction data from multiple sources.

**Requirements:**
1. Ingest CSV files from S3 bucket containing transaction records
2. Clean and validate the data (remove duplicates, validate amounts, check dates)
3. Apply business transformations:
 - Calculate running balances
 - Categorize transactions by type
 - Aggregate daily/monthly summaries
4. Join with customer master data from PostgreSQL
5. Output final dataset to Parquet format in data lake

**Data Sources:**
- Transaction CSV files: s3://data-bucket/transactions/
- Customer master data: PostgreSQL table 'customers'

**Expected Output:**
- Cleaned transaction data in Parquet format
- Summary reports (daily/monthly aggregates)
- Data quality metrics and validation reports

**Performance Requirements:**
- Process up to 10M records per day
- Complete pipeline execution within 2 hours
- Handle incremental loads

Please implement as production-ready PySpark notebooks with comprehensive error handling and monitoring.""",
        "labels": [{"name": "enhancement"}, {"name": "data-pipeline"}, {"name": "pyspark"}],
        "created_at": "2024-01-15T10:30:00Z",
        "repository": {
            "owner": {"login": "company"},
            "name": "data-engineering"
        },
        "assignees": [],
        "comments": 0
    }


if __name__ == "__main__":
    main()

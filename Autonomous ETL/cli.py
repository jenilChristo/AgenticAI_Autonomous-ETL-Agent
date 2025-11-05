#!/usr/bin/env python3
"""
Command Line Interface for Autonomous ETL Agent System
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from orchestration.agent_orchestrator import DataEngineeringAgentOrchestrator


def load_config(config_path: str = ".env") -> dict:
 """Load configuration from .env file"""
 config = {}

 if os.path.exists(config_path):
 with open(config_path, 'r') as f:
 for line in f:
 line = line.strip()
 if line and not line.startswith('#') and '=' in line:
 key, value = line.split('=', 1)
 config[key.strip()] = value.strip().strip('"\'')

 # Override with environment variables
 env_vars = [
 'GITHUB_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
 'OLLAMA_BASE_URL', 'DEFAULT_LLM_PROVIDER'
 ]

 for var in env_vars:
 if os.getenv(var):
 config[var] = os.getenv(var)

 return config


def validate_config(config: dict) -> bool:
 """Validate required configuration"""

 # Check for GitHub token
 if not config.get('GITHUB_TOKEN'):
 print("ERROR: GITHUB_TOKEN is required but not found")
 return False

 # Check for at least one LLM provider
 has_openai = bool(config.get('OPENAI_API_KEY'))
 has_anthropic = bool(config.get('ANTHROPIC_API_KEY'))
 has_ollama = bool(config.get('OLLAMA_BASE_URL'))

 if not (has_openai or has_anthropic or has_ollama):
 print("ERROR: At least one LLM provider must be configured:")
 print(" - Set OPENAI_API_KEY for OpenAI GPT models")
 print(" - Set ANTHROPIC_API_KEY for Claude models")
 print(" - Set OLLAMA_BASE_URL for local Ollama models")
 return False

 return True


def print_status_header():
 """Print the application header"""
 print("=" * 80)
 print("BOT: AUTONOMOUS ETL AGENT SYSTEM")
 print(" Intelligent Data Pipeline Generation from GitHub Issues")
 print("=" * 80)


def print_pipeline_summary(result):
 """Print a summary of the pipeline execution"""

 print("\n" + "=" * 60)
 print("STATS: PIPELINE EXECUTION SUMMARY")
 print("=" * 60)

 # Task breakdown summary
 if result.task_result:
 tasks = result.task_result.get('tasks', [])
 print(f"SUCCESS: Task Breakdown: {len(tasks)} tasks identified")
 for i, task in enumerate(tasks[:3], 1): # Show first 3 tasks
 print(f" {i}. {task.get('description', 'Unknown')[:50]}...")
 if len(tasks) > 3:
 print(f" ... and {len(tasks) - 3} more tasks")

 # Code generation summary
 if result.code_result:
 code_files = [k for k, v in result.code_result.items() if v]
 print(f"SUCCESS: Code Generation: {len(code_files)} files created")
 for filename in code_files:
 print(f" FILE: {filename}")

 # PR/Issue management summary
 if result.pr_result:
 pr_status = result.pr_result.get('status', 'unknown')
 print(f"SUCCESS: PR Management: {pr_status}")

 if 'pr_url' in result.pr_result:
 print(f" PR URL: {result.pr_result['pr_url']}")

 if 'files_created' in result.pr_result:
 print(f" Files in PR: {len(result.pr_result['files_created'])}")

 # Execution status
 print(f"\nTARGET: Overall Status: {'SUCCESS: SUCCESS' if result.success else 'ERROR: FAILED'}")

 if result.error:
 print(f"ERROR: Error: {result.error}")

 print("=" * 60)


async def run_pipeline_async(args):
 """Run the pipeline asynchronously"""

 # Load and validate configuration
 config = load_config(args.config)
 if not validate_config(config):
 return False

 # Initialize orchestrator
 orchestrator = DataEngineeringAgentOrchestrator(
 github_token=config['GITHUB_TOKEN'],
 openai_api_key=config.get('OPENAI_API_KEY'),
 anthropic_api_key=config.get('ANTHROPIC_API_KEY'),
 ollama_base_url=config.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
 default_provider=config.get('DEFAULT_LLM_PROVIDER', 'ollama')
 )

 print_status_header()
 print(f"PROCESS: Processing GitHub issue: {args.repo}/{args.issue}")
 print(f"AI: Using LLM provider: {config.get('DEFAULT_LLM_PROVIDER', 'ollama')}")
 print()

 # Run the pipeline
 try:
 result = await orchestrator.process_issue_async(
 repo_name=args.repo,
 issue_number=args.issue,
 create_pr=not args.no_pr,
 local_output_dir=args.output_dir
 )

 print_pipeline_summary(result)

 if args.output_dir and result.code_result:
 print(f"\nFOLDER: Code files saved to: {args.output_dir}")

 return result.success

 except Exception as e:
 print(f"ERROR: Pipeline execution failed: {str(e)}")
 return False


def run_pipeline_sync(args):
 """Run the pipeline synchronously"""

 # Load and validate configuration
 config = load_config(args.config)
 if not validate_config(config):
 return False

 # Initialize orchestrator
 orchestrator = DataEngineeringAgentOrchestrator(
 github_token=config['GITHUB_TOKEN'],
 openai_api_key=config.get('OPENAI_API_KEY'),
 anthropic_api_key=config.get('ANTHROPIC_API_KEY'),
 ollama_base_url=config.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
 default_provider=config.get('DEFAULT_LLM_PROVIDER', 'ollama')
 )

 print_status_header()
 print(f"PROCESS: Processing GitHub issue: {args.repo}/{args.issue}")
 print(f"AI: Using LLM provider: {config.get('DEFAULT_LLM_PROVIDER', 'ollama')}")
 print()

 # Run the pipeline
 try:
 result = orchestrator.process_issue(
 repo_name=args.repo,
 issue_number=args.issue,
 create_pr=not args.no_pr,
 local_output_dir=args.output_dir
 )

 print_pipeline_summary(result)

 if args.output_dir and result.code_result:
 print(f"\nFOLDER: Code files saved to: {args.output_dir}")

 return result.success

 except Exception as e:
 print(f"ERROR: Pipeline execution failed: {str(e)}")
 return False


def setup_environment():
 """Interactive environment setup"""
 print_status_header()
 print("CONFIG: ENVIRONMENT SETUP")
 print("=" * 60)

 config_file = ".env"
 config_exists = os.path.exists(config_file)

 if config_exists:
 print(f"ℹ Found existing configuration: {config_file}")
 response = input("Do you want to update it? (y/N): ").lower()
 if response != 'y':
 print("Setup cancelled.")
 return

 print("\nLIST: Please provide the following information:")

 # GitHub Token
 github_token = input("\nKEY: GitHub Personal Access Token: ").strip()

 # LLM Provider choice
 print("\nAI: Choose your preferred LLM provider:")
 print(" 1. OpenAI GPT-4 (requires API key)")
 print(" 2. Anthropic Claude (requires API key)")
 print(" 3. Ollama (local models)")
 print(" 4. Configure multiple providers")

 provider_choice = input("Choice (1-4): ").strip()

 openai_key = ""
 anthropic_key = ""
 ollama_url = ""
 default_provider = "ollama"

 if provider_choice in ['1', '4']:
 openai_key = input("KEY: OpenAI API Key: ").strip()
 if provider_choice == '1':
 default_provider = "openai"

 if provider_choice in ['2', '4']:
 anthropic_key = input("KEY: Anthropic API Key: ").strip()
 if provider_choice == '2':
 default_provider = "anthropic"

 if provider_choice in ['3', '4']:
 ollama_url = input(" Ollama Base URL (default: http://localhost:11434): ").strip()
 if not ollama_url:
 ollama_url = "http://localhost:11434"
 if provider_choice == '3':
 default_provider = "ollama"

 if provider_choice == '4':
 print("\nWhich provider should be the default?")
 print(" 1. OpenAI")
 print(" 2. Anthropic")
 print(" 3. Ollama")
 default_choice = input("Default provider (1-3): ").strip()
 default_provider = {"1": "openai", "2": "anthropic", "3": "ollama"}.get(default_choice, "ollama")

 # Create .env file
 env_content = f"""# Autonomous ETL Agent Configuration
# Generated by setup script

# GitHub Configuration
GITHUB_TOKEN={github_token}

# LLM Provider Configuration
DEFAULT_LLM_PROVIDER={default_provider}

# OpenAI Configuration
OPENAI_API_KEY={openai_key}

# Anthropic Configuration
ANTHROPIC_API_KEY={anthropic_key}

# Ollama Configuration
OLLAMA_BASE_URL={ollama_url}

# Optional: Logging Configuration
LOG_LEVEL=INFO
"""

 try:
 with open(config_file, 'w') as f:
 f.write(env_content)

 print(f"\nSUCCESS: Configuration saved to {config_file}")
 print("INFO: You can now run the agent system!")
 print("\nExample usage:")
 print(" python cli.py --repo owner/repo --issue 123")

 except Exception as e:
 print(f"ERROR: Failed to save configuration: {e}")


def main():
 """Main CLI entry point"""

 parser = argparse.ArgumentParser(
 description="Autonomous ETL Agent System - Generate data pipelines from GitHub issues",
 formatter_class=argparse.RawDescriptionHelpFormatter,
 epilog="""
Examples:
 # Process a GitHub issue and create PR
 python cli.py --repo myorg/myrepo --issue 123

 # Process issue without creating PR (dry run)
 python cli.py --repo myorg/myrepo --issue 123 --no-pr

 # Save generated code locally
 python cli.py --repo myorg/myrepo --issue 123 --output-dir ./generated

 # Use async processing
 python cli.py --repo myorg/myrepo --issue 123 --async

 # Interactive setup
 python cli.py --setup
 """
 )

 # Main command arguments
 parser.add_argument(
 '--repo',
 type=str,
 help='GitHub repository (format: owner/repo)'
 )

 parser.add_argument(
 '--issue',
 type=int,
 help='GitHub issue number to process'
 )

 # Optional arguments
 parser.add_argument(
 '--config',
 type=str,
 default='.env',
 help='Configuration file path (default: .env)'
 )

 parser.add_argument(
 '--no-pr',
 action='store_true',
 help='Skip PR creation (generate code only)'
 )

 parser.add_argument(
 '--output-dir',
 type=str,
 help='Local directory to save generated code'
 )

 parser.add_argument(
 '--async',
 action='store_true',
 help='Use asynchronous processing'
 )

 # Setup command
 parser.add_argument(
 '--setup',
 action='store_true',
 help='Interactive environment setup'
 )

 # Version info
 parser.add_argument(
 '--version',
 action='version',
 version='Autonomous ETL Agent System v1.0.0'
 )

 args = parser.parse_args()

 # Handle setup command
 if args.setup:
 setup_environment()
 return

 # Validate required arguments
 if not args.repo or not args.issue:
 parser.error("--repo and --issue are required (or use --setup for configuration)")

 # Run the pipeline
 if getattr(args, 'async', False):
 success = asyncio.run(run_pipeline_async(args))
 else:
 success = run_pipeline_sync(args)

 # Exit with appropriate code
 sys.exit(0 if success else 1)


if __name__ == "__main__":
 main()
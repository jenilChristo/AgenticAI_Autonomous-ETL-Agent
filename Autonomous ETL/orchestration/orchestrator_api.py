"""
API Service for Agent Orchestrator Integration 🔌
Provides REST API endpoints for the LangChain-based orchestrator
"""

from flask import Flask, request, jsonify, Blueprint
import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import threading
import uuid
from dataclasses import asdict

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.agent_orchestrator import DataEngineeringAgentOrchestrator, PipelineResult
from config import AgentConfig

# Create Blueprint for orchestrator API
orchestrator_api = Blueprint('orchestrator_api', __name__, url_prefix='/api/orchestrator')

# Global orchestrator instance
_orchestrator_instance = None
_orchestrator_lock = threading.Lock()

def get_orchestrator_instance():
    """Get or create orchestrator instance (singleton pattern)"""
    global _orchestrator_instance
    
    with _orchestrator_lock:
        if _orchestrator_instance is None:
            try:
                print("🔧 Initializing Agent Orchestrator...")
                
                # Load configuration
                config = {
                    "model_name": "gpt-4o",
                    "api_key": "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5",
                    "azure_endpoint": "https://azureopenaijenil.openai.azure.com/",
                    "azure_api_version": "2024-12-01-preview",
                    "azure_deployment_name": "gpt-4.1",
                    "github_token": os.getenv("GITHUB_TOKEN"),
                    "github_owner": os.getenv("GITHUB_OWNER", "jenilChristo"),
                    "github_repo": "jenilChristo/AgenticAI_Autonomous-ETL-Agent"  # Target repository for PR creation
                }
                
                print("📋 Configuration loaded, creating orchestrator...")
                _orchestrator_instance = DataEngineeringAgentOrchestrator(config)
                print("✅ Agent Orchestrator initialized successfully")
                print("🤖 LangGraph agents loaded and ready")
                
            except Exception as e:
                print(f"❌ Failed to initialize orchestrator: {e}")
                import traceback
                print(f"📋 Full traceback:\n{traceback.format_exc()}")
                raise
                
    return _orchestrator_instance

# In-memory storage for tracking async executions
_active_executions = {}
_execution_results = {}

class ExecutionTracker:
    """Track orchestrator execution status"""
    
    @staticmethod
    def start_execution(execution_id: str, issue_data: Dict[str, Any]) -> str:
        """Start tracking a new execution"""
        _active_executions[execution_id] = {
            'status': 'started',
            'issue_data': issue_data,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'current_step': 'initializing',
            'steps_completed': [],
            'error': None
        }
        return execution_id
    
    @staticmethod
    def update_execution(execution_id: str, status: str, current_step: str = None, error: str = None):
        """Update execution status"""
        if execution_id in _active_executions:
            _active_executions[execution_id]['status'] = status
            if current_step:
                _active_executions[execution_id]['current_step'] = current_step
                if current_step not in _active_executions[execution_id]['steps_completed']:
                    _active_executions[execution_id]['steps_completed'].append(current_step)
            if error:
                _active_executions[execution_id]['error'] = error
            _active_executions[execution_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def complete_execution(execution_id: str, result: PipelineResult):
        """Mark execution as completed"""
        if execution_id in _active_executions:
            _active_executions[execution_id]['status'] = 'completed'
            _active_executions[execution_id]['completed_at'] = datetime.now(timezone.utc).isoformat()
            _execution_results[execution_id] = result

    @staticmethod
    def get_execution_status(execution_id: str) -> Dict[str, Any]:
        """Get current execution status"""
        if execution_id in _active_executions:
            status = _active_executions[execution_id].copy()
            if execution_id in _execution_results:
                status['result'] = asdict(_execution_results[execution_id])
            return status
        return None

# API Endpoints

@orchestrator_api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint 🏥"""
    try:
        orchestrator = get_orchestrator_instance()
        return jsonify({
            'status': 'healthy',
            'orchestrator_ready': orchestrator is not None,
            'model': type(orchestrator.llm).__name__ if orchestrator else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/process', methods=['POST'])
def process_issue():
    """Process a GitHub issue through the orchestrator pipeline 🚀"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        required_fields = ['issue_id', 'title', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare issue data
        issue_data = {
            'number': data['issue_id'],
            'title': data['title'],
            'body': data.get('description', ''),
            'labels': data.get('labels', []),
            'repository': {
                'full_name': data.get('repo_name', 'local/project')
            }
        }
        
        # Determine processing mode
        async_mode = data.get('async', False)
        
        if async_mode:
            # Asynchronous processing
            execution_id = str(uuid.uuid4())
            ExecutionTracker.start_execution(execution_id, issue_data)
            
            # Start background processing
            def run_async_processing():
                try:
                    ExecutionTracker.update_execution(execution_id, 'processing', 'task_breakdown')
                    
                    orchestrator = get_orchestrator_instance()
                    # Use process_user_story_async instead of process_issue to avoid GitHub fetching
                    import asyncio
                    result = asyncio.run(orchestrator.process_user_story_async(issue_data))
                    
                    ExecutionTracker.complete_execution(execution_id, result)
                    
                except Exception as e:
                    ExecutionTracker.update_execution(execution_id, 'failed', error=str(e))
            
            thread = threading.Thread(target=run_async_processing)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'execution_id': execution_id,
                'status': 'processing',
                'message': 'Issue processing started asynchronously',
                'check_status_url': f'/api/orchestrator/status/{execution_id}'
            }), 202
            
        else:
            # Synchronous processing
            orchestrator = get_orchestrator_instance()
            # Use process_user_story instead of process_issue to avoid GitHub fetching
            result = orchestrator.process_user_story(issue_data)
            
            return jsonify({
                'status': 'completed',
                'result': asdict(result),
                'message': 'Issue processed successfully'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/process_batch', methods=['POST'])
def process_multiple_issues():
    """Process multiple issues in batch 📦"""
    try:
        data = request.get_json()
        
        if not data or 'issues' not in data:
            return jsonify({'error': 'No issues data provided'}), 400
        
        issues = data['issues']
        if not isinstance(issues, list):
            return jsonify({'error': 'Issues must be an array'}), 400
        
        # Extract issue IDs
        issue_ids = []
        for issue in issues:
            if isinstance(issue, dict) and 'issue_id' in issue:
                issue_ids.append(str(issue['issue_id']))
            elif isinstance(issue, (str, int)):
                issue_ids.append(str(issue))
        
        if not issue_ids:
            return jsonify({'error': 'No valid issue IDs found'}), 400
        
        # Process issues
        orchestrator = get_orchestrator_instance()
        results = orchestrator.process_multiple_issues(issue_ids)
        
        return jsonify({
            'status': 'completed',
            'processed_count': len(results),
            'total_count': len(issue_ids),
            'results': [asdict(result) for result in results],
            'message': f'Processed {len(results)}/{len(issue_ids)} issues successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/status/<execution_id>', methods=['GET'])
def get_execution_status(execution_id: str):
    """Get status of async execution 📊"""
    try:
        status = ExecutionTracker.get_execution_status(execution_id)
        
        if status is None:
            return jsonify({'error': 'Execution ID not found'}), 404
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get current pipeline status and memory 🧠"""
    try:
        orchestrator = get_orchestrator_instance()
        status = orchestrator.get_pipeline_status()
        
        # Add execution statistics
        status.update({
            'active_executions': len(_active_executions),
            'completed_executions': len(_execution_results),
            'execution_ids': list(_active_executions.keys())
        })
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/config', methods=['GET'])
def get_orchestrator_config():
    """Get orchestrator configuration (safe version) ⚙️"""
    try:
        orchestrator = get_orchestrator_instance()
        
        # Safe config without sensitive data
        safe_config = {
            'model_type': type(orchestrator.llm).__name__,
            'agents_available': {
                'task_agent': orchestrator.task_agent is not None,
                'coding_agent': orchestrator.coding_agent is not None,
                'pr_agent': orchestrator.pr_agent is not None
            },
            'github_configured': orchestrator.github_client.token is not None if hasattr(orchestrator.github_client, 'token') else False,
            'pipeline_ready': orchestrator.pipeline is not None,
            'repository_integration': {
                'target_repository': orchestrator.pr_agent.repo_name if orchestrator.pr_agent else 'Not configured',
                'target_branch': 'develop',
                'notebook_location': 'notebooks/',
                'integration_flow': 'Task Breakdown → PySpark Coding → PR Creation → GitHub Repository'
            }
        }
        
        return jsonify(safe_config)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/integration/status', methods=['GET'])
def get_integration_status():
    """Get detailed integration status including GitHub repository 🔗"""
    try:
        orchestrator = get_orchestrator_instance()
        
        # Check GitHub token availability
        github_token_available = os.getenv('GITHUB_TOKEN') is not None
        
        integration_status = {
            'workflow_enabled': True,
            'agents_pipeline': {
                'step_1': {
                    'name': 'Task Breakdown Agent',
                    'description': 'Analyzes GitHub issues and creates structured tasks',
                    'status': 'ready' if orchestrator.task_agent else 'not_available',
                    'llm': type(orchestrator.llm).__name__ if orchestrator.task_agent else None
                },
                'step_2': {
                    'name': 'PySpark Coding Agent',
                    'description': 'Generates unified Databricks notebook for all tasks',
                    'status': 'ready' if orchestrator.coding_agent else 'not_available',
                    'llm': type(orchestrator.llm).__name__ if orchestrator.coding_agent else None
                },
                'step_3': {
                    'name': 'PR Issue Agent',
                    'description': 'Creates PR with notebook and manages GitHub integration',
                    'status': 'ready' if orchestrator.pr_agent else 'not_available',
                    'llm': 'Claude Sonnet 3.5' if orchestrator.pr_agent else None
                }
            },
            'github_integration': {
                'target_repository': orchestrator.pr_agent.repo_name if orchestrator.pr_agent else 'Not configured',
                'repository_url': f"https://github.com/{orchestrator.pr_agent.repo_name}" if orchestrator.pr_agent else None,
                'target_branch': 'develop',
                'notebook_directory': 'notebooks/',
                'token_configured': github_token_available,
                'api_access': 'available' if github_token_available else 'mock_mode'
            },
            'pipeline_flow': [
                'User Story Input → Task Breakdown Agent',
                'Task Analysis → PySpark Coding Agent', 
                'Unified Notebook Generation → PR Issue Agent',
                'Branch Creation + PR Creation → GitHub Repository',
                'Issue Resolution + PR Linking'
            ],
            'output_locations': {
                'local_notebooks': './generated_notebooks/',
                'github_notebooks': 'notebooks/ directory in target repository',
                'pr_artifacts': './pr_artifacts/'
            }
        }
        
        return jsonify(integration_status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@orchestrator_api.route('/results/<execution_id>', methods=['GET'])
def get_execution_results(execution_id: str):
    """Get detailed results of completed execution 📋"""
    try:
        if execution_id not in _execution_results:
            return jsonify({'error': 'Results not found or execution not completed'}), 404
        
        result = _execution_results[execution_id]
        status = ExecutionTracker.get_execution_status(execution_id)
        
        return jsonify({
            'execution_info': status,
            'result': asdict(result),
            'summary': {
                'tasks_generated': len(result.tasks),
                'files_created': len(result.code_files),
                'execution_time': result.execution_time,
                'pr_created': result.pr_url is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

# Cleanup endpoint for development
@orchestrator_api.route('/cleanup', methods=['POST'])
def cleanup_executions():
    """Cleanup completed executions (development only) 🧹"""
    try:
        completed_count = len(_execution_results)
        active_count = len(_active_executions)
        
        # Clear completed executions older than 1 hour
        current_time = datetime.now(timezone.utc)
        to_remove = []
        
        for exec_id, exec_data in _active_executions.items():
            if exec_data['status'] in ['completed', 'failed']:
                completed_at = datetime.fromisoformat(exec_data.get('completed_at', exec_data.get('updated_at', exec_data['started_at'])))
                if (current_time - completed_at).total_seconds() > 3600:  # 1 hour
                    to_remove.append(exec_id)
        
        for exec_id in to_remove:
            _active_executions.pop(exec_id, None)
            _execution_results.pop(exec_id, None)
        
        return jsonify({
            'message': 'Cleanup completed',
            'removed_executions': len(to_remove),
            'remaining_active': len(_active_executions),
            'remaining_results': len(_execution_results)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

def create_orchestrator_app():
    """Create Flask app with orchestrator API"""
    app = Flask(__name__)
    app.register_blueprint(orchestrator_api)
    return app

if __name__ == '__main__':
    # Run as standalone API service
    app = create_orchestrator_app()
    
    port = int(os.getenv('PORT', 8001))
    
    print("🚀 Agent Orchestrator API starting...")
    print(f"📡 API Base URL: http://localhost:{port}/api/orchestrator")
    print(f"🏥 Health Check: http://localhost:{port}/api/orchestrator/health")
    print(f"📊 Pipeline Status: http://localhost:{port}/api/orchestrator/pipeline/status")
    
    app.run(debug=True, host='0.0.0.0', port=port)

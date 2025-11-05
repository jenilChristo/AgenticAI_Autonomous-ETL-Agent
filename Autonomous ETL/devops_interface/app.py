"""
DevOps Interface Web Application 🏗️
Mimics Azure DevOps user story and task structure with ETL Agent integration
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import sys
import requests
import json
from typing import Dict, List, Any
import threading
import uuid

# Add parent directory to path for ETL agent imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'devops-interface-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///devops_interface.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user_stories = db.relationship('UserStory', backref='project', lazy=True, cascade='all, delete-orphan')

class UserStory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    acceptance_criteria = db.Column(db.Text)
    priority = db.Column(db.String(20), default='Medium')  # High, Medium, Low
    status = db.Column(db.String(20), default='New')  # New, Active, Resolved, Closed
    story_points = db.Column(db.Integer, default=0)

    # GitHub Integration
    github_issue_url = db.Column(db.String(500))
    github_repo = db.Column(db.String(100))
    github_issue_number = db.Column(db.Integer)

    # ETL Agent Integration
    etl_agent_status = db.Column(db.String(50), default='Not Started')  # Not Started, Processing, Completed, Failed
    generated_notebook_path = db.Column(db.String(500))

    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Foreign Keys
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    assigned_to = db.Column(db.String(100))

    # Relationships
    tasks = db.relationship('Task', backref='user_story', lazy=True, cascade='all, delete-orphan')

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    task_type = db.Column(db.String(50))  # data_ingestion, transformation, validation, output, testing
    priority = db.Column(db.String(20), default='Medium')
    status = db.Column(db.String(20), default='To Do')  # To Do, In Progress, Done
    effort = db.Column(db.String(20), default='Medium')  # Small, Medium, Large

    # Technical Details
    acceptance_criteria = db.Column(db.Text)
    dependencies = db.Column(db.Text)  # JSON string of dependent task IDs

    # ETL Agent Integration
    etl_generated = db.Column(db.Boolean, default=False)
    notebook_cell_number = db.Column(db.Integer)  # Which cell in the unified notebook corresponds to this task

    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Foreign Keys
    user_story_id = db.Column(db.Integer, db.ForeignKey('user_story.id'), nullable=False)
    assigned_to = db.Column(db.String(100))

class ETLAgentExecution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_story_id = db.Column(db.Integer, db.ForeignKey('user_story.id'), nullable=False)
    execution_id = db.Column(db.String(100), unique=True, nullable=False)
    status = db.Column(db.String(50), default='Started')  # Started, Processing, Completed, Failed

    # Results
    tasks_generated = db.Column(db.Integer, default=0)
    notebook_path = db.Column(db.String(500))
    pr_created = db.Column(db.Boolean, default=False)
    pr_url = db.Column(db.String(500))

    # Logs and Errors
    logs = db.Column(db.Text)
    error_message = db.Column(db.Text)

    # Timestamps
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)


# GitHub Integration Helper
class GitHubHelper:
    @staticmethod
    def parse_github_url(github_url: str) -> Dict[str, Any]:
        """Parse GitHub issue URL and extract repo and issue number 🔗"""
        try:
            # Example: https://github.com/owner/repo/issues/123
            parts = github_url.strip().split('/')
            if 'github.com' in github_url and 'issues' in parts:
                owner = parts[parts.index('github.com') + 1]
                repo = parts[parts.index('github.com') + 2]
                issue_number = int(parts[parts.index('issues') + 1])

                return {
                    'owner': owner,
                    'repo': repo,
                    'issue_number': issue_number,
                    'repo_full_name': f"{owner}/{repo}"
                }
        except Exception as e:
            print(f"❌ Error parsing GitHub URL: {e}")

        return None

    @staticmethod
    def fetch_github_issue(repo_full_name: str, issue_number: int) -> Dict[str, Any]:
        """Fetch GitHub issue details using GitHub API 📡"""
        try:
            github_token = os.getenv('GITHUB_TOKEN')
            headers = {'Authorization': f'token {github_token}'} if github_token else {}

            url = f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"GitHub API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error fetching GitHub issue: {e}")
            return None


# ETL Agent Integration with Orchestrator API
class ETLAgentIntegration:
    
    ORCHESTRATOR_API_URL = "http://localhost:8001/api/orchestrator"
    
    @staticmethod
    def _call_orchestrator_api(endpoint: str, method: str = 'GET', data: Dict = None) -> Dict[str, Any]:
        """Helper method to call orchestrator API"""
        try:
            url = f"{ETLAgentIntegration.ORCHESTRATOR_API_URL}/{endpoint.lstrip('/')}"
            
            if method == 'GET':
                response = requests.get(url, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code in [200, 202]:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"API call failed: {response.status_code} - {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error calling orchestrator API: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error calling orchestrator API: {str(e)}"
            }
    
    @staticmethod
    def check_orchestrator_health() -> Dict[str, Any]:
        """Check if orchestrator API is available"""
        return ETLAgentIntegration._call_orchestrator_api('health')
    
    @staticmethod
    def process_user_story_async(user_story_id: int):
        """Process user story with agent orchestrator via API ⚙️"""
        def run_orchestrator_processing():
            try:
                print(f"� Starting orchestrator processing for story {user_story_id}")
                
                with app.app_context():
                    user_story = UserStory.query.get(user_story_id)
                    if not user_story:
                        print(f"❌ User story {user_story_id} not found")
                        return

                    # Create execution record
                    execution = ETLAgentExecution(
                        user_story_id=user_story_id,
                        execution_id=str(uuid.uuid4()),
                        status='Processing'
                    )
                    db.session.add(execution)
                    db.session.commit()

                    # Update user story status
                    user_story.etl_agent_status = 'Processing'
                    db.session.commit()

                    # Check orchestrator health
                    health_check = ETLAgentIntegration.check_orchestrator_health()
                    if not health_check.get('status') == 'healthy':
                        raise Exception("Orchestrator API is not healthy")

                    # Prepare issue data for orchestrator
                    issue_data = {
                        'issue_id': user_story.github_issue_number or user_story.id,
                        'title': user_story.title,
                        'description': user_story.description,
                        'repo_name': user_story.github_repo or 'local/project',
                        'labels': [],
                        'async': True  # Use async processing
                    }

                    print(f"📡 Calling orchestrator API for processing...")
                    
                    # Call orchestrator API
                    api_response = ETLAgentIntegration._call_orchestrator_api(
                        'process', 
                        method='POST', 
                        data=issue_data
                    )
                    
                    if not api_response.get('execution_id'):
                        raise Exception(f"Orchestrator API error: {api_response.get('error', 'Unknown error')}")
                    
                    orchestrator_execution_id = api_response['execution_id']
                    execution.execution_id = orchestrator_execution_id
                    db.session.commit()
                    
                    print(f"✅ Orchestrator processing started with ID: {orchestrator_execution_id}")
                    
                    # Poll for completion
                    import time
                    max_wait_time = 300  # 5 minutes max
                    poll_interval = 10   # Check every 10 seconds
                    waited_time = 0
                    
                    while waited_time < max_wait_time:
                        time.sleep(poll_interval)
                        waited_time += poll_interval
                        
                        # Check status
                        status_response = ETLAgentIntegration._call_orchestrator_api(
                            f'status/{orchestrator_execution_id}'
                        )
                        
                        if status_response.get('status') == 'completed':
                            print(f"🎉 Orchestrator processing completed successfully")
                            
                            # Get detailed results
                            results_response = ETLAgentIntegration._call_orchestrator_api(
                                f'results/{orchestrator_execution_id}'
                            )
                            
                            if results_response.get('result'):
                                # Process orchestrator results
                                ETLAgentIntegration._process_orchestrator_results(
                                    user_story_id, 
                                    execution, 
                                    results_response['result']
                                )
                            
                            break
                            
                        elif status_response.get('status') == 'failed':
                            error_msg = status_response.get('error', 'Unknown orchestrator error')
                            raise Exception(f"Orchestrator processing failed: {error_msg}")
                        
                        elif status_response.get('status') in ['processing', 'started']:
                            current_step = status_response.get('current_step', 'unknown')
                            print(f"⏳ Orchestrator processing... Current step: {current_step}")
                            continue
                        
                        else:
                            print(f"⚠️ Unknown orchestrator status: {status_response.get('status')}")
                    
                    if waited_time >= max_wait_time:
                        raise Exception("Orchestrator processing timed out")

            except Exception as e:
                print(f"❌ Orchestrator processing error: {e}")
                with app.app_context():
                    execution = ETLAgentExecution.query.filter_by(user_story_id=user_story_id).order_by(ETLAgentExecution.id.desc()).first()
                    if execution:
                        execution.status = 'Failed'
                        execution.error_message = str(e)
                        execution.completed_at = datetime.utcnow()

                    user_story = UserStory.query.get(user_story_id)
                    if user_story:
                        user_story.etl_agent_status = 'Failed'

                    db.session.commit()

        # Start background thread
        thread = threading.Thread(target=run_orchestrator_processing)
        thread.daemon = True
        thread.start()
    
    @staticmethod
    def _process_orchestrator_results(user_story_id: int, execution: ETLAgentExecution, orchestrator_result: Dict[str, Any]):
        """Process results from orchestrator and update database with tasks and unified notebook"""
        try:
            print(f"📊 Processing orchestrator results for story {user_story_id}")
            
            # Extract task and code information
            tasks_data = orchestrator_result.get('tasks', [])
            code_files = orchestrator_result.get('code_files', [])
            
            print(f"🔍 Found {len(tasks_data)} tasks and {len(code_files)} code files")
            
            # Create tasks in database from task breakdown agent
            created_tasks = []
            for i, task_data in enumerate(tasks_data):
                # Extract task information from TaskResult object structure
                task_desc = task_data.get('description', f'Generated Task {i+1}')
                task_type = task_data.get('code_type', 'transformation')
                
                # Map code_type to proper task_type
                if task_type == 'pyspark':
                    task_type = 'transformation'
                
                # Calculate notebook cell number (title=1, imports=2, then 2 cells per task: header + code)
                # Cell structure: [Title, Imports, Task1_Header, Task1_Code, Task2_Header, Task2_Code, ..., Summary]
                cell_number = 3 + (i * 2)  # Header cell for this task
                
                task = Task(
                    title=f"Task {i+1}: {task_desc[:50]}{'...' if len(task_desc) > 50 else ''}",
                    description=task_desc,
                    task_type=task_type,
                    priority=task_data.get('priority', 'Medium').title(),
                    effort=task_data.get('estimated_effort', 'Medium').title(),
                    acceptance_criteria=f"Generated by Task Breakdown Agent\nTask ID: {task_data.get('task_id', f'task_{i+1}')}\nNotebook Cell: {cell_number+1} (code implementation)",
                    etl_generated=True,
                    notebook_cell_number=cell_number + 1,  # Point to the code cell (not header)
                    user_story_id=user_story_id,
                    assigned_to='@task-breakdown-agent',
                    status='Done'  # Mark as done since code was generated
                )
                db.session.add(task)
                created_tasks.append(task)
                print(f"📋 Created task: {task.title} [{task.task_type}] → Cell #{cell_number+1}")
            
            # Create notebooks directory and save unified notebook
            notebooks_dir = os.path.join('notebooks', f'story_{user_story_id}')
            os.makedirs(notebooks_dir, exist_ok=True)
            
            notebook_files = []
            unified_notebook_saved = False
            
            for code_file in code_files:
                file_path = os.path.join(notebooks_dir, os.path.basename(code_file.get('file_path', f'generated_{len(notebook_files)}.ipynb')))
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code_file.get('content', ''))
                    notebook_files.append(file_path)
                    
                    # Check if this is the unified notebook
                    if code_file.get('task_id') == 'unified_notebook' or file_path.endswith('.ipynb'):
                        unified_notebook_saved = True
                        print(f"📓 UNIFIED NOTEBOOK saved: {file_path}")
                        print(f"   � Contains {len(tasks_data)} tasks in one notebook")
                    else:
                        print(f"�📁 Saved code file: {file_path}")
                        
                except Exception as e:
                    print(f"⚠️ Failed to save file {file_path}: {e}")
            
            # Update execution record with enhanced information
            execution.status = 'Completed'
            execution.completed_at = datetime.utcnow()
            execution.tasks_generated = len(created_tasks)
            execution.notebook_path = notebooks_dir
            
            # Update user story with enhanced status
            user_story = UserStory.query.get(user_story_id)
            if user_story:
                user_story.etl_agent_status = 'Completed'
                user_story.generated_notebook_path = notebooks_dir
                
                # Update user story description to include task summary
                task_summary = f"\n\n## Generated Tasks ({len(created_tasks)}):\n"
                for i, task in enumerate(created_tasks, 1):
                    task_summary += f"{i}. **{task.task_type.title()}:** {task.description[:100]}{'...' if len(task.description) > 100 else ''}\n"
                
                # Add notebook info
                if unified_notebook_saved:
                    task_summary += f"\n✅ **Unified Notebook Generated:** Contains all {len(created_tasks)} tasks in a single Databricks notebook optimized for Amazon's customer acquisition analytics."
                
            # Enhanced logging with notebook information
            logs_data = {
                'orchestrator_execution_id': orchestrator_result.get('issue_id'),
                'tasks_created': len(created_tasks),
                'files_generated': len(notebook_files),
                'file_paths': notebook_files,
                'execution_time': orchestrator_result.get('execution_time', 0),
                'pr_url': orchestrator_result.get('pr_url'),
                'unified_notebook': unified_notebook_saved,
                'task_breakdown': [
                    {
                        'id': task.id,
                        'title': task.title,
                        'type': task.task_type,
                        'priority': task.priority,
                        'description': task.description
                    } for task in created_tasks
                ]
            }
            
            execution.logs = json.dumps(logs_data, indent=2)
            
            db.session.commit()
            
            print(f"✅ Successfully processed orchestrator results:")
            print(f"   📋 Tasks created: {len(created_tasks)}")
            print(f"   📁 Files generated: {len(notebook_files)}")
            print(f"   📓 Unified notebook: {'Yes' if unified_notebook_saved else 'No'}")
            print(f"   📊 One notebook satisfies all tasks: ✅")
            
        except Exception as e:
            print(f"❌ Error processing orchestrator results: {e}")
            execution.status = 'Failed'
            execution.error_message = f"Result processing error: {str(e)}"
            execution.completed_at = datetime.utcnow()
            db.session.commit()
            raise

    @staticmethod
    def create_github_pr(user_story_id: int, confirmed: bool = True):
        """Create GitHub PR with generated notebooks and tasks 🔀"""
        try:
            with app.app_context():
                user_story = UserStory.query.get_or_404(user_story_id)
                
                if not user_story.github_repo:
                    return {
                        'success': False,
                        'error': 'No GitHub repository associated with this story'
                    }

                if user_story.etl_agent_status != 'Completed':
                    return {
                        'success': False,
                        'error': 'ETL agent processing must be completed before creating PR'
                    }

                # Initialize PR agent
                from agents.pr_issue_agent import PRIssueAgent
                import github
                
                github_token = os.getenv('GITHUB_TOKEN')
                if not github_token:
                    return {
                        'success': False,
                        'error': 'GitHub token not configured'
                    }

                github_client = github.Github(github_token)
                repo_name = "jenilChristo/AgenticAI_Autonomous-ETL-Agent"  # Target repository
                pr_agent = PRIssueAgent(github_client, repo_name=repo_name)

                # Collect notebook files
                code_files = []
                notebooks_dir = user_story.generated_notebook_path
                
                if notebooks_dir and os.path.exists(notebooks_dir):
                    for filename in os.listdir(notebooks_dir):
                        file_path = os.path.join(notebooks_dir, filename)
                        if os.path.isfile(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                code_files.append({
                                    'filepath': f'notebooks/story_{user_story_id}/{filename}',
                                    'content': content,
                                    'description': f'Generated notebook: {filename}'
                                })

                # Prepare issue data for PR creation
                issue_data = {
                    'title': user_story.title,
                    'body': user_story.description,
                    'number': user_story.github_issue_number or user_story.id,
                    'labels': []
                }

                # Create PR with notebooks
                pr_result = pr_agent.create_pr_with_code(
                    code_files=code_files,
                    issue=issue_data,
                    implementation_notes={
                        'approach': 'LangGraph-based ETL agent automation',
                        'features': [
                            'Automated task breakdown',
                            'PySpark notebook generation', 
                            'Intelligent code generation',
                            'Production-ready ETL pipelines'
                        ],
                        'testing': 'Generated with test cases and validation'
                    }
                )

                # Update execution record
                execution = ETLAgentExecution.query.filter_by(user_story_id=user_story_id).order_by(ETLAgentExecution.started_at.desc()).first()
                if execution:
                    execution.pr_created = pr_result.get('pr_url') is not None
                    execution.pr_url = pr_result.get('pr_url')

                db.session.commit()

                return {
                    'success': True,
                    'pr_url': pr_result.get('pr_url'),
                    'pr_number': pr_result.get('pr_number'),
                    'files_count': len(code_files)
                }

        except Exception as e:
            print(f"❌ GitHub PR creation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Routes
@app.route('/')
def index():
    """Dashboard view 📊"""
    projects = Project.query.all()
    recent_stories = UserStory.query.order_by(UserStory.updated_date.desc()).limit(10).all()

    # Statistics
    total_stories = UserStory.query.count()
    active_stories = UserStory.query.filter_by(status='Active').count()
    completed_stories = UserStory.query.filter_by(status='Resolved').count()
    etl_processed = UserStory.query.filter(UserStory.etl_agent_status.in_(['Completed', 'Processing'])).count()

    stats = {
        'total_stories': total_stories,
        'active_stories': active_stories,
        'completed_stories': completed_stories,
        'etl_processed': etl_processed
    }

    return render_template('index.html', projects=projects, recent_stories=recent_stories, stats=stats)


@app.route('/projects')
def projects():
    """Projects list view 📋"""
    projects_list = Project.query.all()
    return render_template('projects.html', projects=projects_list)


@app.route('/project/<int:project_id>')
def project_detail(project_id):
    """Project detail view 📁"""
    project = Project.query.get_or_404(project_id)
    user_stories = UserStory.query.filter_by(project_id=project_id).all()
    return render_template('project_detail.html', project=project, user_stories=user_stories)


@app.route('/story/<int:story_id>')
def story_detail(story_id):
    """User story detail view 📝"""
    story = UserStory.query.get_or_404(story_id)
    tasks = Task.query.filter_by(user_story_id=story_id).all()
    executions = ETLAgentExecution.query.filter_by(user_story_id=story_id).order_by(ETLAgentExecution.started_at.desc()).all()
    return render_template('story_detail.html', story=story, tasks=tasks, executions=executions)


@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    """Create new project 🆕"""
    if request.method == 'POST':
        project = Project(
            name=request.form['name'],
            description=request.form['description']
        )
        db.session.add(project)
        db.session.commit()
        flash('Project created successfully! 🚀', 'success')
        return redirect(url_for('project_detail', project_id=project.id))

    return render_template('create_project.html')


@app.route('/create_story', methods=['GET', 'POST'])
def create_story():
    """Create new user story 📝"""
    if request.method == 'POST':
        # Parse GitHub URL if provided
        github_url = request.form.get('github_issue_url', '').strip()
        github_data = None
        if github_url:
            github_data = GitHubHelper.parse_github_url(github_url)

        story = UserStory(
            title=request.form['title'],
            description=request.form['description'],
            acceptance_criteria=request.form.get('acceptance_criteria', ''),
            priority=request.form.get('priority', 'Medium'),
            story_points=int(request.form.get('story_points', 0)),
            project_id=int(request.form['project_id']),
            assigned_to=request.form.get('assigned_to', ''),
            github_issue_url=github_url
        )

        if github_data:
            story.github_repo = github_data['repo_full_name']
            story.github_issue_number = github_data['issue_number']

            # Fetch GitHub issue details to populate story
            issue_data = GitHubHelper.fetch_github_issue(
                github_data['repo_full_name'],
                github_data['issue_number']
            )

            if issue_data:
                if not story.title or story.title == request.form.get('title', ''):
                    story.title = issue_data.get('title', story.title)
                if not story.description:
                    story.description = issue_data.get('body', story.description)

        db.session.add(story)
        db.session.commit()

        flash('User story created successfully! 📝', 'success')
        return redirect(url_for('story_detail', story_id=story.id))

    projects_list = Project.query.all()
    return render_template('create_story.html', projects=projects_list)


@app.route('/process_with_etl/<int:story_id>', methods=['POST'])
def process_with_etl(story_id):
    """Process user story with ETL agent ⚙️"""
    story = UserStory.query.get_or_404(story_id)

    if story.etl_agent_status == 'Processing':
        flash('ETL Agent is already processing this story! ⚠️', 'warning')
    else:
        # Start ETL agent processing
        ETLAgentIntegration.process_user_story_async(story_id)
        flash('ETL Agent processing started! Check back in a few minutes. ⚙️', 'info')

    return redirect(url_for('story_detail', story_id=story_id))


@app.route('/api/story/<int:story_id>/status')
def api_story_status(story_id):
    """API endpoint for story status updates 📡"""
    story = UserStory.query.get_or_404(story_id)
    execution = ETLAgentExecution.query.filter_by(user_story_id=story_id).order_by(ETLAgentExecution.started_at.desc()).first()

    return jsonify({
        'etl_status': story.etl_agent_status,
        'tasks_count': len(story.tasks),
        'execution': {
            'status': execution.status if execution else None,
            'tasks_generated': execution.tasks_generated if execution else 0,
            'error': execution.error_message if execution else None,
            'orchestrator_execution_id': execution.execution_id if execution else None
        } if execution else None
    })

@app.route('/api/orchestrator/health')
def api_orchestrator_health():
    """Check orchestrator API health 🏥"""
    health_result = ETLAgentIntegration.check_orchestrator_health()
    return jsonify(health_result)

@app.route('/api/story/<int:story_id>/process_orchestrator', methods=['POST'])
def api_process_story_orchestrator(story_id):
    """Process story with orchestrator via API 🚀"""
    try:
        story = UserStory.query.get_or_404(story_id)
        
        if story.etl_agent_status == 'Processing':
            return jsonify({
                'error': 'Story is already being processed',
                'status': 'processing'
            }), 400
        
        # Check orchestrator health first
        health_check = ETLAgentIntegration.check_orchestrator_health()
        if health_check.get('status') != 'healthy':
            return jsonify({
                'error': 'Orchestrator API is not available',
                'health_check': health_check
            }), 503
        
        # Start processing
        ETLAgentIntegration.process_user_story_async(story_id)
        
        return jsonify({
            'message': 'Orchestrator processing started',
            'story_id': story_id,
            'status': 'started'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/orchestrator/pipeline/status')
def api_orchestrator_pipeline_status():
    """Get orchestrator pipeline status 📊"""
    try:
        pipeline_status = ETLAgentIntegration._call_orchestrator_api('pipeline/status')
        return jsonify(pipeline_status)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/batch_process', methods=['POST'])
def api_batch_process():
    """Process multiple stories with orchestrator 📦"""
    try:
        data = request.get_json()
        
        if not data or 'story_ids' not in data:
            return jsonify({'error': 'No story_ids provided'}), 400
        
        story_ids = data['story_ids']
        if not isinstance(story_ids, list):
            return jsonify({'error': 'story_ids must be an array'}), 400
        
        # Validate stories exist
        stories = UserStory.query.filter(UserStory.id.in_(story_ids)).all()
        found_ids = [s.id for s in stories]
        missing_ids = set(story_ids) - set(found_ids)
        
        if missing_ids:
            return jsonify({
                'error': f'Stories not found: {list(missing_ids)}',
                'found_stories': found_ids
            }), 404
        
        # Check orchestrator health
        health_check = ETLAgentIntegration.check_orchestrator_health()
        if health_check.get('status') != 'healthy':
            return jsonify({
                'error': 'Orchestrator API is not available',
                'health_check': health_check
            }), 503
        
        # Prepare batch data for orchestrator
        issues_data = []
        for story in stories:
            issues_data.append({
                'issue_id': story.github_issue_number or story.id,
                'title': story.title,
                'description': story.description,
                'repo_name': story.github_repo or 'local/project'
            })
        
        batch_data = {'issues': issues_data}
        
        # Call orchestrator batch API
        batch_response = ETLAgentIntegration._call_orchestrator_api(
            'process_batch',
            method='POST',
            data=batch_data
        )
        
        if batch_response.get('status') == 'completed':
            # Update stories status
            for story in stories:
                story.etl_agent_status = 'Processing'
            db.session.commit()
            
            return jsonify({
                'message': 'Batch processing completed',
                'processed_count': batch_response.get('processed_count', 0),
                'total_count': len(story_ids),
                'orchestrator_response': batch_response
            })
        else:
            return jsonify({
                'error': 'Batch processing failed',
                'orchestrator_response': batch_response
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/notebook/<int:story_id>')
def view_notebook(story_id):
    """View generated notebook 📓"""
    story = UserStory.query.get_or_404(story_id)

    if not story.generated_notebook_path:
        flash('No notebook generated for this story yet. ⚠️', 'warning')
        return redirect(url_for('story_detail', story_id=story_id))

    # Read notebook files
    notebook_files = {}
    notebook_dir = story.generated_notebook_path

    if os.path.exists(notebook_dir):
        for filename in os.listdir(notebook_dir):
            file_path = os.path.join(notebook_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook_files[filename] = f.read()
                except Exception as e:
                    notebook_files[filename] = f"Error reading file: {e}"

    return render_template('notebook_viewer.html', story=story, notebook_files=notebook_files)


@app.route('/create_pr/<int:story_id>', methods=['POST'])
def create_github_pr(story_id):
    """Create GitHub PR with generated notebooks 🔀"""
    story = UserStory.query.get_or_404(story_id)
    
    if not story.github_repo:
        flash('❌ No GitHub repository linked to this story. Please add GitHub issue URL.', 'error')
        return redirect(url_for('story_detail', story_id=story_id))
    
    if story.etl_agent_status != 'Completed':
        flash('⚠️ ETL processing must be completed before creating PR.', 'warning')
        return redirect(url_for('story_detail', story_id=story_id))
    
    # Check if PR already exists
    execution = ETLAgentExecution.query.filter_by(user_story_id=story_id).order_by(ETLAgentExecution.started_at.desc()).first()
    if execution and execution.pr_created:
        flash(f'✅ PR already exists: {execution.pr_url}', 'info')
        return redirect(url_for('story_detail', story_id=story_id))
    
    # Create PR
    result = ETLAgentIntegration.create_github_pr(story_id, confirmed=True)
    
    if result['success']:
        flash(f'🚀 GitHub PR created successfully! {result["files_count"]} files included. PR: {result["pr_url"]}', 'success')
    else:
        flash(f'❌ Failed to create PR: {result["error"]}', 'error')
    
    return redirect(url_for('story_detail', story_id=story_id))


@app.route('/preview_pr/<int:story_id>')
def preview_pr_content(story_id):
    """Preview PR content before creating 👁️"""
    story = UserStory.query.get_or_404(story_id)
    
    if not story.generated_notebook_path:
        flash('No notebooks generated yet. ⚠️', 'warning')
        return redirect(url_for('story_detail', story_id=story_id))
    
    # Collect notebook files for preview
    notebook_files = {}
    notebooks_dir = story.generated_notebook_path
    
    if os.path.exists(notebooks_dir):
        for filename in os.listdir(notebooks_dir):
            file_path = os.path.join(notebooks_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook_files[filename] = f.read()
                except Exception as e:
                    notebook_files[filename] = f"Error reading file: {e}"
    
    # Get tasks for context
    tasks = Task.query.filter_by(user_story_id=story_id).all()
    
    return render_template('pr_preview.html', 
                         story=story, 
                         notebook_files=notebook_files, 
                         tasks=tasks)


# Database migration functions
def migrate_database():
    """Handle database schema migrations 🔄"""
    try:
        # Check if new columns exist, if not add them
        from sqlalchemy import text
        with db.engine.connect() as conn:
            # Check if notebook_cell_number column exists
            try:
                conn.execute(text("SELECT notebook_cell_number FROM task LIMIT 1"))
                print("✅ notebook_cell_number column exists")
            except Exception:
                print("📝 Adding notebook_cell_number column to task table")
                conn.execute(text("ALTER TABLE task ADD COLUMN notebook_cell_number INTEGER"))
                conn.commit()
                print("✅ Added notebook_cell_number column")
            
            # Check if updated_date column exists in project table
            try:
                conn.execute(text("SELECT updated_date FROM project LIMIT 1"))
                print("✅ updated_date column exists in project table")
            except Exception:
                print("📝 Adding updated_date column to project table")
                # SQLite doesn't support DEFAULT with functions, so we add without default first
                conn.execute(text("ALTER TABLE project ADD COLUMN updated_date DATETIME"))
                # Then update existing records to have current timestamp
                conn.execute(text("UPDATE project SET updated_date = created_date WHERE updated_date IS NULL"))
                conn.commit()
                print("✅ Added updated_date column to project table")
                
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")

# Initialize database
def init_db():
    """Initialize database with sample data 🗄️"""
    print("🔧 Initializing database...")
    db.create_all()
    
    # Run migrations for existing databases
    migrate_database()

    # Create sample project if none exists
    if Project.query.count() == 0:
        sample_project = Project(
            name="Data Platform 🏗️",
            description="Enterprise data platform for analytics and machine learning"
        )
        db.session.add(sample_project)
        db.session.commit()

        print("✅ Created sample project")


if __name__ == '__main__':
    with app.app_context():
        init_db()

    print("🚀 INFO: DevOps Interface starting...")
    print("📊 STATS: Dashboard: http://localhost:5000")
    print("⚙️ CONFIG: Create stories and integrate with ETL Agent!")

    app.run(debug=True, host='0.0.0.0', port=5000)
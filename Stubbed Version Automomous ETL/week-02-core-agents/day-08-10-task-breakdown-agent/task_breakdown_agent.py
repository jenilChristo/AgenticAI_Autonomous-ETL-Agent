"""
Enhanced Task Breakdown Agent with LangGraph and Azure OpenAI
Uses advanced graph-based reasoning for intelligent task decomposition
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from config import AgentConfig, create_llm_client


class TaskBreakdownState(TypedDict):
    """State for task breakdown workflow"""
    issue_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    complexity_assessment: Dict[str, Any]
    error_messages: List[str]
    current_step: str


@dataclass
class TaskBreakdownAgent:
    """
    Advanced Task Breakdown Agent using LangGraph and Azure OpenAI

    This agent uses a multi-step graph workflow to analyze GitHub issues
    and break them down into actionable tasks with intelligent prioritization.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize with Azure OpenAI configuration"""
        self.config = config or AgentConfig()
        self.llm = create_llm_client(self.config.llm)

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

        print(f"🚀 Task Breakdown Agent initialized with {self.config.llm.provider}")
        print(f"📦 Model: {self.config.llm.model}")

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for task breakdown"""

        workflow = StateGraph(TaskBreakdownState)

        # Define workflow nodes - simplified to just task creation and validation
        workflow.add_node("analyze_issue", self._analyze_issue_node)
        workflow.add_node("validate_output", self._validate_output_node)

        # Define workflow edges - direct path from analysis to validation
        workflow.set_entry_point("analyze_issue")
        workflow.add_edge("analyze_issue", "validate_output")
        workflow.add_edge("validate_output", END)

        return workflow.compile()

    def _analyze_issue_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Analyze the issue for key insights"""

        issue = state["issue_data"]

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data engineering project manager creating Azure DevOps-style tasks from user stories.

Analyze the user story and break it down into 3-5 specific, actionable tasks that can be implemented with PySpark.

Focus on:
1. Data ingestion tasks
2. Data transformation/processing tasks  
3. Data validation/quality tasks
4. Output/export tasks
5. Testing/monitoring tasks

Each task should be clear, specific, and implementable by a developer.

Return ONLY a JSON array of tasks in this exact format:
[
  {{
    "task_title": "Clear, specific task title",
    "task_description": "Detailed description of what needs to be implemented", 
    "priority": "High|Medium|Low",
    "task_type": "Data Ingestion|Data Processing|Data Validation|Output Generation|Testing",
    "estimated_effort": "Small|Medium|Large"
  }}
]

Return only the JSON array, no additional text."""),

            ("human", """
User Story Title: {title}
User Story Description: {body}

Create 3-5 specific PySpark implementation tasks for this user story.
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = analysis_prompt.format_messages(
                title=issue.get('title', 'N/A'),
                body=issue.get('body', 'N/A')
            )
            print("\n" + "="*80)
            print("🤖 AZURE OPENAI GPT-4 REQUEST - Issue Analysis")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:800]}{'...' if len(msg.content) > 800 else ''}")
                print("-" * 40)
            
            print("🚀 SENDING REQUEST TO GPT-4...")
            response = self.llm.invoke(formatted_messages)

            # Debug: Print response from Azure OpenAI GPT-4
            print("\n📨 GPT-4 RESPONSE - Issue Analysis:")
            print("="*80)
            print(f"Response Type: {type(response).__name__}")
            print(f"Content Length: {len(response.content)} characters")
            print(f"Response Content: {response.content[:1200]}{'...' if len(response.content) > 1200 else ''}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            # Parse JSON response - expecting array of tasks
            raw_content = response.content
            tasks_text = raw_content.strip()
            
            print(f"\n🔍 PARSING TASKS JSON:")
            print(f"Raw content length: {len(tasks_text)}")
            print(f"Raw content preview: {repr(tasks_text[:200])}")
            
            # Extract JSON from code blocks if present
            if '```json' in tasks_text:
                start_marker = '```json'
                end_marker = '```'
                start_idx = tasks_text.find(start_marker) + len(start_marker)
                end_idx = tasks_text.find(end_marker, start_idx)
                if end_idx > start_idx:
                    tasks_text = tasks_text[start_idx:end_idx].strip()
                    print(f"📝 Extracted from JSON code block")
            elif '```' in tasks_text:
                # Handle generic code blocks
                parts = tasks_text.split('```')
                if len(parts) >= 3:
                    tasks_text = parts[1].strip()
                    print(f"📝 Extracted from generic code block")
            
            # Look for JSON array boundaries if not already extracted
            if not (tasks_text.startswith('[') and tasks_text.endswith(']')):
                start_idx = tasks_text.find('[')
                end_idx = tasks_text.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    tasks_text = tasks_text[start_idx:end_idx+1]
                    print(f"📝 Extracted JSON array from boundaries")
                else:
                    print(f"❌ No valid JSON array found")
                    print(f"🔍 Full response: {repr(raw_content)}")
                    raise ValueError("No valid JSON array found in GPT-4 response")

            print(f"📋 Final JSON to parse: {repr(tasks_text[:100])}...")
            
            try:
                tasks = json.loads(tasks_text)
                print(f"✅ Tasks parsed successfully: {len(tasks)} tasks")
                
                # Validate task structure
                if isinstance(tasks, list) and len(tasks) > 0:
                    for i, task in enumerate(tasks):
                        if not isinstance(task, dict):
                            print(f"⚠️ Task {i+1} is not a dict: {type(task)}")
                        else:
                            missing_keys = []
                            for key in ["task_title", "task_description"]:
                                if key not in task:
                                    missing_keys.append(key)
                            if missing_keys:
                                print(f"⚠️ Task {i+1} missing keys: {missing_keys}")
                else:
                    print(f"⚠️ Expected list of tasks, got: {type(tasks)}")
                    if not isinstance(tasks, list):
                        tasks = [tasks] if tasks else []
                
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing failed: {json_error}")
                print(f"🔍 Error at position: {getattr(json_error, 'pos', 'unknown')}")
                print(f"🔍 Problematic text: {repr(tasks_text)}")
                raise RuntimeError(f"GPT-4 returned invalid JSON for tasks: {json_error}")

            # Convert to the expected format for the rest of the pipeline
            analysis_results = {
                "tasks": tasks,
                "business_context": "User story analysis",
                "data_sources": ["user_provided"],
                "analytics_type": "descriptive"
            }

        except json.JSONDecodeError as json_error:
            # This catches JSON errors from the outer try block
            self.logger.error(f"Issue analysis JSON parsing failed: {json_error}")
            raise RuntimeError(f"GPT-4 API call failed for issue analysis - JSON error: {str(json_error)}")
        except Exception as e:
            self.logger.error(f"Issue analysis failed: {str(e)}")
            # Check if this is already a RuntimeError from JSON parsing
            if "GPT-4 returned invalid JSON" in str(e):
                raise  # Re-raise the detailed JSON error
            else:
                raise RuntimeError(f"GPT-4 API call failed for issue analysis: {str(e)}")

        state["analysis_results"] = analysis_results
        state["tasks"] = tasks  # Store tasks directly
        state["current_step"] = "analyze_issue"

        return state

    def _extract_requirements_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Extract detailed requirements from the analysis"""

        analysis = state["analysis_results"]
        issue = state["issue_data"]

        requirements_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a requirements engineer specializing in ETL systems.

Based on the issue analysis, extract detailed functional and non-functional requirements.
Focus on:
1. Data ingestion requirements
2. Transformation logic needed
3. Output specifications
4. Quality requirements
5. Performance requirements
6. Security considerations

IMPORTANT: Return ONLY valid JSON in this exact structure:
{
  "functional": {
    "data_ingestion": ["requirement1", "requirement2"],
    "transformation": ["requirement1", "requirement2"],
    "output": ["requirement1", "requirement2"]
  },
  "non_functional": {
    "performance": ["requirement1", "requirement2"],
    "quality": ["requirement1", "requirement2"],
    "security": ["requirement1", "requirement2"]
  }
}

Return only the JSON object, no additional text or formatting."""),

            ("human", """
Analysis Results: {analysis_json}

Original Issue:
Title: {title}
Body: {body}
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = requirements_prompt.format_messages(
                analysis_json=json.dumps(analysis, indent=2),
                title=issue.get('title', 'N/A'),
                body=issue.get('body', 'N/A')
            )
            print("\n" + "="*80)
            print("🤖 AZURE OPENAI GPT-4 REQUEST - Requirements Extraction")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}")
                print("-" * 40)
            
            response = self.llm.invoke(formatted_messages)
            
            # Debug: Print response from Azure OpenAI GPT-4
            print("\n🎯 AZURE OPENAI GPT-4 RESPONSE - Requirements Extraction")
            print("="*80)
            print(f"Response Type: {type(response)}")
            print(f"Response Content: {response.content}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            # Extract and parse JSON from response with enhanced debugging
            raw_content = response.content
            requirements_text = raw_content.strip()
            
            print(f"\n🔍 PARSING REQUIREMENTS JSON:")
            print(f"Raw content length: {len(requirements_text)}")
            print(f"Raw content preview: {repr(requirements_text[:200])}")
            print(f"Raw content ends with: {repr(requirements_text[-50:])}")
            
            # Enhanced JSON extraction logic
            original_text = requirements_text
            
            # Method 1: Try to extract from code blocks
            if '```json' in requirements_text:
                try:
                    start_marker = '```json'
                    end_marker = '```'
                    start_idx = requirements_text.find(start_marker) + len(start_marker)
                    end_idx = requirements_text.find(end_marker, start_idx)
                    if end_idx > start_idx:
                        requirements_text = requirements_text[start_idx:end_idx].strip()
                        print(f"📝 Extracted from json code block: {len(requirements_text)} chars")
                    else:
                        raise ValueError("Could not find closing ```")
                except Exception as e:
                    print(f"⚠️ Failed to extract from json code block: {e}")
                    requirements_text = original_text
                    
            elif '```' in requirements_text:
                try:
                    parts = requirements_text.split('```')
                    if len(parts) >= 3:
                        # Take the first content block
                        for i in range(1, len(parts), 2):  # Odd indices are content
                            candidate = parts[i].strip()
                            if candidate.startswith('{') and candidate.endswith('}'):
                                requirements_text = candidate
                                print(f"📝 Extracted from generic code block: {len(requirements_text)} chars")
                                break
                    else:
                        raise ValueError("Not enough parts in code block")
                except Exception as e:
                    print(f"⚠️ Failed to extract from generic code block: {e}")
                    requirements_text = original_text

            # Method 2: Direct JSON detection
            if requirements_text.startswith('{') and requirements_text.endswith('}'):
                print("📝 Using direct JSON format")
            else:
                # Method 3: Search for JSON boundaries
                start_idx = requirements_text.find('{')
                end_idx = requirements_text.rfind('}')
                
                print(f"🔍 JSON search: start_idx={start_idx}, end_idx={end_idx}")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    candidate_json = requirements_text[start_idx:end_idx+1]
                    print(f"📝 Candidate JSON: {len(candidate_json)} chars")
                    print(f"📝 Candidate preview: {repr(candidate_json[:100])}")
                    
                    # Validate that this looks like complete JSON
                    try:
                        # Quick validation parse
                        json.loads(candidate_json)
                        requirements_text = candidate_json
                        print("✅ JSON boundary extraction successful")
                    except json.JSONDecodeError:
                        print("❌ Extracted text is not valid JSON")
                        print(f"🔍 Full response for debugging: {repr(raw_content)}")
                        raise ValueError("No valid JSON found in GPT-4 response")
                else:
                    print("❌ No JSON boundaries found")
                    print(f"🔍 Full response for debugging: {repr(raw_content)}")
                    raise ValueError("No valid JSON found in GPT-4 response")

            print(f"\n📋 Final JSON to parse ({len(requirements_text)} chars):")
            print(f"Start: {repr(requirements_text[:50])}")
            print(f"End: {repr(requirements_text[-50:])}")
            
            try:
                requirements = json.loads(requirements_text)
                print(f"✅ Requirements parsed successfully: {type(requirements)}")
                
                # Log structure for debugging
                if isinstance(requirements, dict):
                    print(f"📊 JSON structure: {list(requirements.keys())}")
                
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing failed: {json_error}")
                print(f"🔍 Error position: {json_error.pos if hasattr(json_error, 'pos') else 'unknown'}")
                print(f"🔍 Problematic content around error:")
                if hasattr(json_error, 'pos') and json_error.pos:
                    start = max(0, json_error.pos - 50)
                    end = min(len(requirements_text), json_error.pos + 50)
                    print(f"   {repr(requirements_text[start:end])}")
                print(f"🔍 Full JSON that failed: {repr(requirements_text)}")
                raise RuntimeError(f"GPT-4 returned invalid JSON for requirements: {json_error}")

        except Exception as e:
            self.logger.error(f"Requirements extraction failed: {str(e)}")
            # Remove mock responses - fail the process instead
            raise RuntimeError(f"GPT-4 API call failed for requirements extraction: {str(e)}")

        # Merge requirements into analysis
        state["analysis_results"]["requirements"] = requirements
        state["current_step"] = "extract_requirements"

        return state

    def _assess_complexity_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Assess complexity and estimate effort"""

        analysis = state["analysis_results"]

        complexity_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical lead assessing ETL project complexity.

Analyze the requirements and provide complexity assessment.

IMPORTANT: Return ONLY valid JSON in this exact structure:
{
  "complexity_score": 7,
  "technical_challenges": ["challenge1", "challenge2"],
  "effort_estimation": "2-3 person-days",
  "risk_level": "Medium",
  "recommended_approach": "Incremental development",
  "resource_requirements": ["Python developer", "Data engineer"]
}

Return only the JSON object, no additional text or formatting."""),

            ("human", """
Requirements Analysis: {analysis_json}
""")
        ])

        try:
            print("\n🚀 SENDING COMPLEXITY ASSESSMENT TO GPT-4...")
            response = self.llm.invoke(complexity_prompt.format_messages(
                analysis_json=json.dumps(analysis, indent=2)
            ))
            
            print(f"\n📨 GPT-4 COMPLEXITY RESPONSE:")
            print(f"Content Length: {len(response.content)} characters")
            print(f"Response Content: {response.content}")

            # Extract and parse JSON from response
            complexity_text = response.content.strip()
            
            print(f"\n🔍 PARSING COMPLEXITY JSON:")
            
            # Remove code blocks if present
            if '```json' in complexity_text:
                complexity_text = complexity_text.split('```json')[1].split('```')[0].strip()
                print("📝 Extracted from json code block")
            elif '```' in complexity_text:
                parts = complexity_text.split('```')
                if len(parts) >= 3:
                    complexity_text = parts[1].strip()
                print("📝 Extracted from generic code block")
            elif complexity_text.startswith('{') and complexity_text.endswith('}'):
                print("📝 Using plain JSON format")
            else:
                # Try to find JSON in the response
                start_idx = complexity_text.find('{')
                end_idx = complexity_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    complexity_text = complexity_text[start_idx:end_idx+1]
                    print("📝 Extracted JSON from text")
                else:
                    print("❌ No valid JSON found in response")
                    raise ValueError("No valid JSON found in GPT-4 response")

            print(f"JSON to parse: {complexity_text}")
            
            try:
                complexity_assessment = json.loads(complexity_text)
                print(f"✅ Complexity assessment parsed successfully: {type(complexity_assessment)}")
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing failed: {json_error}")
                raise RuntimeError(f"GPT-4 returned invalid JSON for complexity: {json_error}")

        except Exception as e:
            self.logger.error(f"Complexity assessment failed: {str(e)}")
            # Remove mock responses - fail the process instead
            raise RuntimeError(f"GPT-4 API call failed for complexity assessment: {str(e)}")

        state["complexity_assessment"] = complexity_assessment
        state["current_step"] = "assess_complexity"

        return state

    def _generate_tasks_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Generate specific actionable tasks"""

        analysis = state["analysis_results"]
        complexity = state["complexity_assessment"]

        task_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a project manager creating actionable tasks for ETL development.

Generate 3-6 specific, actionable tasks based on the analysis and complexity assessment.

IMPORTANT: Return ONLY a valid JSON array in this exact structure:
[
  {
    "description": "Task description here",
    "priority": "high",
    "type": "pyspark",
    "effort": "medium",
    "estimated_hours": 8,
    "dependencies": [],
    "acceptance_criteria": ["criteria1", "criteria2"]
  }
]

Focus on ETL pipeline components: data ingestion, transformation, validation, output generation.
Return only the JSON array, no additional text or formatting."""),

            ("human", """
Analysis: {analysis_json}
Complexity Assessment: {complexity_json}

Generate tasks that can be implemented with PySpark for production ETL pipelines.
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = task_generation_prompt.format_messages(
                analysis_json=json.dumps(analysis, indent=2),
                complexity_json=json.dumps(complexity, indent=2)
            )
            print("\n" + "="*80)
            print("🤖 AZURE OPENAI GPT-4 REQUEST - Task Generation")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}")
                print("-" * 40)
            
            print("🚀 SENDING REQUEST TO GPT-4...")
            response = self.llm.invoke(formatted_messages)
            
            print("\n📨 GPT-4 RESPONSE:")
            print("="*80)
            print(f"Response Type: {type(response).__name__}")
            print(f"Content Length: {len(response.content)} characters")
            print(f"Response Content: {response.content}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            # Extract and parse JSON from response
            tasks_text = response.content.strip()
            
            # Remove code blocks if present
            if '```json' in tasks_text:
                tasks_text = tasks_text.split('```json')[1].split('```')[0].strip()
            elif '```' in tasks_text:
                # Handle generic code blocks
                parts = tasks_text.split('```')
                if len(parts) >= 3:
                    tasks_text = parts[1].strip()
            
            print(f"\n🔍 PARSING JSON:")
            print(f"JSON Text to parse: {tasks_text}")
            
            # Parse JSON response
            try:
                parsed_response = json.loads(tasks_text)
                print(f"✅ JSON parsed successfully: {type(parsed_response)}")
                
                # Extract tasks from the parsed response
                if isinstance(parsed_response, dict):
                    if 'tasks' in parsed_response:
                        tasks = parsed_response['tasks']
                        print(f"📋 Found tasks array with {len(tasks)} items")
                    elif 'issue_title' in parsed_response or 'issue_body' in parsed_response:
                        # The response contains the full analysis, extract tasks
                        if 'tasks' in parsed_response:
                            tasks = parsed_response['tasks']
                        else:
                            # Create tasks from the response structure
                            tasks = []
                            for key, value in parsed_response.items():
                                if key.startswith('task_') or 'task' in key.lower():
                                    if isinstance(value, dict):
                                        tasks.append(value)
                                    elif isinstance(value, str):
                                        tasks.append({
                                            "description": value,
                                            "priority": "medium",
                                            "type": "pyspark",
                                            "effort": "medium"
                                        })
                    else:
                        # Treat the entire response as a single task if no tasks array found
                        tasks = [parsed_response] if parsed_response else []
                elif isinstance(parsed_response, list):
                    tasks = parsed_response
                    print(f"📋 Found direct tasks list with {len(tasks)} items")
                else:
                    raise ValueError(f"Unexpected response format: {type(parsed_response)}")
                    
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing failed: {json_error}")
                print(f"🔍 Problematic JSON text: {repr(tasks_text[:200])}")
                raise RuntimeError(f"GPT-4 returned invalid JSON: {json_error}")

            # Ensure tasks is a valid list
            if not isinstance(tasks, list):
                tasks = [tasks] if tasks else []
                
            print(f"✅ Final tasks count: {len(tasks)}")
            for i, task in enumerate(tasks):
                print(f"   Task {i+1}: {task.get('description', str(task))[:100]}...")

        except Exception as e:
            self.logger.error(f"Task generation failed: {str(e)}")
            # Fail the process - no fallback responses
            raise RuntimeError(f"GPT-4 API call failed for task generation: {str(e)}")

        state["tasks"] = tasks
        state["current_step"] = "generate_tasks"

        return state

    def _prioritize_tasks_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Prioritize and sequence tasks optimally"""

        tasks = state["tasks"]
        complexity = state["complexity_assessment"]

        prioritization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical project manager optimizing task execution order.

Analyze the tasks and provide:
1. Optimal execution sequence
2. Parallel execution opportunities
3. Critical path identification
4. Resource allocation suggestions
5. Risk mitigation priorities

Consider:
- Task dependencies
- Resource constraints
- Risk factors
- Business value delivery

Return updated tasks array with sequence numbers and execution groups."""),

            ("human", """
Current Tasks: {tasks_json}
Complexity Assessment: {complexity_json}

Optimize the task sequence for efficient ETL development.
""")
        ])

        try:
            response = self.llm.invoke(prioritization_prompt.format_messages(
                tasks_json=json.dumps(tasks, indent=2),
                complexity_json=json.dumps(complexity, indent=2)
            ))

            prioritized_text = response.content
            if '```json' in prioritized_text:
                prioritized_text = prioritized_text.split('```json')[1].split('```')[0]

            prioritized_data = json.loads(prioritized_text)

            # Extract tasks if wrapped in response object
            if isinstance(prioritized_data, dict) and 'tasks' in prioritized_data:
                prioritized_tasks = prioritized_data['tasks']
            elif isinstance(prioritized_data, list):
                prioritized_tasks = prioritized_data
            else:
                raise ValueError("Invalid GPT-4 response format for task prioritization")

        except Exception as e:
            self.logger.error(f"Task prioritization failed: {str(e)}")
            # Fail the process - no fallback responses
            raise RuntimeError(f"GPT-4 API call failed for task prioritization: {str(e)}")

        state["tasks"] = prioritized_tasks
        state["current_step"] = "prioritize_tasks"

        return state

    def _validate_output_node(self, state: TaskBreakdownState) -> TaskBreakdownState:
        """Validate and finalize the task breakdown"""

        tasks = state["tasks"]
        
        print(f"\n🔍 VALIDATING TASKS:")
        print(f"Task count: {len(tasks)}")
        
        # Convert tasks to expected format for the orchestrator
        formatted_tasks = []
        for i, task in enumerate(tasks):
            # Map from our simplified format to the expected format
            formatted_task = {
                "description": task.get("task_description", task.get("description", f"Task {i+1}")),
                "priority": task.get("priority", "Medium").lower(),
                "type": "pyspark",  # All tasks are PySpark tasks
                "effort": task.get("estimated_effort", "Medium").lower(),
                "task_id": f"task_{i+1}",
                "title": task.get("task_title", f"Task {i+1}")
            }
            formatted_tasks.append(formatted_task)
            print(f"   Task {i+1}: {formatted_task['description'][:80]}...")

        # Update state with formatted tasks
        state["tasks"] = formatted_tasks
        
        # Basic validation summary
        priority_distribution = {}
        for task in formatted_tasks:
            priority = task.get("priority", "medium")
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

        state["analysis_results"]["summary"] = {
            "total_tasks": len(formatted_tasks),
            "priority_distribution": priority_distribution,
            "task_types": ["PySpark ETL tasks"],
            "estimated_total_effort": f"{len(formatted_tasks) * 4} hours"
        }

        state["current_step"] = "validate_output"
        print(f"✅ Task validation completed")

        return state

    def analyze_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to analyze an issue and break it down into tasks

        Args:
            issue_data: GitHub issue data

        Returns:
            Dictionary containing analysis results and tasks
        """
        try:
            print(f"🔥 TASK BREAKDOWN AGENT - Analyzing user story...")
            print(f"   📝 Issue Title: {issue_data.get('title', 'N/A')}")
            print(f"   📄 Issue Body: {issue_data.get('body', 'N/A')[:100]}...")
            print(f"   🆔 Issue Number: {issue_data.get('number', 'N/A')}")
            
            # Initialize state
            initial_state = TaskBreakdownState(
                issue_data=issue_data,
                analysis_results={},
                tasks=[],
                complexity_assessment={},
                error_messages=[],
                current_step="start"
            )

            print(f"🚀 EXECUTING LANGGRAPH WORKFLOW...")
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            print(f"✅ WORKFLOW COMPLETED")
            
            tasks_found = final_state.get("tasks", [])
            print(f"📊 BREAKDOWN RESULT: {len(tasks_found)} tasks identified")
            for i, task in enumerate(tasks_found, 1):
                print(f"   Task {i}: {task.get('description', 'No description')[:60]}...")
                print(f"           Priority: {task.get('priority', 'medium')}, Type: {task.get('type', 'pyspark')}")

            # Extract results
            return {
                "success": True,
                "analysis": final_state["analysis_results"],
                "tasks": final_state["tasks"],
                "complexity": {},  # No complexity assessment in simplified workflow
                "summary": final_state["analysis_results"].get("summary", {}),
                "errors": final_state.get("error_messages", [])
            }

        except Exception as e:
            self.logger.error(f"Task breakdown failed: {str(e)}")
            # Fail the process - no fallback responses
            raise RuntimeError(f"TaskBreakdownAgent failed due to GPT-4 API error: {str(e)}")


# Factory function for backward compatibility
def create_task_breakdown_agent(config: Optional[AgentConfig] = None) -> TaskBreakdownAgent:
    """Create a TaskBreakdownAgent instance"""
    return TaskBreakdownAgent(config)


if __name__ == "__main__":
    print("TaskBreakdownAgent - Use through orchestrator API")
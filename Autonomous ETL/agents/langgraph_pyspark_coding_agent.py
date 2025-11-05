"""
Enhanced PySpark Coding Agent with LangGraph and Azure OpenAI
Advanced notebook generation using graph-based reasoning
"""

import json
import logging
import os
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


class PySparkCodingState(TypedDict):
    """State for PySpark notebook generation workflow"""
    task_data: Dict[str, Any]
    code_structure: Dict[str, Any]
    notebook_sections: List[Dict[str, Any]]
    generated_code: Dict[str, str]
    test_cases: List[Dict[str, Any]]
    documentation: Dict[str, str]
    error_messages: List[str]
    current_step: str


@dataclass
class LangGraphPySparkCodingAgent:
    """
    Advanced PySpark Coding Agent using LangGraph and Azure OpenAI

    This agent uses a multi-step graph workflow to generate production-ready
    PySpark notebooks with comprehensive error handling, testing, and documentation.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize with Azure OpenAI configuration"""
        self.config = config or AgentConfig()
        self.llm = create_llm_client(self.config.llm)

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

        print(f"🚀 PySpark Coding Agent initialized with {self.config.llm.provider}")
        print(f"📦 Model: {self.config.llm.model}")

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for PySpark notebook generation"""

        workflow = StateGraph(PySparkCodingState)

        # Define workflow nodes
        workflow.add_node("analyze_requirements", self._analyze_requirements_node)
        workflow.add_node("design_architecture", self._design_architecture_node)
        workflow.add_node("generate_imports", self._generate_imports_node)
        workflow.add_node("generate_config", self._generate_config_node)
        workflow.add_node("generate_data_ingestion", self._generate_data_ingestion_node)
        workflow.add_node("generate_transformations", self._generate_transformations_node)
        workflow.add_node("generate_output", self._generate_output_node)
        workflow.add_node("generate_tests", self._generate_tests_node)
        workflow.add_node("create_documentation", self._create_documentation_node)
        workflow.add_node("assemble_notebook", self._assemble_notebook_node)

        # Define workflow edges
        workflow.set_entry_point("analyze_requirements")
        workflow.add_edge("analyze_requirements", "design_architecture")
        workflow.add_edge("design_architecture", "generate_imports")
        workflow.add_edge("generate_imports", "generate_config")
        workflow.add_edge("generate_config", "generate_data_ingestion")
        workflow.add_edge("generate_data_ingestion", "generate_transformations")
        workflow.add_edge("generate_transformations", "generate_output")
        workflow.add_edge("generate_output", "generate_tests")
        workflow.add_edge("generate_tests", "create_documentation")
        workflow.add_edge("create_documentation", "assemble_notebook")
        workflow.add_edge("assemble_notebook", END)

        return workflow.compile()

    def _analyze_requirements_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Analyze task requirements for PySpark implementation"""

        task = state["task_data"]

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon working on customer acquisition data platforms. Your role is to take tasks broken down by the previous agent and create production-ready Databricks notebooks with comprehensive educational content.

CRITICAL: Each notebook cell must include detailed HTML markdown descriptions for learning purposes. Make the notebooks educational and self-explanatory for developers learning PySpark and data engineering.

As an Amazon Senior Data Engineer, analyze the task requirements and extract:
1. datasources and formats like adlsgen2, s3, azure sql and formats like delta, parquet
2. read the task carefully and create the pyspark code to either do analysis if its a analysis task or create production ready code for ingestion, transformation, validation if its a etl task
3. Output should always be ingested as delta table format
4. Performance requirements for large-scale data processing and spark config tuning has to be done
5. Error handling and monitoring needs for production systems
6. Add atlease one Unit testing, with the required mock data and data quality validation requirements
7. PySpark components needed (DataFrame, SQL, MLlib, Delta Lake, etc.)

MARKDOWN REQUIREMENTS:
- Start each section with detailed markdown cells explaining the purpose, approach, and learning objectives
- Include inline comments in code that explain complex logic
- Add markdown cells after major code blocks explaining what was accomplished
- Use HTML formatting for better presentation (headings, lists, code blocks, etc.)
- Include best practices and tips for production deployment
- Add performance considerations and optimization notes
- Include troubleshooting and common issues sections

Focus on scalable, maintainable solutions that follow Amazon's data engineering best practices. The generated Databricks notebook will be picked up by the next agent for deployment and scheduling.
create the .ipynb notebooks in the output folder ./notebooks with a unique id. there should be 2 notebooks one used by the developer for analysis purposes and another notebook should contain production ready pyspark code to ingest the table in adlsgen2 path abfss://output@customeranalytics.dfs.core.windows.net/analytics/{a unique name}/bronze for ingestion,abfss://output@customeranalytics.dfs.core.windows.net/analytics/{a unique name}/gold for final facts and dimensions to be used in BI reporting.
Return structured JSON analysis."""),

            ("human", f"""
Task: {json.dumps(task, indent=2)}

Focus on PySpark-specific requirements for production ETL pipeline.
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = analysis_prompt.format_messages()
            print("\n" + "="*80)
            print("AZURE OPENAI GPT-4 REQUEST - PySpark Code Analysis")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}")
                print("-" * 40)
            
            response = self.llm.invoke(formatted_messages)
            
            # Debug: Print response from Azure OpenAI GPT-4
            print("\n AZURE OPENAI GPT-4 RESPONSE - PySpark Code Analysis")
            print("="*80)
            print(f"Response Type: {type(response)}")
            print(f"Response Content: {response.content}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            analysis_text = response.content
            if '```json' in analysis_text:
                analysis_text = analysis_text.split('```json')[1].split('```')[0]

            requirements = json.loads(analysis_text)

        except Exception as e:
            self.logger.error(f"Requirements analysis failed: {str(e)}")
            # Fail the process - no fallback responses for AI API failures
            raise RuntimeError(f"GPT-4 API call failed for requirements analysis: {str(e)}")

        state["task_data"]["requirements"] = requirements
        state["current_step"] = "analyze_requirements"

        return state

    def _design_architecture_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Design PySpark code architecture"""

        requirements = state["task_data"]["requirements"]

        architecture_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon designing Databricks notebook architecture for customer acquisition data platforms.

Based on requirements, design a production-ready Databricks notebook structure with:
1. Code organization following Amazon data engineering standards (modular functions/classes)
2. Data flow architecture optimized for customer acquisition analytics
3. Comprehensive error handling and alerting strategy for production systems
4. Performance optimization approach using Delta Lake, caching, and partitioning
5. Unit testing strategy with pytest and data quality validation
6. Configuration management for multi-environment deployment (dev/staging/prod)
7. Integration with Amazon services (S3, CloudWatch, SNS for monitoring)

The notebook should be production-ready for Databricks deployment and scheduling.
Return structured architecture plan in JSON format."""),

            ("human", f"""
Requirements: {json.dumps(requirements, indent=2)}

Design a robust PySpark notebook architecture.
""")
        ])

        try:
            response = self.llm.invoke(architecture_prompt.format_messages())

            architecture_text = response.content
            if '```json' in architecture_text:
                architecture_text = architecture_text.split('```json')[1].split('```')[0]

            architecture = json.loads(architecture_text)

        except Exception as e:
            self.logger.error(f"Architecture design failed: {str(e)}")
            # Fail the process - no fallback responses for AI API failures
            raise RuntimeError(f"GPT-4 API call failed for architecture design: {str(e)}")

        state["code_structure"] = architecture
        state["current_step"] = "design_architecture"

        return state

    def _generate_imports_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate imports and setup section"""

        requirements = state["task_data"]["requirements"]

        imports_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating Databricks notebook imports for customer acquisition data platforms.

Create comprehensive imports section for production Databricks environment including:
1. Essential PySpark imports (DataFrame, functions, types)
2. Delta Lake imports for data lake operations
3. MLlib imports for customer analytics and ML features
4. Databricks utilities (dbutils) for secret management and file operations
5. Additional libraries (boto3 for AWS services, pandas for data manipulation)
6. Spark session configuration optimized for customer data processing
7. Comprehensive logging setup with structured logging for production monitoring
8. Error handling imports with custom exception classes

Generate production-ready Databricks notebook code that follows Amazon data engineering standards."""),

            ("human", f"""
Requirements: {json.dumps(requirements, indent=2)}

Generate complete imports and Spark session setup for a Jupyter notebook.
Include all necessary PySpark components and utilities.
""")
        ])

        try:
            response = self.llm.invoke(imports_prompt.format_messages())

            # Extract code from response
            imports_code = response.content
            if '```python' in imports_code:
                imports_code = imports_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Imports generation failed: {str(e)}")
            imports_code = """
# Databricks Essential Imports for Customer Acquisition Platform
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F

# Delta Lake imports for ACID transactions
from delta.tables import DeltaTable
import delta

# MLlib imports for customer analytics
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Databricks utilities
from pyspark.dbutils import DBUtils

# AWS and additional utilities
import logging
import os
from datetime import datetime
import json

# Set up structured logging for production monitoring
import boto3
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Databricks utilities
dbutils = DBUtils(spark)

# Initialize Spark Session for Customer Acquisition Platform
spark = SparkSession.builder \\
    .appName("Customer_Acquisition_Analytics") \\
    .config("spark.sql.adaptive.enabled", "true") \\
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
    .config("spark.databricks.delta.optimizeWrite.enabled", "true") \\
    .config("spark.databricks.delta.autoCompact.enabled", "true") \\
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") \\
    .getOrCreate()

# Configure Delta Lake
spark.sql("SET spark.databricks.delta.retentionDurationCheck.enabled = false")
spark.sparkContext.setLogLevel("WARN")

print("✅ Databricks Spark session initialized for Customer Acquisition Platform")
print(f"📊 Spark Version: {spark.version}")
print(f"🗂️  Delta Lake enabled: {spark.conf.get('spark.sql.extensions', 'Not configured')}")
"""

        state["generated_code"] = {"imports": imports_code}
        state["current_step"] = "generate_imports"

        return state

    def _generate_config_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate configuration section"""

        config_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate PySpark configuration code.

Create a configuration section with:
1. Data paths and connection strings
2. Processing parameters
3. Output specifications
4. Performance tuning parameters
5. Error handling thresholds

Make it easily configurable for different environments."""),

            ("human", """
Generate configuration section for a production ETL pipeline.
Include file paths, database connections, and processing parameters.
""")
        ])

        try:
            response = self.llm.invoke(config_prompt.format_messages())

            config_code = response.content
            if '```python' in config_code:
                config_code = config_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Config generation failed: {str(e)}")
            config_code = """
# Configuration
CONFIG = {
    'input_paths': {
        'csv_files': '/path/to/input/*.csv',
        'database_url': 'jdbc:postgresql://localhost:5432/etl_db'
    },
    'output_paths': {
        'processed_data': '/path/to/output/',
        'reports': '/path/to/reports/'
    },
    'processing': {
        'batch_size': 10000,
        'max_partitions': 200,
        'cache_level': 'MEMORY_AND_DISK'
    },
    'quality': {
        'null_threshold': 0.05,
        'duplicate_threshold': 0.01
    }
}

print("📋 Configuration loaded")
"""

        state["generated_code"]["config"] = config_code
        state["current_step"] = "generate_config"

        return state

    def _generate_data_ingestion_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate data ingestion code"""

        requirements = state["task_data"]["requirements"]

        ingestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating data ingestion code for customer acquisition data platforms in Databricks.

Create robust data ingestion functions for customer acquisition analytics with:
1. Multi-source data ingestion (S3 buckets, Kinesis streams, RDS, Delta Lake, external APIs)
2. Schema evolution and validation for customer data models
3. Comprehensive error handling with CloudWatch logging and SNS alerting
4. Performance optimization using Delta Lake, partitioning by customer segments
5. Data quality checks specific to customer acquisition metrics
6. Integration with AWS Glue catalog for metadata management
7. PII data handling and compliance with Amazon data governance standards
8. Real-time and batch processing capabilities for customer data

Generate production-ready Databricks notebook code following Amazon's data engineering best practices."""),

            ("human", f"""
Requirements: {json.dumps(requirements, indent=2)}

Generate data ingestion code for the specified data sources.
Include error handling and data validation.
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = ingestion_prompt.format_messages()
            print("\n" + "="*80)
            print("🐍 AZURE OPENAI GPT-4 REQUEST - PySpark Data Ingestion Code")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}")
                print("-" * 40)
            
            response = self.llm.invoke(formatted_messages)
            
            # Debug: Print response from Azure OpenAI GPT-4
            print("\n🔥 AZURE OPENAI GPT-4 RESPONSE - PySpark Data Ingestion Code")
            print("="*80)
            print(f"Response Type: {type(response)}")
            print(f"Response Content: {response.content}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            ingestion_code = response.content
            if '```python' in ingestion_code:
                ingestion_code = ingestion_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Ingestion code generation failed: {str(e)}")
            ingestion_code = """
def load_data(file_path, file_format='csv'):
    \"\"\"Load data from various sources with error handling\"\"\"
    try:
        logger.info(f"Loading data from: {file_path}")

        if file_format.lower() == 'csv':
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
        elif file_format.lower() == 'json':
            df = spark.read.json(file_path)
        elif file_format.lower() == 'parquet':
            df = spark.read.parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"✅ Data loaded successfully. Rows: {df.count()}, Columns: {len(df.columns)}")
        return df

    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        raise

# Load input data
raw_data = load_data(CONFIG['input_paths']['csv_files'])
raw_data.printSchema()
raw_data.show(5)
"""

        state["generated_code"]["ingestion"] = ingestion_code
        state["current_step"] = "generate_data_ingestion"

        return state

    def _generate_transformations_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate data transformation code"""

        requirements = state["task_data"]["requirements"]
        task_description = state["task_data"].get("description", "")

        transformation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating data transformation code for customer acquisition analytics in Databricks.

Create comprehensive transformation functions for customer acquisition data with:
1. Advanced data cleaning and validation for customer behavioral data
2. Business logic implementation for customer acquisition metrics (CAC, LTV, conversion rates)
3. Customer segmentation and cohort analysis using advanced aggregations
4. Performance optimizations using Delta Lake merge operations and Z-ordering
5. Comprehensive error handling with structured logging for production monitoring
6. Data lineage tracking and audit trails for compliance
7. Feature engineering for downstream ML models and analytics
8. Real-time streaming transformations for customer events
9. Integration with Amazon ML services for advanced analytics

Generate production-ready Databricks notebook code that scales to Amazon's customer data volume."""),

            ("human", f"""
Task Description: {task_description}
Requirements: {json.dumps(requirements, indent=2)}

Generate transformation code that implements the business logic described.
Include data quality checks and performance optimizations.
""")
        ])

        try:
            # Debug: Print input to Azure OpenAI GPT-4
            formatted_messages = transformation_prompt.format_messages()
            print("\n" + "="*80)
            print("🐍 AZURE OPENAI GPT-4 REQUEST - PySpark Transformation Code")
            print("="*80)
            for i, msg in enumerate(formatted_messages):
                print(f"Message {i+1} ({msg.__class__.__name__}):")
                print(f"Content: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}")
                print("-" * 40)
            
            response = self.llm.invoke(formatted_messages)
            
            # Debug: Print response from Azure OpenAI GPT-4
            print("\n🔥 AZURE OPENAI GPT-4 RESPONSE - PySpark Transformation Code")
            print("="*80)
            print(f"Response Type: {type(response)}")
            print(f"Response Content: {response.content}")
            if hasattr(response, 'response_metadata'):
                print(f"Response Metadata: {response.response_metadata}")
            print("="*80)

            transformation_code = response.content
            if '```python' in transformation_code:
                transformation_code = transformation_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Transformation code generation failed: {str(e)}")
            # Fail the process - no fallback responses for AI API failures
            raise RuntimeError(f"GPT-4 API call failed for transformation code generation: {str(e)}")

        state["generated_code"]["transformations"] = transformation_code
        state["current_step"] = "generate_transformations"

        return state

    def _generate_output_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate output/save code"""

        output_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating output operations for customer acquisition data platforms in Databricks.

Create robust output functions for customer analytics with:
1. Delta Lake output with ACID transactions and time travel capabilities
2. Intelligent partitioning strategy for customer data (by region, acquisition channel, date)
3. Performance optimization using Z-ordering and data skipping
4. Comprehensive error handling with rollback capabilities
5. Data validation and quality checks before save operations
6. Integration with AWS Glue catalog for metadata management
7. Support for multiple output formats (Delta, Parquet, S3) for different consumers
8. Data archival and retention policies for compliance
9. Monitoring and alerting for data freshness and quality metrics

Generate production-ready Databricks notebook code for scalable data output operations."""),

            ("human", """
Generate output code for saving processed data.
Include multiple formats and partitioning for performance.
""")
        ])

        try:
            response = self.llm.invoke(output_prompt.format_messages())

            output_code = response.content
            if '```python' in output_code:
                output_code = output_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Output code generation failed: {str(e)}")
            output_code = """
def save_data(df, output_path, format='parquet', partition_cols=None):
    \"\"\"Save data with optimizations\"\"\"
    try:
        logger.info(f"💾 Saving data to: {output_path}")

        # Data validation before save
        if df.count() == 0:
            raise ValueError("Cannot save empty DataFrame")

        writer = df.write.mode('overwrite')

        if partition_cols:
            writer = writer.partitionBy(*partition_cols)

        if format.lower() == 'parquet':
            writer.parquet(output_path)
        elif format.lower() == 'csv':
            writer.option("header", "true").csv(output_path)
        elif format.lower() == 'json':
            writer.json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("✅ Data saved successfully")

    except Exception as e:
        logger.error(f"❌ Save operation failed: {str(e)}")
        raise

def generate_summary_report(df):
    \"\"\"Generate data summary report\"\"\"
    try:
        logger.info("📊 Generating summary report...")

        summary = {
            'total_records': df.count(),
            'columns': len(df.columns),
            'processing_time': datetime.now().isoformat()
        }

        # Add column statistics
        df.describe().show()

        return summary

    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        return {}

# Save processed data
save_data(transformed_data, CONFIG['output_paths']['processed_data'])

# Generate summary report
summary = generate_summary_report(transformed_data)
print(f"📋 Processing Summary: {summary}")
"""

        state["generated_code"]["output"] = output_code
        state["current_step"] = "generate_output"

        return state

    def _generate_tests_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Generate test cases"""

        test_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating comprehensive test suites for customer acquisition data platforms in Databricks.

Create production-ready test framework including:
1. Unit tests using pytest for all transformation functions
2. Data quality tests for customer acquisition metrics validation
3. Integration tests for end-to-end pipeline validation
4. Performance benchmarks for large-scale customer data processing
5. Schema evolution tests for backward compatibility
6. Data lineage and audit trail validation tests
7. Error handling and recovery scenario tests
8. Data freshness and SLA compliance tests
9. Customer PII data privacy and masking validation
10. Load testing for peak customer traffic scenarios

Generate comprehensive Databricks notebook test code that ensures production reliability for Amazon-scale customer data processing."""),

            ("human", """
Generate test cases for the PySpark ETL pipeline.
Include data quality checks and transformation validation.
""")
        ])

        try:
            response = self.llm.invoke(test_prompt.format_messages())

            test_code = response.content
            if '```python' in test_code:
                test_code = test_code.split('```python')[1].split('```')[0].strip()

        except Exception as e:
            self.logger.error(f"Test generation failed: {str(e)}")
            test_code = """
def test_data_quality(df):
    \"\"\"Test data quality\"\"\"
    try:
        logger.info("🧪 Running data quality tests...")

        # Check for null values
        null_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
        null_counts.show()

        # Check for duplicates
        total_rows = df.count()
        unique_rows = df.dropDuplicates().count()
        duplicate_rate = (total_rows - unique_rows) / total_rows if total_rows > 0 else 0

        logger.info(f"📊 Duplicate rate: {duplicate_rate:.2%}")

        # Add more quality checks as needed

        logger.info("✅ Data quality tests completed")

    except Exception as e:
        logger.error(f"❌ Data quality test failed: {str(e)}")

def validate_transformations(original_df, transformed_df):
    \"\"\"Validate transformation results\"\"\"
    try:
        logger.info("🔍 Validating transformations...")

        # Check row counts
        original_count = original_df.count()
        transformed_count = transformed_df.count()

        logger.info(f"Original rows: {original_count}, Transformed rows: {transformed_count}")

        # Check schema changes
        logger.info(f"Schema columns added: {len(transformed_df.columns) - len(original_df.columns)}")

        # Add specific business logic validations

        logger.info("✅ Transformation validation completed")

    except Exception as e:
        logger.error(f"❌ Transformation validation failed: {str(e)}")

# Run tests
test_data_quality(transformed_data)
validate_transformations(raw_data, transformed_data)
"""

        state["generated_code"]["tests"] = test_code
        state["current_step"] = "generate_tests"

        return state

    def _create_documentation_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Create documentation"""

        task = state["task_data"]

        doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Engineer at Amazon creating comprehensive documentation for customer acquisition data platform notebooks in Databricks.

Generate production-ready documentation including:
1. Executive summary and business context for customer acquisition analytics
2. Technical architecture overview and data flow diagrams
3. Databricks cluster requirements and configuration specifications
4. AWS services integration (S3, Glue, CloudWatch, SNS) setup instructions
5. Environment-specific deployment guide (dev/staging/prod)
6. Data governance and compliance considerations for customer data
7. Monitoring, alerting, and troubleshooting runbook
8. Performance tuning guidelines for large-scale customer data
9. Data quality validation and SLA requirements
10. Disaster recovery and backup procedures
11. API documentation for downstream consumers
12. Change management and versioning processes

Write enterprise-grade documentation that follows Amazon's technical writing standards."""),

            ("human", f"""
Task: {json.dumps(task, indent=2)}

Create documentation for this PySpark ETL notebook.
Include setup instructions and usage guide.
""")
        ])

        try:
            response = self.llm.invoke(doc_prompt.format_messages())
            documentation = response.content

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {str(e)}")
            documentation = f"""
# ETL Pipeline Documentation

## Overview
This notebook implements an ETL pipeline for {task.get('title', 'data processing')}.

## Requirements
- Apache Spark 3.x
- Python 3.8+
- Required Python packages: pyspark, pandas, numpy

## Usage
1. Update the CONFIG dictionary with your paths
2. Run cells sequentially
3. Monitor logs for progress and errors

## Configuration
Update the CONFIG section with:
- Input data paths
- Output destinations
- Processing parameters

## Troubleshooting
- Check Spark logs for detailed error messages
- Ensure sufficient memory allocation
- Verify data file accessibility
"""

        state["documentation"] = {"main": documentation}
        state["current_step"] = "create_documentation"

        return state

    def _assemble_notebook_node(self, state: PySparkCodingState) -> PySparkCodingState:
        """Assemble final notebook structure"""

        try:
            # Create notebook sections
            sections = []

            # Documentation section
            sections.append({
                "cell_type": "markdown",
                "source": [state["documentation"]["main"]]
            })

            # Code sections
            code_sections = [
                ("Imports and Setup", state["generated_code"].get("imports", "")),
                ("Configuration", state["generated_code"].get("config", "")),
                ("Data Ingestion", state["generated_code"].get("ingestion", "")),
                ("Data Transformations", state["generated_code"].get("transformations", "")),
                ("Output and Save", state["generated_code"].get("output", "")),
                ("Testing and Validation", state["generated_code"].get("tests", ""))
            ]

            for title, code in code_sections:
                if code:
                    sections.append({
                        "cell_type": "markdown",
                        "source": [f"## {title}"]
                    })
                    sections.append({
                        "cell_type": "code",
                        "source": [code],
                        "execution_count": None,
                        "outputs": []
                    })

            state["notebook_sections"] = sections
            state["current_step"] = "assemble_notebook"

        except Exception as e:
            self.logger.error(f"Notebook assembly failed: {str(e)}")
            state["error_messages"].append(f"Assembly failed: {str(e)}")

        return state

    def generate_multi_task_notebook(self, user_story: Dict[str, Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a complete PySpark notebook for multiple tasks in a user story

        Args:
            user_story: User story information and requirements
            tasks: List of tasks to include in the notebook

        Returns:
            Dictionary containing notebook content and metadata
        """
        try:
            print(f"\n🔥 PYSPARK AGENT - Starting notebook generation...")
            print(f"   📋 Story Title: {user_story.get('title', 'Unknown')}")
            print(f"   📊 Tasks to Process: {len(tasks)}")
            for i, task in enumerate(tasks, 1):
                print(f"      Task {i}: {task.get('description', 'No description')[:80]}...")
            
            self.logger.info(f"📓 Generating multi-task notebook for story: {user_story.get('title', 'Unknown')}")
            
            # Create notebook cells
            notebook_cells = []
            
            # 1. Title and Overview cell
            title_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {user_story.get('title', 'ETL Pipeline')} 📊\n\n",
                    f"**Description:** {user_story.get('description', 'Customer acquisition analytics pipeline')}\n\n",
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                    f"**Tasks Covered:** {len(tasks)} tasks\n\n",
                    "---\n"
                ]
            }
            notebook_cells.append(title_cell)
            
            # 2. Imports and Setup cell  
            imports_cell = {
                "cell_type": "code",
                "metadata": {
                    "tags": ["setup"]
                },
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Amazon Senior Data Engineer - Customer Acquisition Analytics\n",
                    "# Production-ready Databricks notebook for customer acquisition metrics\n",
                    "\n",
                    "from pyspark.sql import SparkSession\n",
                    "from pyspark.sql.functions import *\n",
                    "from pyspark.sql.types import *\n",
                    "from delta import *\n",
                    "import boto3\n",
                    "from datetime import datetime, timedelta\n",
                    "import logging\n",
                    "\n",
                    "# Configure logging for production\n",
                    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n",
                    "logger = logging.getLogger(__name__)\n",
                    "\n",
                    "# Initialize Spark session with Delta Lake and production configurations\n",
                    "spark = SparkSession.builder \\\n",
                    "    .appName('CustomerAcquisitionAnalytics') \\\n",
                    "    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \\\n",
                    "    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \\\n",
                    "    .config('spark.sql.adaptive.enabled', 'true') \\\n",
                    "    .config('spark.sql.adaptive.coalescePartitions.enabled', 'true') \\\n",
                    "    .config('spark.databricks.delta.optimizeWrite.enabled', 'true') \\\n",
                    "    .getOrCreate()\n",
                    "\n",
                    "# Set log level to reduce verbose output\n",
                    "spark.sparkContext.setLogLevel('WARN')\n",
                    "\n",
                    "print('✅ Spark session initialized for customer acquisition analytics')\n",
                    "print(f'🔧 Spark version: {spark.version}')\n",
                    "print(f'📊 Available cores: {spark.sparkContext.defaultParallelism}')"
                ]
            }
            notebook_cells.append(imports_cell)
            
            # 3. Process each task separately and generate cells
            for i, task in enumerate(tasks, 1):
                print(f"🔄 Processing Task {i}/{len(tasks)}: {task.get('description', 'Unknown task')[:50]}...")
                
                # Process task individually to generate comprehensive cells
                task_cells = self._process_task_separately(task, user_story, i)
                
                # Add all cells generated for this task
                notebook_cells.extend(task_cells)
                
                print(f"✅ Task {i} processed - Added {len(task_cells)} cells")
            
            # 4. Final summary cell
            summary_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary 📈\n\n",
                    f"This notebook completed {len(tasks)} tasks for customer acquisition analytics:\n\n",
                    "\n".join([f"- **Task {i+1}:** {task.get('description', 'Data processing')}" for i, task in enumerate(tasks)]),
                    "\n\n**Next Steps:**\n",
                    "- Monitor job execution in Databricks\n", 
                    "- Validate data quality metrics\n",
                    "- Set up automated scheduling\n",
                    "- Configure alerts for pipeline failures\n\n",
                    "**Production Ready:** ✅ Optimized for Amazon's customer acquisition platform"
                ]
            }
            notebook_cells.append(summary_cell)
            
            # Create complete notebook structure in proper .ipynb format
            notebook_content = {
                "cells": notebook_cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3 (PySpark)",
                        "language": "python", 
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.10"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Convert notebook to properly formatted .ipynb string
            notebook_json = json.dumps(notebook_content, indent=2, ensure_ascii=False)
            
            print(f"🔥 PYSPARK AGENT - Notebook generation completed!")
            print(f"   📓 Total Cells: {len(notebook_cells)}")
            print(f"   📊 Tasks Processed: {len(tasks)}")
            print(f"   📄 Notebook Size: {len(notebook_json)} characters")
            print(f"   ✅ Ready for .ipynb export")
            
            self.logger.info(f"✅ Multi-task notebook generated with {len(notebook_cells)} cells")
            
            return {
                "success": True,
                "notebook": notebook_content,
                "notebook_content": notebook_json,
                "cells_count": len(notebook_cells),
                "tasks_processed": len(tasks),
                "format": "ipynb"
            }
            
        except Exception as e:
            self.logger.error(f"Multi-task notebook generation failed: {str(e)}")
            # Fail the process - no fallback responses for AI API failures
            raise RuntimeError(f"PySpark coding agent failed due to API error: {str(e)}")

    def _process_task_separately(self, task: Dict[str, Any], user_story: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Process a single task and generate multiple cells for comprehensive implementation"""
        
        cells = []
        task_type = task.get('task_type', 'transformation').lower()
        task_desc = task.get('description', '')
        
        print(f"   🎯 Task Type: {task_type}")
        print(f"   📝 Description: {task_desc[:80]}...")
        
        # 1. Task Header Cell with Educational Context
        header_cell = {
            "cell_type": "markdown", 
            "metadata": {"tags": [f"task-{task_number}"]},
            "source": [
                f"## Task {task_number}: {task.get('title', task.get('description', 'Data Processing'))} 🎯\n\n",
                f"<div style='background-color: #e7f3ff; padding: 15px; border-left: 4px solid #2196F3;'>\n",
                f"<h3>📋 Task Overview</h3>\n",
                f"<ul>\n",
                f"<li><strong>Type:</strong> {task.get('task_type', 'transformation')}</li>\n",
                f"<li><strong>Priority:</strong> {task.get('priority', 'Medium')}</li>\n",
                f"<li><strong>Estimated Effort:</strong> {task.get('effort', 'Medium')}</li>\n",
                f"</ul>\n",
                f"<p><strong>Description:</strong> {task.get('description', 'Process data for customer acquisition analytics')}</p>\n",
                f"</div>\n\n",
                f"### 🎓 Learning Objectives\n",
                f"In this task, you will learn:\n",
                f"- Production-ready PySpark patterns for {task.get('task_type', 'data processing')}\n",
                f"- Error handling and logging best practices\n",
                f"- Performance optimization techniques for large datasets\n",
                f"- Data quality validation methods\n",
                f"- Amazon-scale data engineering patterns\n\n",
                f"### 🏗️ Architecture Pattern\n",
                f"This task follows the **Medallion Architecture** pattern:\n",
                f"- **Bronze Layer:** Raw data ingestion with minimal processing\n",
                f"- **Silver Layer:** Cleaned and validated data with business rules applied\n",
                f"- **Gold Layer:** Aggregated metrics ready for analytics and reporting\n\n",
                "---\n"
            ]
        }
        cells.append(header_cell)
        
        # 2. Generate task-specific cells based on type
        if 'ingestion' in task_type or 'load' in task_desc.lower():
            cells.extend(self._generate_ingestion_cells(task, task_number))
        elif 'transformation' in task_type or 'transform' in task_desc.lower():
            cells.extend(self._generate_transformation_cells(task, task_number))
        elif 'validation' in task_type or 'quality' in task_desc.lower():
            cells.extend(self._generate_validation_cells(task, task_number))
        elif 'output' in task_type or 'save' in task_desc.lower():
            cells.extend(self._generate_output_cells(task, task_number))
        else:
            cells.extend(self._generate_generic_processing_cells(task, task_number))
        
        # 3. Task Summary Cell
        summary_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-summary"]},
            "source": [
                f"### ✅ Task {task_number} Completed\n\n",
                f"Task '{task.get('title', 'Data Processing')}' has been executed successfully.\n",
                "Check the output above for results and any validation messages.\n\n"
            ]
        }
        cells.append(summary_cell)
        
        return cells

    def _generate_ingestion_cells(self, task: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Generate cells specifically for data ingestion tasks"""
        cells = []
        
        # Educational introduction for data ingestion
        intro_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-intro"]},
            "source": [
                f"### 📥 Data Ingestion Deep Dive\n\n",
                f"<div style='background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107;'>\n",
                f"<h4>🎓 What You'll Learn</h4>\n",
                f"<p>Data ingestion is the first critical step in any ETL pipeline. In this section, you'll learn:</p>\n",
                f"<ul>\n",
                f"<li><strong>Schema Enforcement:</strong> How to define and enforce data schemas for reliability</li>\n",
                f"<li><strong>Performance Optimization:</strong> Caching strategies and partition management</li>\n",
                f"<li><strong>Error Handling:</strong> Robust error handling for production systems</li>\n",
                f"<li><strong>Data Validation:</strong> Early detection of data quality issues</li>\n",
                f"</ul>\n",
                f"</div>\n\n",
                f"### 🏗️ Ingestion Architecture\n",
                f"```\n",
                f"Raw Data Sources → Schema Validation → Data Loading → Quality Checks → Bronze Layer\n",
                f"     (S3/ADLS)         (StructType)      (DataFrame)      (Validation)    (Delta Lake)\n",
                f"```\n\n",
                f"### ⚡ Performance Tips\n",
                f"- **Use schema enforcement** to catch errors early\n",
                f"- **Cache frequently accessed DataFrames** to avoid recomputation\n",
                f"- **Monitor partition sizes** to optimize Spark performance\n",
                f"- **Use appropriate file formats** (Parquet/Delta) for better compression\n\n"
            ]
        }
        cells.append(intro_cell)
        
        # Configuration cell
        config_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-config"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Data Ingestion Configuration\n",
                "# Configure paths and parameters for customer acquisition data ingestion\n",
                "\n",
                "# S3 Configuration for Amazon Customer Data\n",
                "s3_config = {\n",
                "    'bucket': 'amazon-customer-acquisition-data',\n",
                "    'raw_data_path': 's3a://amazon-customer-acquisition-data/raw/customer_events/',\n",
                "    'processed_path': 's3a://amazon-customer-acquisition-data/processed/',\n",
                "    'delta_path': 's3a://amazon-customer-acquisition-data/delta/customer_acquisition/'\n",
                "}\n",
                "\n",
                "# Data schema definition for validation\n",
                "customer_event_schema = StructType([\n",
                "    StructField('customer_id', StringType(), False),\n",
                "    StructField('event_type', StringType(), False),\n",
                "    StructField('event_date', TimestampType(), False),\n",
                "    StructField('channel', StringType(), True),\n",
                "    StructField('campaign_id', StringType(), True),\n",
                "    StructField('marketing_spend', DoubleType(), True),\n",
                "    StructField('revenue', DoubleType(), True),\n",
                "    StructField('region', StringType(), True)\n",
                "])\n",
                "\n",
                "print('📋 Ingestion configuration loaded')\n",
                "print(f'🗂️  Source: {s3_config[\"raw_data_path\"]}')\n",
                "print(f'📊 Schema fields: {len(customer_event_schema.fields)}')"
            ]
        }
        cells.append(config_cell)
        
        # Educational explanation for schema definition
        schema_explanation_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-schema-explanation"]},
            "source": [
                f"#### 📋 Schema Definition Explained\n\n",
                f"<div style='background-color: #e8f5e8; padding: 15px; border-left: 4px solid #28a745;'>\n",
                f"<h5>🔍 Why Schema Enforcement Matters</h5>\n",
                f"<p>The schema definition above serves several critical purposes:</p>\n",
                f"<ul>\n",
                f"<li><strong>Data Type Safety:</strong> Ensures consistent data types across all records</li>\n",
                f"<li><strong>Null Handling:</strong> Defines which fields are required (False) vs optional (True)</li>\n",
                f"<li><strong>Performance:</strong> Avoids expensive schema inference on large datasets</li>\n",
                f"<li><strong>Error Detection:</strong> Catches malformed data early in the pipeline</li>\n",
                f"</ul>\n",
                f"</div>\n\n",
                f"**Key Schema Patterns:**\n",
                f"- `StringType()` for text fields (customer_id, event_type, channel)\n",
                f"- `TimestampType()` for date/time fields with time zone support\n",
                f"- `DoubleType()` for precise numeric calculations (revenue, spend)\n",
                f"- `nullable=False` for business-critical fields that must be present\n\n",
                f"💡 **Pro Tip:** Always define schemas explicitly in production to avoid performance penalties from schema inference.\n\n"
            ]
        }
        cells.append(schema_explanation_cell)
        
        # Data loading cell
        loading_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-load"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Load Raw Customer Data\n",
                "# Load and validate customer acquisition data from S3\n",
                "\n",
                "try:\n",
                "    logger.info(f'🔄 Starting data ingestion from {s3_config[\"raw_data_path\"]}')\n",
                "    \n",
                "    # Read data with schema validation\n",
                "    df_raw_events = spark.read \\\n",
                "        .option('header', 'true') \\\n",
                "        .schema(customer_event_schema) \\\n",
                "        .csv(s3_config['raw_data_path'])\n",
                "    \n",
                "    # Cache for performance optimization\n",
                "    df_raw_events.cache()\n",
                "    \n",
                "    # Get basic statistics\n",
                "    total_records = df_raw_events.count()\n",
                "    \n",
                "    logger.info(f'📥 Successfully loaded {total_records:,} customer event records')\n",
                "    \n",
                "    print(f'✅ Data Loading Completed:')\n",
                "    print(f'   📊 Total Records: {total_records:,}')\n",
                "    print(f'   📋 Columns: {len(df_raw_events.columns)}')\n",
                "    print(f'   🗂️  Partitions: {df_raw_events.rdd.getNumPartitions()}')\n",
                "    \n",
                "    # Display schema\n",
                "    print('\\n📋 Data Schema:')\n",
                "    df_raw_events.printSchema()\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data ingestion failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(loading_cell)
        
        # Educational explanation for loading optimization
        loading_explanation_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-loading-explanation"]},
            "source": [
                f"#### ⚡ Data Loading Optimization Explained\n\n",
                f"<div style='background-color: #f0f8ff; padding: 15px; border-left: 4px solid #17a2b8;'>\n",
                f"<h5>🚀 Performance Optimization Techniques</h5>\n",
                f"<p>The loading code above implements several performance optimizations:</p>\n",
                f"<ul>\n",
                f"<li><strong>.cache():</strong> Stores DataFrame in memory for repeated access</li>\n",
                f"<li><strong>Schema enforcement:</strong> Avoids costly schema inference</li>\n",
                f"<li><strong>Partition monitoring:</strong> Ensures optimal parallel processing</li>\n",
                f"<li><strong>Error handling:</strong> Graceful failure with detailed logging</li>\n",
                f"</ul>\n",
                f"</div>\n\n",
                f"**Caching Strategy:**\n",
                f"```python\n",
                f"df.cache()  # Stores in memory (fast access, limited by RAM)\n",
                f"df.persist(StorageLevel.MEMORY_AND_DISK)  # Spills to disk if needed\n",
                f"df.unpersist()  # Releases cached data when no longer needed\n",
                f"```\n\n",
                f"**Partition Management:**\n",
                f"- **Too few partitions:** Underutilizes cluster resources\n",
                f"- **Too many partitions:** High overhead from task scheduling\n",
                f"- **Rule of thumb:** 2-3 partitions per CPU core in cluster\n\n",
                f"💡 **Pro Tip:** Use `.rdd.getNumPartitions()` to monitor and `.repartition(n)` or `.coalesce(n)` to optimize.\n\n"
            ]
        }
        cells.append(loading_explanation_cell)
        
        # Data preview cell
        preview_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-preview"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Data Preview and Quality Check\n",
                "# Preview loaded data and perform initial quality assessment\n",
                "\n",
                "# Display sample data\n",
                "print('📊 Sample Customer Event Data:')\n",
                "df_raw_events.show(10, truncate=False)\n",
                "\n",
                "# Basic data quality checks\n",
                "print('\\n🔍 Data Quality Summary:')\n",
                "print(f'   Null customer_ids: {df_raw_events.filter(col(\"customer_id\").isNull()).count():,}')\n",
                "print(f'   Invalid revenues: {df_raw_events.filter(col(\"revenue\") < 0).count():,}')\n",
                "print(f'   Future dates: {df_raw_events.filter(col(\"event_date\") > current_timestamp()).count():,}')\n",
                "\n",
                "# Event type distribution\n",
                "print('\\n📈 Event Type Distribution:')\n",
                "df_raw_events.groupBy('event_type').count().orderBy(desc('count')).show()\n",
                "\n",
                "# Channel distribution\n",
                "print('\\n📊 Acquisition Channel Distribution:')\n",
                "df_raw_events.groupBy('channel').count().orderBy(desc('count')).show()"
            ]
        }
        cells.append(preview_cell)
        
        return cells

    def _generate_transformation_cells(self, task: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Generate cells specifically for data transformation tasks"""
        cells = []
        
        # Educational introduction for transformations
        transformation_intro_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-transform-intro"]},
            "source": [
                f"### 🔄 Data Transformation Deep Dive\n\n",
                f"<div style='background-color: #fff3e0; padding: 15px; border-left: 4px solid #ff9800;'>\n",
                f"<h4>🎓 Transformation Learning Objectives</h4>\n",
                f"<p>Data transformation is where raw data becomes business-ready insights. You'll learn:</p>\n",
                f"<ul>\n",
                f"<li><strong>Data Cleaning:</strong> Null handling, deduplication, and validation</li>\n",
                f"<li><strong>Feature Engineering:</strong> Creating derived columns for analytics</li>\n",
                f"<li><strong>Business Logic:</strong> Implementing domain-specific rules</li>\n",
                f"<li><strong>Performance Patterns:</strong> Efficient transformation techniques</li>\n",
                f"</ul>\n",
                f"</div>\n\n",
                f"### 🏗️ Transformation Pipeline Architecture\n",
                f"```\n",
                f"Raw Data → Data Cleaning → Feature Engineering → Business Logic → Metrics Calculation\n",
                f"(Bronze)     (Validation)    (New Columns)      (Rules)         (Aggregations)\n",
                f"```\n\n",
                f"### 🔧 PySpark Transformation Best Practices\n",
                f"- **Use Column API** (`col()`, `when()`, `otherwise()`) for readable code\n",
                f"- **Chain operations** efficiently to minimize shuffles\n",
                f"- **Cache intermediate results** that are used multiple times\n",
                f"- **Use coalesce()** to handle null values gracefully\n",
                f"- **Partition by relevant keys** to optimize joins and aggregations\n\n"
            ]
        }
        cells.append(transformation_intro_cell)
        
        # Data cleaning cell
        cleaning_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-cleaning"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Data Cleaning and Preparation\n",
                "# Clean and prepare customer data for analytics\n",
                "\n",
                "try:\n",
                "    logger.info('🧹 Starting data cleaning and preparation')\n",
                "    \n",
                "    # Data cleaning pipeline\n",
                "    df_cleaned = df_raw_events \\\n",
                "        .filter(col('customer_id').isNotNull()) \\\n",
                "        .filter(col('event_date').isNotNull()) \\\n",
                "        .filter(col('revenue') >= 0) \\\n",
                "        .withColumn('year_month', date_format(col('event_date'), 'yyyy-MM')) \\\n",
                "        .withColumn('year', year(col('event_date'))) \\\n",
                "        .withColumn('quarter', quarter(col('event_date'))) \\\n",
                "        .withColumn('acquisition_channel', coalesce(col('channel'), lit('unknown'))) \\\n",
                "        .withColumn('clean_campaign_id', coalesce(col('campaign_id'), lit('organic'))) \\\n",
                "        .withColumn('revenue_bucket', \n",
                "            when(col('revenue') == 0, 'zero')\n",
                "            .when(col('revenue') <= 50, 'low')\n",
                "            .when(col('revenue') <= 200, 'medium')\n",
                "            .when(col('revenue') <= 500, 'high')\n",
                "            .otherwise('premium')\n",
                "        )\n",
                "    \n",
                "    # Remove duplicates based on customer_id, event_type, and event_date\n",
                "    df_cleaned = df_cleaned.dropDuplicates(['customer_id', 'event_type', 'event_date'])\n",
                "    \n",
                "    # Cache cleaned data\n",
                "    df_cleaned.cache()\n",
                "    \n",
                "    cleaned_count = df_cleaned.count()\n",
                "    original_count = df_raw_events.count()\n",
                "    \n",
                "    print(f'✅ Data Cleaning Completed:')\n",
                "    print(f'   📥 Original Records: {original_count:,}')\n",
                "    print(f'   📤 Cleaned Records: {cleaned_count:,}')\n",
                "    print(f'   🗑️  Records Removed: {original_count - cleaned_count:,} ({((original_count - cleaned_count) / original_count * 100):.2f}%)')\n",
                "    \n",
                "    logger.info(f'✅ Data cleaning completed - {cleaned_count:,} clean records')\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data cleaning failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(cleaning_cell)
        
        # Educational explanation for data cleaning techniques
        cleaning_explanation_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-cleaning-explanation"]},
            "source": [
                f"#### 🧹 Data Cleaning Techniques Explained\n\n",
                f"<div style='background-color: #f8f9fa; padding: 15px; border-left: 4px solid #6c757d;'>\n",
                f"<h5>🔍 Cleaning Operations Breakdown</h5>\n",
                f"<p>The data cleaning pipeline above implements several critical operations:</p>\n",
                f"</div>\n\n",
                f"**1. Null Value Filtering:**\n",
                f"```python\n",
                f".filter(col('customer_id').isNotNull())  # Remove records without customer ID\n",
                f".filter(col('event_date').isNotNull())   # Ensure all events have dates\n",
                f"```\n\n",
                f"**2. Business Rule Validation:**\n",
                f"```python\n",
                f".filter(col('revenue') >= 0)  # No negative revenue values\n",
                f"```\n\n",
                f"**3. Feature Engineering:**\n",
                f"```python\n",
                f".withColumn('year_month', date_format(col('event_date'), 'yyyy-MM'))  # Time grouping\n",
                f".withColumn('revenue_bucket', when(...))  # Categorical segmentation\n",
                f"```\n\n",
                f"**4. Data Standardization:**\n",
                f"```python\n",
                f".withColumn('acquisition_channel', coalesce(col('channel'), lit('unknown')))\n",
                f"# Handles null channels with default value\n",
                f"```\n\n",
                f"**5. Deduplication:**\n",
                f"```python\n",
                f".dropDuplicates(['customer_id', 'event_type', 'event_date'])\n",
                f"# Removes duplicate events for same customer on same day\n",
                f"```\n\n",
                f"💡 **Pro Tip:** Always measure data loss during cleaning to ensure you're not removing valid business data.\n\n"
            ]
        }
        cells.append(cleaning_explanation_cell)
        
        # Metrics calculation cell
        metrics_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-metrics"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Customer Acquisition Metrics Calculation\n",
                "# Calculate CAC, LTV, and conversion metrics for Amazon's customer acquisition platform\n",
                "\n",
                "try:\n",
                "    logger.info('📊 Calculating customer acquisition metrics')\n",
                "    \n",
                "    # Calculate Customer Acquisition Cost (CAC) by channel and time period\n",
                "    df_cac_metrics = df_cleaned \\\n",
                "        .filter(col('event_type') == 'acquisition') \\\n",
                "        .groupBy('acquisition_channel', 'clean_campaign_id', 'year_month') \\\n",
                "        .agg(\n",
                "            sum('marketing_spend').alias('total_marketing_spend'),\n",
                "            countDistinct('customer_id').alias('customers_acquired'),\n",
                "            avg('marketing_spend').alias('avg_spend_per_acquisition')\n",
                "        ) \\\n",
                "        .withColumn('cac', \n",
                "            when(col('customers_acquired') > 0, \n",
                "                 col('total_marketing_spend') / col('customers_acquired')\n",
                "            ).otherwise(0)\n",
                "        ) \\\n",
                "        .withColumn('calculated_timestamp', current_timestamp())\n",
                "    \n",
                "    # Calculate Customer Lifetime Value (LTV) metrics\n",
                "    df_ltv_metrics = df_cleaned \\\n",
                "        .filter(col('event_type') == 'purchase') \\\n",
                "        .groupBy('customer_id', 'acquisition_channel') \\\n",
                "        .agg(\n",
                "            sum('revenue').alias('total_revenue'),\n",
                "            count('*').alias('purchase_frequency'),\n",
                "            avg('revenue').alias('avg_order_value'),\n",
                "            min('event_date').alias('first_purchase_date'),\n",
                "            max('event_date').alias('last_purchase_date')\n",
                "        ) \\\n",
                "        .withColumn('customer_tenure_days',\n",
                "            greatest(datediff(col('last_purchase_date'), col('first_purchase_date')), lit(1))\n",
                "        ) \\\n",
                "        .withColumn('ltv_12_month_estimate', \n",
                "            col('total_revenue') * (365.0 / col('customer_tenure_days'))\n",
                "        ) \\\n",
                "        .withColumn('customer_segment',\n",
                "            when(col('total_revenue') >= 1000, 'high_value')\n",
                "            .when(col('total_revenue') >= 500, 'medium_value')\n",
                "            .otherwise('low_value')\n",
                "        )\n",
                "    \n",
                "    # Calculate conversion rates by channel and time period\n",
                "    df_conversion_metrics = df_cleaned \\\n",
                "        .groupBy('acquisition_channel', 'year_month') \\\n",
                "        .agg(\n",
                "            countDistinct('customer_id').alias('total_customers'),\n",
                "            countDistinct(\n",
                "                when(col('event_type') == 'purchase', col('customer_id'))\n",
                "            ).alias('converted_customers')\n",
                "        ) \\\n",
                "        .withColumn('conversion_rate_pct',\n",
                "            when(col('total_customers') > 0,\n",
                "                 (col('converted_customers') / col('total_customers')) * 100\n",
                "            ).otherwise(0)\n",
                "        )\n",
                "    \n",
                "    # Cache metrics for performance\n",
                "    df_cac_metrics.cache()\n",
                "    df_ltv_metrics.cache()\n",
                "    df_conversion_metrics.cache()\n",
                "    \n",
                "    print('✅ Customer Acquisition Metrics Calculated Successfully')\n",
                "    \n",
                "    logger.info('✅ Customer acquisition metrics calculation completed')\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Metrics calculation failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(metrics_cell)
        
        # Educational explanation for metrics calculation
        metrics_explanation_cell = {
            "cell_type": "markdown",
            "metadata": {"tags": [f"task-{task_number}-metrics-explanation"]},
            "source": [
                f"#### 📊 Customer Acquisition Metrics Deep Dive\n\n",
                f"<div style='background-color: #e8f5e8; padding: 15px; border-left: 4px solid #28a745;'>\n",
                f"<h5>💰 Business Metrics Explained</h5>\n",
                f"<p>Understanding these key customer acquisition metrics is crucial for business success:</p>\n",
                f"</div>\n\n",
                f"**1. Customer Acquisition Cost (CAC):**\n",
                f"```\n",
                f"CAC = Total Marketing Spend / Number of Customers Acquired\n",
                f"```\n",
                f"- Measures efficiency of marketing investments\n",
                f"- Lower CAC = more efficient acquisition\n",
                f"- Should be tracked by channel and campaign\n\n",
                f"**2. Customer Lifetime Value (LTV):**\n",
                f"```\n",
                f"LTV = Total Revenue × (365 / Customer Tenure Days)\n",
                f"```\n",
                f"- Estimates total revenue from a customer over 12 months\n",
                f"- Higher LTV = more valuable customers\n",
                f"- Used to justify marketing investments\n\n",
                f"**3. LTV/CAC Ratio:**\n",
                f"```\n",
                f"LTV/CAC Ratio = Average LTV / Average CAC\n",
                f"```\n",
                f"- **> 3.0:** Healthy acquisition channel\n",
                f"- **< 1.0:** Losing money on acquisitions\n",
                f"- **Optimal:** Between 3.0 and 5.0\n\n",
                f"**4. Conversion Rate:**\n",
                f"```\n",
                f"Conversion Rate = (Converted Customers / Total Customers) × 100\n",
                f"```\n",
                f"- Measures funnel efficiency\n",
                f"- Higher conversion = better targeting\n\n",
                f"### 🎯 Advanced PySpark Aggregation Patterns\n",
                f"```python\n",
                f"# Window functions for cohort analysis\n",
                f".withColumn('customer_rank', row_number().over(Window.partitionBy('customer_id').orderBy('event_date')))\n\n",
                f"# Conditional aggregations\n",
                f"countDistinct(when(col('event_type') == 'purchase', col('customer_id')))\n\n",
                f"# Percentile calculations\n",
                f"percentile_approx('revenue', 0.5).alias('median_revenue')\n",
                f"```\n\n"
            ]
        }
        cells.append(metrics_explanation_cell)
        
        # Results display cell
        results_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-results"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Display Transformation Results\n",
                "# Show calculated metrics and key insights\n",
                "\n",
                "print('📊 CUSTOMER ACQUISITION COST (CAC) BY CHANNEL:')\n",
                "print('=' * 60)\n",
                "df_cac_metrics.orderBy(desc('cac')).show(20, truncate=False)\n",
                "\n",
                "print('\\n📈 CUSTOMER LIFETIME VALUE (LTV) SUMMARY:')\n",
                "print('=' * 60)\n",
                "df_ltv_metrics.agg(\n",
                "    count('customer_id').alias('total_customers'),\n",
                "    avg('total_revenue').alias('avg_ltv'),\n",
                "    percentile_approx('total_revenue', 0.5).alias('median_ltv'),\n",
                "    max('total_revenue').alias('max_ltv'),\n",
                "    avg('purchase_frequency').alias('avg_purchases_per_customer')\n",
                ").show(truncate=False)\n",
                "\n",
                "print('\\n🎯 CONVERSION RATES BY CHANNEL:')\n",
                "print('=' * 60)\n",
                "df_conversion_metrics.orderBy(desc('conversion_rate_pct')).show(20, truncate=False)\n",
                "\n",
                "print('\\n💰 LTV/CAC RATIO BY CHANNEL (Key Business Metric):')\n",
                "print('=' * 60)\n",
                "# Calculate LTV/CAC ratio for business insights\n",
                "ltv_by_channel = df_ltv_metrics.groupBy('acquisition_channel').agg(\n",
                "    avg('ltv_12_month_estimate').alias('avg_ltv_12m')\n",
                ")\n",
                "\n",
                "cac_by_channel = df_cac_metrics.groupBy('acquisition_channel').agg(\n",
                "    avg('cac').alias('avg_cac')\n",
                ")\n",
                "\n",
                "ltv_cac_ratio = ltv_by_channel.join(cac_by_channel, 'acquisition_channel', 'inner') \\\n",
                "    .withColumn('ltv_cac_ratio', \n",
                "        when(col('avg_cac') > 0, col('avg_ltv_12m') / col('avg_cac')).otherwise(0)\n",
                "    ) \\\n",
                "    .orderBy(desc('ltv_cac_ratio'))\n",
                "\n",
                "ltv_cac_ratio.show(truncate=False)\n",
                "\n",
                "print('\\n📋 Business Insights:')\n",
                "print('   • LTV/CAC ratio > 3.0 indicates healthy acquisition channels')\n",
                "print('   • Focus investment on channels with highest LTV/CAC ratios')\n",
                "print('   • Monitor conversion rates and optimize underperforming channels')"
            ]
        }
        cells.append(results_cell)
        
        return cells

    def _generate_validation_cells(self, task: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Generate cells specifically for data validation tasks"""
        cells = []
        
        validation_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-validation"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Data Quality Validation\n",
                "# Comprehensive validation for customer acquisition data\n",
                "\n",
                "try:\n",
                "    logger.info('🔍 Starting comprehensive data quality validation')\n",
                "    \n",
                "    # Data quality validation suite\n",
                "    validation_results = {}\n",
                "    \n",
                "    # 1. Completeness checks\n",
                "    total_records = df_cleaned.count()\n",
                "    null_customer_ids = df_cleaned.filter(col('customer_id').isNull()).count()\n",
                "    null_event_dates = df_cleaned.filter(col('event_date').isNull()).count()\n",
                "    \n",
                "    validation_results['completeness'] = {\n",
                "        'total_records': total_records,\n",
                "        'null_customer_ids': null_customer_ids,\n",
                "        'null_event_dates': null_event_dates,\n",
                "        'completeness_rate': ((total_records - null_customer_ids) / total_records) * 100\n",
                "    }\n",
                "    \n",
                "    # 2. Validity checks\n",
                "    negative_revenues = df_cleaned.filter(col('revenue') < 0).count()\n",
                "    future_events = df_cleaned.filter(col('event_date') > current_timestamp()).count()\n",
                "    invalid_channels = df_cleaned.filter(col('acquisition_channel').isin(['', ' ', 'null'])).count()\n",
                "    \n",
                "    validation_results['validity'] = {\n",
                "        'negative_revenues': negative_revenues,\n",
                "        'future_events': future_events,\n",
                "        'invalid_channels': invalid_channels\n",
                "    }\n",
                "    \n",
                "    # 3. Consistency checks\n",
                "    duplicate_events = df_cleaned.count() - df_cleaned.dropDuplicates(['customer_id', 'event_type', 'event_date']).count()\n",
                "    \n",
                "    validation_results['consistency'] = {\n",
                "        'duplicate_events': duplicate_events\n",
                "    }\n",
                "    \n",
                "    # 4. Business rule validations\n",
                "    acquisition_without_spend = df_cleaned.filter(\n",
                "        (col('event_type') == 'acquisition') & \n",
                "        ((col('marketing_spend').isNull()) | (col('marketing_spend') <= 0))\n",
                "    ).count()\n",
                "    \n",
                "    purchases_without_revenue = df_cleaned.filter(\n",
                "        (col('event_type') == 'purchase') & \n",
                "        ((col('revenue').isNull()) | (col('revenue') <= 0))\n",
                "    ).count()\n",
                "    \n",
                "    validation_results['business_rules'] = {\n",
                "        'acquisition_without_spend': acquisition_without_spend,\n",
                "        'purchases_without_revenue': purchases_without_revenue\n",
                "    }\n",
                "    \n",
                "    print('🔍 DATA QUALITY VALIDATION REPORT')\n",
                "    print('=' * 60)\n",
                "    \n",
                "    print(f'\\n📊 COMPLETENESS METRICS:')\n",
                "    comp = validation_results['completeness']\n",
                "    print(f'   Total Records: {comp[\"total_records\"]:,}')\n",
                "    print(f'   Null Customer IDs: {comp[\"null_customer_ids\"]:,}')\n",
                "    print(f'   Completeness Rate: {comp[\"completeness_rate\"]:.2f}%')\n",
                "    \n",
                "    print(f'\\n✅ VALIDITY CHECKS:')\n",
                "    val = validation_results['validity']\n",
                "    print(f'   Negative Revenues: {val[\"negative_revenues\"]:,}')\n",
                "    print(f'   Future Events: {val[\"future_events\"]:,}')\n",
                "    print(f'   Invalid Channels: {val[\"invalid_channels\"]:,}')\n",
                "    \n",
                "    print(f'\\n🔄 CONSISTENCY CHECKS:')\n",
                "    cons = validation_results['consistency']\n",
                "    print(f'   Duplicate Events: {cons[\"duplicate_events\"]:,}')\n",
                "    \n",
                "    print(f'\\n💼 BUSINESS RULE VALIDATION:')\n",
                "    biz = validation_results['business_rules']\n",
                "    print(f'   Acquisitions without Marketing Spend: {biz[\"acquisition_without_spend\"]:,}')\n",
                "    print(f'   Purchases without Revenue: {biz[\"purchases_without_revenue\"]:,}')\n",
                "    \n",
                "    # Overall data quality score\n",
                "    quality_score = (\n",
                "        validation_results['completeness']['completeness_rate'] * 0.4 +\n",
                "        (100 - (val['negative_revenues'] + val['future_events']) / total_records * 100) * 0.3 +\n",
                "        (100 - cons['duplicate_events'] / total_records * 100) * 0.3\n",
                "    )\n",
                "    \n",
                "    print(f'\\n🏆 OVERALL DATA QUALITY SCORE: {quality_score:.1f}/100')\n",
                "    \n",
                "    if quality_score >= 95:\n",
                "        print('   ✅ EXCELLENT - Data ready for production analytics')\n",
                "    elif quality_score >= 85:\n",
                "        print('   ⚠️  GOOD - Minor quality issues detected')\n",
                "    else:\n",
                "        print('   ❌ POOR - Significant data quality issues require attention')\n",
                "    \n",
                "    logger.info(f'✅ Data quality validation completed - Score: {quality_score:.1f}/100')\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data validation failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(validation_cell)
        
        return cells

    def _generate_output_cells(self, task: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Generate cells specifically for data output tasks"""
        cells = []
        
        output_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-output"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: Save Results to Delta Lake\n",
                "# Persist customer acquisition metrics for analytics and reporting\n",
                "\n",
                "try:\n",
                "    logger.info('💾 Starting data output to Delta Lake')\n",
                "    \n",
                "    # Output configuration\n",
                "    output_config = {\n",
                "        'cac_table_path': s3_config['delta_path'] + 'cac_metrics/',\n",
                "        'ltv_table_path': s3_config['delta_path'] + 'ltv_metrics/', \n",
                "        'conversion_table_path': s3_config['delta_path'] + 'conversion_metrics/'\n",
                "    }\n",
                "    \n",
                "    # Write CAC metrics to Delta table with partitioning\n",
                "    print('📊 Saving Customer Acquisition Cost (CAC) metrics...')\n",
                "    df_cac_metrics.write \\\n",
                "        .format('delta') \\\n",
                "        .mode('overwrite') \\\n",
                "        .option('mergeSchema', 'true') \\\n",
                "        .partitionBy('acquisition_channel', 'year_month') \\\n",
                "        .save(output_config['cac_table_path'])\n",
                "    \n",
                "    # Write LTV metrics to Delta table\n",
                "    print('📈 Saving Customer Lifetime Value (LTV) metrics...')\n",
                "    df_ltv_metrics.write \\\n",
                "        .format('delta') \\\n",
                "        .mode('overwrite') \\\n",
                "        .option('mergeSchema', 'true') \\\n",
                "        .partitionBy('acquisition_channel', 'customer_segment') \\\n",
                "        .save(output_config['ltv_table_path'])\n",
                "    \n",
                "    # Write conversion metrics to Delta table\n",
                "    print('🎯 Saving conversion rate metrics...')\n",
                "    df_conversion_metrics.write \\\n",
                "        .format('delta') \\\n",
                "        .mode('overwrite') \\\n",
                "        .option('mergeSchema', 'true') \\\n",
                "        .partitionBy('acquisition_channel') \\\n",
                "        .save(output_config['conversion_table_path'])\n",
                "    \n",
                "    # Register tables in Databricks metastore for easy access\n",
                "    print('🗂️  Registering tables in Databricks metastore...')\n",
                "    \n",
                "    spark.sql(f\"\"\"CREATE TABLE IF NOT EXISTS customer_acquisition.cac_metrics\n",
                "        USING DELTA\n",
                "        LOCATION '{output_config['cac_table_path']}'\"\"\")\n",
                "    \n",
                "    spark.sql(f\"\"\"CREATE TABLE IF NOT EXISTS customer_acquisition.ltv_metrics\n",
                "        USING DELTA\n",
                "        LOCATION '{output_config['ltv_table_path']}'\"\"\")\n",
                "    \n",
                "    spark.sql(f\"\"\"CREATE TABLE IF NOT EXISTS customer_acquisition.conversion_metrics\n",
                "        USING DELTA\n",
                "        LOCATION '{output_config['conversion_table_path']}'\"\"\")\n",
                "    \n",
                "    # Generate summary statistics for validation\n",
                "    cac_records = spark.read.format('delta').load(output_config['cac_table_path']).count()\n",
                "    ltv_records = spark.read.format('delta').load(output_config['ltv_table_path']).count()\n",
                "    conversion_records = spark.read.format('delta').load(output_config['conversion_table_path']).count()\n",
                "    \n",
                "    print('✅ DATA OUTPUT COMPLETED SUCCESSFULLY')\n",
                "    print('=' * 50)\n",
                "    print(f'📊 CAC Metrics Records: {cac_records:,}')\n",
                "    print(f'📈 LTV Metrics Records: {ltv_records:,}')\n",
                "    print(f'🎯 Conversion Metrics Records: {conversion_records:,}')\n",
                "    print(f'💾 Total Records Saved: {cac_records + ltv_records + conversion_records:,}')\n",
                "    \n",
                "    print('\\n🔗 Tables Available in Databricks:')\n",
                "    print('   • customer_acquisition.cac_metrics')\n",
                "    print('   • customer_acquisition.ltv_metrics')\n",
                "    print('   • customer_acquisition.conversion_metrics')\n",
                "    \n",
                "    logger.info('✅ Data output to Delta Lake completed successfully')\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data output failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(output_cell)
        
        return cells

    def _generate_generic_processing_cells(self, task: Dict[str, Any], task_number: int) -> List[Dict[str, Any]]:
        """Generate cells for generic processing tasks"""
        cells = []
        
        processing_cell = {
            "cell_type": "code",
            "metadata": {"tags": [f"task-{task_number}-processing"]},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# Task {task_number}: {task.get('title', 'Data Processing')}\n",
                f"# {task.get('description', 'Generic data processing for customer acquisition analytics')}\n",
                "\n",
                "try:\n",
                "    logger.info(f'🔄 Starting task: {task.get(\"description\", \"Data processing\")}')\n",
                "    \n",
                "    # Generic processing based on available data\n",
                "    if 'df_cleaned' in locals():\n",
                "        df_processed = df_cleaned\n",
                "    elif 'df_raw_events' in locals():\n",
                "        df_processed = df_raw_events\n",
                "    else:\n",
                "        raise ValueError('No source dataframe available for processing')\n",
                "    \n",
                "    # Add common processing steps\n",
                "    df_task_result = df_processed \\\n",
                "        .filter(col('customer_id').isNotNull()) \\\n",
                "        .withColumn('processed_timestamp', current_timestamp()) \\\n",
                "        .withColumn('processing_batch_id', lit(f'task_{task_number}_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}'))\n",
                "    \n",
                "    # Show results\n",
                "    record_count = df_task_result.count()\n",
                "    \n",
                "    print(f'✅ Task {task_number} Processing Completed:')\n",
                "    print(f'   📊 Records Processed: {record_count:,}')\n",
                "    print(f'   📋 Task Description: {task.get(\"description\", \"Generic processing\")}')\n",
                "    print(f'   🏷️  Task Type: {task.get(\"task_type\", \"processing\")}')\n",
                "    \n",
                "    # Display sample results\n",
                "    print(f'\\n📋 Sample Results:')\n",
                "    df_task_result.select('customer_id', 'event_type', 'processed_timestamp').show(5)\n",
                "    \n",
                "    logger.info(f'✅ Task {task_number} completed - {record_count:,} records processed')\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Task {task_number} failed: {str(e)}')\n",
                "    raise"
            ]
        }
        cells.append(processing_cell)
        
        return cells
    
    def _generate_task_code(self, task: Dict[str, Any], user_story: Dict[str, Any]) -> List[str]:
        """Generate specific PySpark code based on task type and context"""
        
        task_type = task.get('task_type', 'transformation')
        task_desc = task.get('description', '')
        
        if 'ingestion' in task_type.lower() or 'load' in task_desc.lower():
            return [
                f"# {task.get('title', 'Data Ingestion')}\n",
                "# Load data from S3 into Delta Lake for customer acquisition analytics\n",
                "\n",
                "try:\n",
                "    # Configure S3 paths for customer data\n",
                "    s3_bucket = 'amazon-customer-acquisition-data'\n",
                "    raw_data_path = f's3a://{s3_bucket}/raw/customer_events/'\n",
                "    delta_table_path = f's3a://{s3_bucket}/delta/customer_acquisition/'\n",
                "    \n",
                "    logger.info(f'🔄 Starting data ingestion from {raw_data_path}')\n",
                "    \n",
                "    # Define schema for better performance and data quality\n",
                "    customer_schema = StructType([\n",
                "        StructField('customer_id', StringType(), False),\n",
                "        StructField('event_type', StringType(), False),\n",
                "        StructField('event_date', TimestampType(), False),\n",
                "        StructField('channel', StringType(), True),\n",
                "        StructField('campaign_id', StringType(), True),\n",
                "        StructField('marketing_spend', DoubleType(), True),\n",
                "        StructField('revenue', DoubleType(), True)\n",
                "    ])\n",
                "    \n",
                "    # Read raw customer event data with schema\n",
                "    df_raw = spark.read \\\n",
                "        .option('header', 'true') \\\n",
                "        .schema(customer_schema) \\\n",
                "        .csv(raw_data_path)\n",
                "    \n",
                "    # Cache for better performance in subsequent operations\n",
                "    df_raw.cache()\n",
                "    \n",
                "    record_count = df_raw.count()\n",
                "    logger.info(f'📥 Successfully loaded {record_count:,} customer event records')\n",
                "    \n",
                "    print(f'📥 Loaded {record_count:,} customer event records')\n",
                "    print('📋 Schema:')\n",
                "    df_raw.printSchema()\n",
                "    print('📊 Sample data:')\n",
                "    df_raw.show(5, truncate=False)\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data ingestion failed: {str(e)}')\n",
                "    raise"
            ]
        elif 'transformation' in task_type.lower() or 'transform' in task_desc.lower():
            return [
                f"# {task.get('title', 'Data Transformation')}\n", 
                "# Transform data for customer acquisition metrics (CAC, LTV, conversion rates)\n",
                "\n",
                "try:\n",
                "    logger.info('🔄 Starting data transformations for customer acquisition metrics')\n",
                "    \n",
                "    # Data cleaning and preprocessing\n",
                "    df_cleaned = df_raw \\\n",
                "        .filter(col('customer_id').isNotNull()) \\\n",
                "        .filter(col('event_date').isNotNull()) \\\n",
                "        .filter(col('revenue') >= 0) \\\n",
                "        .withColumn('year_month', date_format(col('event_date'), 'yyyy-MM')) \\\n",
                "        .withColumn('acquisition_channel', coalesce(col('channel'), lit('unknown')))\n",
                "    \n",
                "    # Calculate Customer Acquisition Cost (CAC) by channel and campaign\n",
                "    df_cac = df_cleaned \\\n",
                "        .filter(col('event_type') == 'acquisition') \\\n",
                "        .groupBy('acquisition_channel', 'campaign_id', 'year_month') \\\n",
                "        .agg(\n",
                "            sum('marketing_spend').alias('total_marketing_spend'),\n",
                "            countDistinct('customer_id').alias('customers_acquired'),\n",
                "            avg('marketing_spend').alias('avg_spend_per_customer')\n",
                "        ) \\\n",
                "        .withColumn('cac', \n",
                "            when(col('customers_acquired') > 0, \n",
                "                 col('total_marketing_spend') / col('customers_acquired')\n",
                "            ).otherwise(0)\n",
                "        ) \\\n",
                "        .withColumn('calculated_date', current_timestamp())\n",
                "    \n",
                "    # Calculate Customer Lifetime Value (LTV) metrics\n",
                "    df_ltv = df_cleaned \\\n",
                "        .filter(col('event_type') == 'purchase') \\\n",
                "        .groupBy('customer_id', 'acquisition_channel') \\\n",
                "        .agg(\n",
                "            sum('revenue').alias('total_revenue'),\n",
                "            count('*').alias('purchase_frequency'),\n",
                "            avg('revenue').alias('avg_order_value'),\n",
                "            min('event_date').alias('first_purchase_date'),\n",
                "            max('event_date').alias('last_purchase_date')\n",
                "        ) \\\n",
                "        .withColumn('customer_tenure_days',\n",
                "            datediff(col('last_purchase_date'), col('first_purchase_date'))\n",
                "        ) \\\n",
                "        .withColumn('ltv_estimate', \n",
                "            col('total_revenue') * (col('customer_tenure_days') / 365.0)\n",
                "        )\n",
                "    \n",
                "    # Calculate conversion rates by channel\n",
                "    df_conversion = df_cleaned \\\n",
                "        .groupBy('acquisition_channel', 'year_month') \\\n",
                "        .agg(\n",
                "            countDistinct('customer_id').alias('total_customers'),\n",
                "            countDistinct(\n",
                "                when(col('event_type') == 'purchase', col('customer_id'))\n",
                "            ).alias('converted_customers')\n",
                "        ) \\\n",
                "        .withColumn('conversion_rate',\n",
                "            when(col('total_customers') > 0,\n",
                "                 (col('converted_customers') / col('total_customers')) * 100\n",
                "            ).otherwise(0)\n",
                "        )\n",
                "    \n",
                "    # Cache results for performance\n",
                "    df_cac.cache()\n",
                "    df_ltv.cache()\n",
                "    df_conversion.cache()\n",
                "    \n",
                "    logger.info('✅ Customer acquisition metrics transformation completed')\n",
                "    \n",
                "    print('📊 Customer Acquisition Cost (CAC) by Channel:')\n",
                "    df_cac.orderBy(desc('cac')).show(10, truncate=False)\n",
                "    \n",
                "    print('📈 Customer Lifetime Value (LTV) Summary:')\n",
                "    df_ltv.agg(\n",
                "        avg('total_revenue').alias('avg_ltv'),\n",
                "        max('total_revenue').alias('max_ltv'),\n",
                "        count('customer_id').alias('total_customers')\n",
                "    ).show()\n",
                "    \n",
                "    print('🎯 Conversion Rates by Channel:')\n",
                "    df_conversion.orderBy(desc('conversion_rate')).show(10, truncate=False)\n",
                "    \n",
                "except Exception as e:\n",
                "    logger.error(f'❌ Data transformation failed: {str(e)}')\n",
                "    raise"
            ]
        elif 'validation' in task_type.lower() or 'quality' in task_desc.lower():
            return [
                f"# {task.get('title', 'Data Quality Validation')}\n",
                "# Validate data quality for customer acquisition analytics\n\n",
                "# Data quality checks for production readiness\n",
                "total_records = df_raw.count()\n",
                "null_customer_ids = df_raw.filter(col('customer_id').isNull()).count()\n",
                "duplicate_records = df_raw.count() - df_raw.dropDuplicates().count()\n",
                "invalid_revenues = df_raw.filter(col('revenue') < 0).count()\n\n",
                "# Quality metrics\n",
                "quality_metrics = {\n",
                "    'total_records': total_records,\n",
                "    'null_customer_ids': null_customer_ids,\n", 
                "    'duplicate_records': duplicate_records,\n",
                "    'invalid_revenues': invalid_revenues,\n",
                "    'data_completeness': ((total_records - null_customer_ids) / total_records) * 100\n",
                "}\n\n",
                "print('🔍 Data Quality Report:')\n",
                "for metric, value in quality_metrics.items():\n",
                "    print(f'  {metric}: {value}')\n\n",
                "# Alert if data quality issues detected\n",
                "if quality_metrics['data_completeness'] < 95:\n",
                "    print('⚠️  WARNING: Data completeness below threshold!')"
            ]
        elif 'output' in task_type.lower() or 'save' in task_desc.lower():
            return [
                f"# {task.get('title', 'Data Output')}\n",
                "# Save processed data to Delta Lake for customer acquisition dashboard\n\n",
                "# Write CAC metrics to Delta table\n",
                "df_cac.write \\\n",
                "    .format('delta') \\\n",
                "    .mode('overwrite') \\\n",
                "    .option('mergeSchema', 'true') \\\n",
                "    .save(f'{delta_table_path}/customer_acquisition_cost/')\n\n",
                "# Write LTV metrics to Delta table\n",
                "df_ltv.write \\\n",
                "    .format('delta') \\\n",
                "    .mode('overwrite') \\\n",
                "    .option('mergeSchema', 'true') \\\n",
                "    .save(f'{delta_table_path}/customer_lifetime_value/')\n\n",
                "# Register tables in Databricks metastore\n",
                "spark.sql(f\"\"\"CREATE TABLE IF NOT EXISTS customer_acquisition.cac_metrics\n",
                "    USING DELTA\n",
                "    LOCATION '{delta_table_path}/customer_acquisition_cost/'\"\"\")\n\n",
                "spark.sql(f\"\"\"CREATE TABLE IF NOT EXISTS customer_acquisition.ltv_metrics\n",
                "    USING DELTA  \n",
                "    LOCATION '{delta_table_path}/customer_lifetime_value/'\"\"\")\n\n",
                "print('💾 Data successfully saved to Delta Lake')\n",
                "print('✅ Tables registered in Databricks metastore')"
            ]
        else:
            # Default generic processing code
            return [
                f"# {task.get('title', 'Data Processing')}\n",
                f"# {task_desc}\n\n",
                "# Generic data processing for customer acquisition analytics\n",
                "df_processed = df_raw \\\n",
                "    .filter(col('customer_id').isNotNull()) \\\n",
                "    .withColumn('processed_date', current_timestamp()) \\\n",
                "    .withColumn('year_month', date_format(col('event_date'), 'yyyy-MM'))\n\n",
                "# Show processing results\n",
                "print(f'📋 Processed {df_processed.count():,} records')\n",
                "df_processed.groupBy('year_month').count().orderBy('year_month').show()"
            ]

    def generate_notebook(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete PySpark notebook for the given task

        Args:
            task_data: Task information and requirements

        Returns:
            Dictionary containing notebook content and metadata
        """
        try:
            # Initialize state
            initial_state = PySparkCodingState(
                task_data=task_data,
                code_structure={},
                notebook_sections=[],
                generated_code={},
                test_cases=[],
                documentation={},
                error_messages=[],
                current_step="start"
            )

            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            # Create notebook structure
            notebook_content = {
                "cells": final_state["notebook_sections"],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }

            return {
                "success": True,
                "notebook": notebook_content,
                "generated_code": final_state["generated_code"],
                "documentation": final_state["documentation"],
                "errors": final_state.get("error_messages", [])
            }

        except Exception as e:
            self.logger.error(f"Notebook generation failed: {str(e)}")

            # Return minimal fallback notebook
            return {
                "success": False,
                "error": str(e),
                "notebook": {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": [f"# ETL Pipeline: {task_data.get('title', 'Unknown Task')}"]
                        },
                        {
                            "cell_type": "code",
                            "source": ["# Notebook generation failed. Please implement manually."],
                            "execution_count": None,
                            "outputs": []
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 4
                }
            }


# Factory function for backward compatibility
def create_pyspark_coding_agent(config: Optional[AgentConfig] = None) -> LangGraphPySparkCodingAgent:
    """Create a LangGraphPySparkCodingAgent instance"""
    return LangGraphPySparkCodingAgent(config)


if __name__ == "__main__":
    # Test the agent
    agent = LangGraphPySparkCodingAgent()

    # Sample task for testing
    test_task = {
        "title": "Customer Data Processing Pipeline",
        "description": """
Create a PySpark pipeline to process customer data:
1. Load customer CSV files
2. Clean and validate data
3. Calculate customer metrics
4. Generate insights using aggregations
5. Save processed data to Parquet format
""",
        "requirements": {
            "data_sources": ["customer_data.csv", "transaction_data.csv"],
            "transformations": ["Data cleaning", "Metric calculations", "Aggregations"],
            "outputs": ["Processed Parquet files", "Summary reports"]
        }
    }

    print("🧪 Testing PySpark Coding Agent with Azure OpenAI + LangGraph")
    result = agent.generate_notebook(test_task)

    print(f"\n📊 Results:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Notebook cells: {len(result['notebook']['cells'])}")
        print(f"Code sections: {len(result['generated_code'])}")
    print("✅ PySpark notebook generation test completed")
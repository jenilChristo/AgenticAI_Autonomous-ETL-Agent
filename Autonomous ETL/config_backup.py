"""
Configuration management for the Autonomous ETL Agent System
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
 """LLM provider configuration - Azure OpenAI as primary with best coding models"""
 provider: str = "azure_openai" # Primary: Azure OpenAI
 model: str = "gpt-4" # Best coding model
 temperature: float = 0.1
 max_tokens: int = 8192
 timeout: int = 60
 api_key: Optional[str] = "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5"

 # Azure OpenAI specific settings
 azure_endpoint: Optional[str] = "https://azureopenaijenil.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
 azure_api_version: str = "2024-12-01-preview"
 azure_deployment_name: str = "gpt-4.1" # Change this to your deployment name

# LangSmith configuration removed - no tracing needed

@dataclass
class GitHubConfig:
 """GitHub integration configuration"""
 token: Optional[str] = None
 owner: Optional[str] = None
 repo: Optional[str] = None
 base_url: str = "https://api.github.com"
 timeout: int = 30

@dataclass
class NotebookConfig:
 """Notebook generation configuration"""
 output_format: str = "ipynb"
 include_markdown_cells: bool = True
 include_test_cells: bool = True
 add_documentation: bool = True
 pyspark_version: str = "3.4.0"
 python_version: str = "3.8+"

@dataclass
class AgentConfig:
 """Main configuration class for the agent system"""
 llm: LLMConfig = field(default_factory=LLMConfig)
 github: GitHubConfig = field(default_factory=GitHubConfig)
 notebook: NotebookConfig = field(default_factory=NotebookConfig)
 output_dir: str = "./generated_pipelines"
 debug_mode: bool = False

 def __post_init__(self):
 """Initialize configuration from environment variables"""
 # LLM Configuration - Default to Azure OpenAI with your credentials
 provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
 self.llm.provider = provider

 if provider == "azure_openai":
 # Use your Azure OpenAI credentials by default
 self.llm.api_key = os.getenv("AZURE_OPENAI_API_KEY", "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5")
 # Clean endpoint - remove deployment path and query params for LangChain
 base_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azureopenaijenil.openai.azure.com/")
 if "/openai/deployments/" in base_endpoint:
 base_endpoint = base_endpoint.split("/openai/deployments/")[0] + "/"
 self.llm.azure_endpoint = base_endpoint
 self.llm.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
 self.llm.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
 self.llm.model = self.llm.azure_deployment_name # Use deployment name as model

 print(f"INFO: Using Azure OpenAI: {self.llm.azure_endpoint}")
 print(f"INFO: Deployment: {self.llm.azure_deployment_name}")

 # Check if deployment exists (helpful warning)
 if not os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
 print(f"NOTE: Create deployment '{self.llm.azure_deployment_name}' in Azure Portal first!")

 elif provider == "anthropic":
 self.llm.api_key = os.getenv("ANTHROPIC_API_KEY")
 if not self.llm.api_key:
 raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")

 else:
 raise ValueError(f"Unsupported LLM provider: {provider}. Use 'azure_openai' or 'anthropic'")

 # GitHub Configuration
 self.github.token = os.getenv("GITHUB_TOKEN")
 self.github.owner = os.getenv("GITHUB_OWNER")
 self.github.repo = os.getenv("GITHUB_REPO")

 # Output directory
 self.output_dir = os.getenv("ETL_OUTPUT_DIR", self.output_dir)

 # Debug mode
 self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

 # Ensure output directory exists
 os.makedirs(self.output_dir, exist_ok=True)

def get_config() -> AgentConfig:
 """Get the main configuration instance"""
 return AgentConfig()

# LangSmith setup removed - no tracing functionality needed


def create_llm_client(llm_config: LLMConfig):
 """Create appropriate LLM client based on provider configuration"""

 if llm_config.provider == "anthropic":
 try:
 from langchain_anthropic import ChatAnthropic

 return ChatAnthropic(
 model=llm_config.model,
 temperature=llm_config.temperature,
 max_tokens=llm_config.max_tokens,
 api_key=llm_config.api_key
 )
 except ImportError:
 raise ImportError("langchain_anthropic package required for Anthropic provider. Install with: pip install langchain-anthropic")

 elif llm_config.provider == "azure_openai":
 try:
 from langchain_openai import AzureChatOpenAI

 return AzureChatOpenAI(
 azure_deployment=llm_config.azure_deployment_name,
 api_version=llm_config.azure_api_version,
 azure_endpoint=llm_config.azure_endpoint,
 api_key=llm_config.api_key,
 temperature=llm_config.temperature,
 max_tokens=llm_config.max_tokens
 )
 except ImportError:
 raise ImportError("langchain_openai package required for Azure OpenAI provider. Install with: pip install langchain-openai")

 else:
 raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")


def get_provider_info(llm_config: LLMConfig) -> str:
 """Get human-readable provider information"""

 if llm_config.provider == "anthropic":
 return f"Anthropic Claude ({llm_config.model})"
 elif llm_config.provider == "azure_openai":
 return f"Azure OpenAI ({llm_config.azure_deployment_name})"
 else:
 return f"Unknown provider ({llm_config.provider})"

def get_llm_config() -> Dict[str, Any]:
 """Get LLM configuration as dictionary"""
 config = get_config()
 return {
 "provider": config.llm.provider,
 "model": config.llm.model,
 "temperature": config.llm.temperature,
 "max_tokens": config.llm.max_tokens,
 "api_key": config.llm.api_key
 }

# Environment setup
if __name__ == "__main__":
 # Disable any LangChain tracing
 os.environ["LANGCHAIN_TRACING_V2"] = "false"

 config = get_config()
 print(f"Configuration loaded successfully:")
 print(f" LLM: {config.llm.provider} ({config.llm.model})")
 print(f" GitHub: {'CONFIGURED' if config.github.token else 'NOT_SET'}")
 print(f" Output: {config.output_dir}")
 print(f" Debug: {config.debug_mode}")
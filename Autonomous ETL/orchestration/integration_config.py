"""
Integration Configuration 🔧
Centralized configuration for DevOps Interface and Agent Orchestrator integration
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    host: str = "localhost"
    port: int = 5000
    url: Optional[str] = None
    health_endpoint: str = "/health"
    enabled: bool = True
    
    def __post_init__(self):
        if not self.url:
            self.url = f"http://{self.host}:{self.port}"

@dataclass
class OrchestratorConfig:
    """Configuration for Agent Orchestrator"""
    model_name: str = "gpt-4o"
    api_key: str = "FttlVCdWMspCqApwBuWYXRiiL831GHMk2BbPVY8uFH8Wmvf0JUjrJQQJ99BIACYeBjFXJ3w3AAABACOGNwY5"
    azure_endpoint: str = "https://azureopenaijenil.openai.azure.com/"
    azure_api_version: str = "2024-12-01-preview"
    azure_deployment_name: str = "gpt-4.1"
    github_token: Optional[str] = None
    github_owner: Optional[str] = None
    github_repo: Optional[str] = None

@dataclass
class IntegrationConfig:
    """Main integration configuration"""
    
    # Service configurations
    devops_interface: ServiceConfig
    orchestrator_api: ServiceConfig
    
    # Orchestrator configuration
    orchestrator: OrchestratorConfig
    
    # Integration settings
    auto_start_services: bool = True
    health_check_interval: int = 30  # seconds
    request_timeout: int = 30  # seconds
    max_retries: int = 3
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    # Development settings
    debug_mode: bool = True
    hot_reload: bool = True
    
    @classmethod
    def default(cls) -> 'IntegrationConfig':
        """Create default configuration"""
        return cls(
            devops_interface=ServiceConfig(
                name="DevOps Interface",
                host="localhost",
                port=5000,
                health_endpoint="/api/orchestrator/health"
            ),
            orchestrator_api=ServiceConfig(
                name="Agent Orchestrator API",
                host="localhost", 
                port=8001,
                health_endpoint="/api/orchestrator/health"
            ),
            orchestrator=OrchestratorConfig(
                github_token=os.getenv("GITHUB_TOKEN"),
                github_owner=os.getenv("GITHUB_OWNER", "your-org"),
                github_repo=os.getenv("GITHUB_REPO", "jenilChristo/AgenticAI_Autonomous-ETL-Agent")
            )
        )
    
    @classmethod
    def from_env(cls) -> 'IntegrationConfig':
        """Create configuration from environment variables"""
        config = cls.default()
        
        # Override with environment variables if present
        if os.getenv("DEVOPS_HOST"):
            config.devops_interface.host = os.getenv("DEVOPS_HOST")
        if os.getenv("DEVOPS_PORT"):
            config.devops_interface.port = int(os.getenv("DEVOPS_PORT"))
        
        if os.getenv("ORCHESTRATOR_HOST"):
            config.orchestrator_api.host = os.getenv("ORCHESTRATOR_HOST")
        if os.getenv("ORCHESTRATOR_PORT"):
            config.orchestrator_api.port = int(os.getenv("ORCHESTRATOR_PORT"))
        
        if os.getenv("AZURE_OPENAI_API_KEY"):
            config.orchestrator.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            config.orchestrator.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if os.getenv("AZURE_DEPLOYMENT_NAME"):
            config.orchestrator.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        
        if os.getenv("INTEGRATION_DEBUG"):
            config.debug_mode = os.getenv("INTEGRATION_DEBUG").lower() == "true"
        
        # Update URLs after potential host/port changes
        config.devops_interface.url = f"http://{config.devops_interface.host}:{config.devops_interface.port}"
        config.orchestrator_api.url = f"http://{config.orchestrator_api.host}:{config.orchestrator_api.port}"
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "services": {
                "devops_interface": {
                    "name": self.devops_interface.name,
                    "url": self.devops_interface.url,
                    "host": self.devops_interface.host,
                    "port": self.devops_interface.port,
                    "enabled": self.devops_interface.enabled
                },
                "orchestrator_api": {
                    "name": self.orchestrator_api.name,
                    "url": self.orchestrator_api.url,
                    "host": self.orchestrator_api.host,
                    "port": self.orchestrator_api.port,
                    "enabled": self.orchestrator_api.enabled
                }
            },
            "orchestrator": {
                "model_name": self.orchestrator.model_name,
                "azure_endpoint": self.orchestrator.azure_endpoint,
                "azure_deployment_name": self.orchestrator.azure_deployment_name,
                "has_github_token": self.orchestrator.github_token is not None,
                "github_owner": self.orchestrator.github_owner,
                "github_repo": self.orchestrator.github_repo
            },
            "integration": {
                "auto_start_services": self.auto_start_services,
                "health_check_interval": self.health_check_interval,
                "request_timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "enable_monitoring": self.enable_monitoring,
                "debug_mode": self.debug_mode
            }
        }
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return issues"""
        issues = {
            "errors": [],
            "warnings": [],
            "valid": True
        }
        
        # Check required fields
        if not self.orchestrator.api_key:
            issues["errors"].append("Azure OpenAI API key is required")
            issues["valid"] = False
        
        if not self.orchestrator.azure_endpoint:
            issues["errors"].append("Azure OpenAI endpoint is required")
            issues["valid"] = False
        
        # Check GitHub configuration
        if not self.orchestrator.github_token:
            issues["warnings"].append("GitHub token not configured - PR creation will not work")
        
        if not self.orchestrator.github_owner or not self.orchestrator.github_repo:
            issues["warnings"].append("GitHub owner/repo not configured - using defaults")
        
        # Check port conflicts
        if self.devops_interface.port == self.orchestrator_api.port:
            issues["errors"].append("DevOps Interface and Orchestrator API cannot use the same port")
            issues["valid"] = False
        
        # Check reasonable port ranges
        for service in [self.devops_interface, self.orchestrator_api]:
            if service.port < 1024 or service.port > 65535:
                issues["warnings"].append(f"{service.name} port {service.port} may require elevated privileges or is invalid")
        
        return issues


# Global configuration instance
_config_instance = None

def get_integration_config() -> IntegrationConfig:
    """Get global configuration instance (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = IntegrationConfig.from_env()
    return _config_instance

def reload_config() -> IntegrationConfig:
    """Reload configuration from environment"""
    global _config_instance
    _config_instance = IntegrationConfig.from_env()
    return _config_instance


if __name__ == "__main__":
    # Configuration validation script
    print("🔧 Integration Configuration Validator")
    print("=" * 50)
    
    config = get_integration_config()
    validation = config.validate()
    
    print(f"Configuration Status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
    print()
    
    if validation['errors']:
        print("❌ Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
        print()
    
    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
        print()
    
    print("📊 Configuration Summary:")
    print(f"  DevOps Interface: {config.devops_interface.url}")
    print(f"  Orchestrator API: {config.orchestrator_api.url}")
    print(f"  Azure OpenAI Model: {config.orchestrator.model_name}")
    print(f"  GitHub Integration: {'✅' if config.orchestrator.github_token else '❌'}")
    print(f"  Debug Mode: {'✅' if config.debug_mode else '❌'}")
    print()
    
    if validation['valid']:
        print("🚀 Configuration is valid! Ready to start integration.")
    else:
        print("🛑 Please fix configuration errors before proceeding.")
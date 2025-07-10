"""
Configuration module for the Political Analysis Scaling project.

This module provides a simple Config class to manage environment variables
and configuration settings for the entire project.
"""

import os
from typing import Optional
from dotenv import load_dotenv


class Config:
    """
    Configuration class that loads and manages environment variables
    for the Political Analysis Scaling project.
    """
    
    def __init__(self):
        """Initialize the config by loading environment variables."""
        load_dotenv()
    
    # LLM API Keys
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key for GPT models."""
        return os.environ.get("OPENAI_API_KEY")
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key for Claude models."""
        return os.environ.get("ANTHROPIC_API_KEY")
    
    @property
    def azure_openai_endpoint(self) -> Optional[str]:
        """Azure OpenAI endpoint URL."""
        return os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    @property
    def azure_openai_key(self) -> Optional[str]:
        """Azure OpenAI API key."""
        return os.environ.get("AZURE_OPENAI_KEY")
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Google API key for Gemini models."""
        return os.environ.get("GOOGLE_API_KEY")
    
    @property
    def deepseek_api_key(self) -> Optional[str]:
        """DeepSeek API key."""
        return os.environ.get("DEEPSEEK_API_KEY")
    
    @property
    def grok_api_key(self) -> Optional[str]:
        """Grok API key."""
        return os.environ.get("GROK_API_KEY")
    
    @property
    def novita_api_key(self) -> Optional[str]:
        """Novita API key."""
        return os.environ.get("NOVITA_API_KEY")
    
    # LangSmith Configuration
    @property
    def langsmith_api_key(self) -> Optional[str]:
        """LangSmith API key for prompt management and tracing."""
        return os.environ.get("LANGSMITH_API_KEY")
    
    # Logging Configuration
    @property
    def logging_level(self) -> str:
        """Logging level for the application."""
        return os.environ.get("LOGGING_LEVEL", "info").lower()
    
    # Cost Calculation
    @property
    def calculate_cost(self) -> bool:
        """Whether to calculate and track API costs."""
        return os.environ.get("CALCULATE_COST", "false").lower() == "true"
    
    # Logging File Configuration
    @property
    def log_file_path(self) -> Optional[str]:
        """Optional file path for logging output."""
        return os.environ.get("LOG_FILE_PATH")
    
    # Utility Methods
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic', 'google')
            
        Returns:
            The API key for the provider, or None if not found
        """
        provider_map = {
            'openai': self.openai_api_key,
            'anthropic': self.anthropic_api_key,
            'azure': self.azure_openai_key,
            'google': self.google_api_key,
            'deepseek': self.deepseek_api_key,
            'grok': self.grok_api_key,
            'novita': self.novita_api_key,
            'langsmith': self.langsmith_api_key,
        }
        return provider_map.get(provider.lower())
    
    def has_api_key(self, provider: str) -> bool:
        """
        Check if API key exists for a specific provider.
        
        Args:
            provider: The provider name
            
        Returns:
            True if the API key exists and is not empty, False otherwise
        """
        api_key = self.get_api_key(provider)
        return api_key is not None and api_key.strip() != ""
    
    def get_available_providers(self) -> list[str]:
        """
        Get list of providers that have API keys configured.
        
        Returns:
            List of provider names that have valid API keys
        """
        providers = ['openai', 'anthropic', 'azure', 'google', 'deepseek', 'grok', 'novita']
        return [provider for provider in providers if self.has_api_key(provider)]
    
    def validate_required_keys(self, required_providers: list[str]) -> None:
        """
        Validate that required API keys are present.
        
        Args:
            required_providers: List of provider names that are required
            
        Raises:
            ValueError: If any required API key is missing
        """
        missing_keys = []
        for provider in required_providers:
            if not self.has_api_key(provider):
                missing_keys.append(provider)
        
        if missing_keys:
            raise ValueError(
                f"Missing required API keys for providers: {', '.join(missing_keys)}. "
                f"Please check your .env file."
            )
    
    def __repr__(self) -> str:
        """String representation of the config (without exposing API keys)."""
        available_providers = self.get_available_providers()
        return (
            f"Config(logging_level='{self.logging_level}', "
            f"calculate_cost={self.calculate_cost}, "
            f"available_providers={available_providers})"
        )


# Global config instance
config = Config()
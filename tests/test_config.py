"""
Pytest tests for the Config class.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path so we can import from scaling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling.utils.config import Config


class TestConfig:
    """Test suite for the Config class."""

    @pytest.fixture
    def mock_env(self):
        """Fixture to provide mock environment variables."""
        return {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'GOOGLE_API_KEY': 'test-google-key',
            'DEEPSEEK_API_KEY': 'test-deepseek-key',
            'GROK_API_KEY': 'test-grok-key',
            'NOVITA_API_KEY': 'test-novita-key',
            'LANGSMITH_API_KEY': 'test-langsmith-key',
            'LOGGING_LEVEL': 'debug',
            'CALCULATE_COST': 'true'
        }

    @pytest.fixture
    def empty_env(self):
        """Fixture to provide empty environment variables."""
        return {}

    @pytest.fixture
    def partial_env(self):
        """Fixture to provide partially filled environment variables."""
        return {
            'OPENAI_API_KEY': 'test-openai-key',
            'LOGGING_LEVEL': 'warning',
            'CALCULATE_COST': 'false'
        }

    @patch('scaling.utils.config.load_dotenv')
    def test_config_initialization(self, mock_load_dotenv):
        """Test that Config initializes and calls load_dotenv."""
        config = Config()
        mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch('scaling.utils.config.load_dotenv')
    def test_api_keys_with_empty_environment(self, mock_load_dotenv):
        """Test API key properties return None when environment is empty."""
        config = Config()
        
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None
        assert config.azure_openai_endpoint is None
        assert config.azure_openai_key is None
        assert config.google_api_key is None
        assert config.deepseek_api_key is None
        assert config.grok_api_key is None
        assert config.novita_api_key is None
        assert config.langsmith_api_key is None

    @patch('scaling.utils.config.load_dotenv')
    def test_api_keys_with_mock_environment(self, mock_load_dotenv, mock_env):
        """Test API key properties return correct values from environment."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            
            assert config.openai_api_key == 'test-openai-key'
            assert config.anthropic_api_key == 'test-anthropic-key'
            assert config.azure_openai_endpoint == 'https://test.openai.azure.com'
            assert config.azure_openai_key == 'test-azure-key'
            assert config.google_api_key == 'test-google-key'
            assert config.deepseek_api_key == 'test-deepseek-key'
            assert config.grok_api_key == 'test-grok-key'
            assert config.novita_api_key == 'test-novita-key'
            assert config.langsmith_api_key == 'test-langsmith-key'

    @patch('scaling.utils.config.load_dotenv')
    def test_logging_level_default(self, mock_load_dotenv):
        """Test logging level defaults to 'info' when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.logging_level == 'info'

    @patch('scaling.utils.config.load_dotenv')
    def test_logging_level_custom(self, mock_load_dotenv):
        """Test logging level returns custom value when set."""
        with patch.dict(os.environ, {'LOGGING_LEVEL': 'DEBUG'}, clear=True):
            config = Config()
            assert config.logging_level == 'debug'  # Should be lowercased

    @patch('scaling.utils.config.load_dotenv')
    def test_calculate_cost_default(self, mock_load_dotenv):
        """Test calculate_cost defaults to False when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.calculate_cost is False

    @pytest.mark.parametrize("env_value,expected", [
        ('true', True),
        ('True', True),
        ('TRUE', True),
        ('false', False),
        ('False', False),
        ('FALSE', False),
        ('yes', False),  # Only 'true' should be True
        ('1', False),    # Only 'true' should be True
        ('', False),     # Empty string should be False
    ])
    @patch('scaling.utils.config.load_dotenv')
    def test_calculate_cost_values(self, mock_load_dotenv, env_value, expected):
        """Test calculate_cost with various string values."""
        with patch.dict(os.environ, {'CALCULATE_COST': env_value}, clear=True):
            config = Config()
            assert config.calculate_cost is expected

    @patch('scaling.utils.config.load_dotenv')
    def test_get_api_key_method(self, mock_load_dotenv, mock_env):
        """Test get_api_key method returns correct values."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            
            assert config.get_api_key('openai') == 'test-openai-key'
            assert config.get_api_key('anthropic') == 'test-anthropic-key'
            assert config.get_api_key('azure') == 'test-azure-key'
            assert config.get_api_key('google') == 'test-google-key'
            assert config.get_api_key('deepseek') == 'test-deepseek-key'
            assert config.get_api_key('grok') == 'test-grok-key'
            assert config.get_api_key('novita') == 'test-novita-key'
            assert config.get_api_key('langsmith') == 'test-langsmith-key'

    @patch('scaling.utils.config.load_dotenv')
    def test_get_api_key_case_insensitive(self, mock_load_dotenv, mock_env):
        """Test get_api_key method is case insensitive."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            
            assert config.get_api_key('OPENAI') == 'test-openai-key'
            assert config.get_api_key('OpenAI') == 'test-openai-key'
            assert config.get_api_key('openai') == 'test-openai-key'

    @patch('scaling.utils.config.load_dotenv')
    def test_get_api_key_unknown_provider(self, mock_load_dotenv):
        """Test get_api_key returns None for unknown provider."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.get_api_key('unknown') is None

    @patch('scaling.utils.config.load_dotenv')
    def test_has_api_key_method(self, mock_load_dotenv, mock_env):
        """Test has_api_key method returns correct boolean values."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            
            assert config.has_api_key('openai') is True
            assert config.has_api_key('anthropic') is True
            assert config.has_api_key('unknown') is False

    @patch('scaling.utils.config.load_dotenv')
    def test_has_api_key_empty_string(self, mock_load_dotenv):
        """Test has_api_key returns False for empty string values."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
            config = Config()
            assert config.has_api_key('openai') is False

    @patch('scaling.utils.config.load_dotenv')
    def test_has_api_key_whitespace_only(self, mock_load_dotenv):
        """Test has_api_key returns False for whitespace-only values."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': '   '}, clear=True):
            config = Config()
            assert config.has_api_key('openai') is False

    @patch('scaling.utils.config.load_dotenv')
    def test_get_available_providers(self, mock_load_dotenv, partial_env):
        """Test get_available_providers returns only configured providers."""
        with patch.dict(os.environ, partial_env, clear=True):
            config = Config()
            available = config.get_available_providers()
            
            assert 'openai' in available
            assert 'anthropic' not in available
            assert 'google' not in available
            assert len(available) == 1

    @patch('scaling.utils.config.load_dotenv')
    def test_get_available_providers_empty(self, mock_load_dotenv, empty_env):
        """Test get_available_providers returns empty list when no providers configured."""
        with patch.dict(os.environ, empty_env, clear=True):
            config = Config()
            available = config.get_available_providers()
            assert available == []

    @patch('scaling.utils.config.load_dotenv')
    def test_validate_required_keys_success(self, mock_load_dotenv, mock_env):
        """Test validate_required_keys passes when all keys are present."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            
            # Should not raise any exception
            config.validate_required_keys(['openai', 'anthropic'])
            config.validate_required_keys(['openai'])
            config.validate_required_keys([])

    @patch('scaling.utils.config.load_dotenv')
    def test_validate_required_keys_failure(self, mock_load_dotenv, partial_env):
        """Test validate_required_keys raises ValueError when keys are missing."""
        with patch.dict(os.environ, partial_env, clear=True):
            config = Config()
            
            with pytest.raises(ValueError, match="Missing required API keys"):
                config.validate_required_keys(['openai', 'anthropic'])
            
            with pytest.raises(ValueError, match="anthropic"):
                config.validate_required_keys(['anthropic'])

    @patch('scaling.utils.config.load_dotenv')
    def test_validate_required_keys_multiple_missing(self, mock_load_dotenv, empty_env):
        """Test validate_required_keys lists all missing keys."""
        with patch.dict(os.environ, empty_env, clear=True):
            config = Config()
            
            with pytest.raises(ValueError) as exc_info:
                config.validate_required_keys(['openai', 'anthropic', 'google'])
            
            error_message = str(exc_info.value)
            assert 'openai' in error_message
            assert 'anthropic' in error_message
            assert 'google' in error_message

    @patch('scaling.utils.config.load_dotenv')
    def test_repr_method(self, mock_load_dotenv, partial_env):
        """Test __repr__ method returns expected format."""
        with patch.dict(os.environ, partial_env, clear=True):
            config = Config()
            repr_str = repr(config)
            
            assert 'Config(' in repr_str
            assert 'logging_level=' in repr_str
            assert 'calculate_cost=' in repr_str
            assert 'available_providers=' in repr_str
            assert 'warning' in repr_str  # From partial_env
            assert 'false' in repr_str.lower()  # From partial_env

    @patch('scaling.utils.config.load_dotenv')
    def test_repr_no_api_keys_exposed(self, mock_load_dotenv, mock_env):
        """Test __repr__ method doesn't expose actual API keys."""
        with patch.dict(os.environ, mock_env, clear=True):
            config = Config()
            repr_str = repr(config)
            
            # Should not contain actual API key values
            assert 'test-openai-key' not in repr_str
            assert 'test-anthropic-key' not in repr_str
            assert 'test-langsmith-key' not in repr_str


class TestConfigIntegration:
    """Integration tests for Config class usage patterns."""

    @patch('scaling.utils.config.load_dotenv')
    def test_realistic_usage_pattern(self, mock_load_dotenv):
        """Test realistic usage pattern for choosing LLM provider."""
        env = {
            'OPENAI_API_KEY': 'sk-test123',
            'LOGGING_LEVEL': 'info',
            'CALCULATE_COST': 'true'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            
            # Common usage pattern: choose available provider
            if config.has_api_key('openai'):
                provider = 'openai'
                api_key = config.openai_api_key
            elif config.has_api_key('anthropic'):
                provider = 'anthropic'
                api_key = config.anthropic_api_key
            else:
                provider = None
                api_key = None
            
            assert provider == 'openai'
            assert api_key == 'sk-test123'
            assert config.calculate_cost is True

    @patch('scaling.utils.config.load_dotenv')
    def test_langsmith_integration_pattern(self, mock_load_dotenv):
        """Test integration pattern for LangSmith usage."""
        env = {'LANGSMITH_API_KEY': 'lsv2_test123'}
        
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            
            # Common pattern: conditional LangSmith usage
            use_langsmith = config.has_api_key('langsmith')
            langsmith_key = config.langsmith_api_key if use_langsmith else None
            
            assert use_langsmith is True
            assert langsmith_key == 'lsv2_test123'
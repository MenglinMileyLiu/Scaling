"""
Pytest tests for the unified logging system.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling.utils.logging import (
    get_logger, 
    setup_logging, 
    get_file_logger,
    _get_module_name,
    ScalingFormatter
)


class TestModuleNaming:
    """Test module name conversion logic."""
    
    def test_get_module_name_main(self):
        """Test __main__ module name conversion."""
        assert _get_module_name("__main__") == "scaling.main"
    
    def test_get_module_name_already_scaling(self):
        """Test module names that already start with scaling."""
        assert _get_module_name("scaling.utils.config") == "scaling.utils.config"
        assert _get_module_name("scaling.pipeline.analyzer") == "scaling.pipeline.analyzer"
    
    def test_get_module_name_src_scaling(self):
        """Test module names with src.scaling prefix."""
        assert _get_module_name("src.scaling.utils.config") == "scaling.utils.config"
        assert _get_module_name("src.scaling.pipeline.analyzer") == "scaling.pipeline.analyzer"
    
    def test_get_module_name_file_path(self):
        """Test file path conversion to module names."""
        assert _get_module_name("src/scaling/utils/config.py") == "scaling.utils.config"
        assert _get_module_name("src/scaling/pipeline/analyzer.py") == "scaling.pipeline.analyzer"
    
    def test_get_module_name_simple(self):
        """Test simple module names."""
        assert _get_module_name("config") == "scaling.config"
        assert _get_module_name("analyzer") == "scaling.analyzer"
    
    def test_get_module_name_empty(self):
        """Test empty module name."""
        assert _get_module_name("") == "scaling"
    
    def test_get_module_name_with_scaling_in_name(self):
        """Test module names that contain 'scaling' but don't start with it."""
        assert _get_module_name("myproject.scaling.utils") == "scaling.utils"
        assert _get_module_name("test_scaling_module") == "scaling.module"


class TestScalingFormatter:
    """Test the custom formatter."""
    
    def test_formatter_initialization(self):
        """Test formatter initializes correctly."""
        formatter = ScalingFormatter()
        assert formatter.fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"
    
    def test_formatter_format(self):
        """Test formatter produces expected output."""
        formatter = ScalingFormatter()
        record = logging.LogRecord(
            name="scaling.utils.config",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "scaling.utils.config" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "-" in formatted  # Check timestamp format separators


class TestLoggingSetup:
    """Test logging configuration setup."""
    
    def setup_method(self):
        """Clean up logging state before each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Reset level
        root_logger.setLevel(logging.WARNING)
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_default_config(self, mock_config):
        """Test setup_logging uses config values by default."""
        mock_config.logging_level = 'debug'
        mock_config.log_file_path = None
        
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_custom_level(self, mock_config):
        """Test setup_logging with custom level."""
        mock_config.logging_level = 'info'  # Default from config
        mock_config.log_file_path = None
        
        setup_logging(level='warning')
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_with_file(self, mock_config):
        """Test setup_logging with file output."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            setup_logging(log_file=tmp_path)
            
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2  # Console + File
            
            # Check that one handler is a file handler
            file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            
        finally:
            os.unlink(tmp_path)
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_invalid_level(self, mock_config):
        """Test setup_logging with invalid level defaults to INFO."""
        mock_config.logging_level = 'invalid'
        mock_config.log_file_path = None
        
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_force_reconfigure(self, mock_config):
        """Test setup_logging can be forced to reconfigure."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        # First setup
        setup_logging()
        initial_handlers = len(logging.getLogger().handlers)
        
        # Force reconfigure
        setup_logging(force_reconfigure=True)
        
        # Should still have handlers (not doubled)
        assert len(logging.getLogger().handlers) == initial_handlers
    
    @patch('scaling.utils.logging.config')
    def test_setup_logging_invalid_file_path(self, mock_config):
        """Test setup_logging handles invalid file path gracefully."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        # Use an invalid file path
        invalid_path = "/invalid/path/that/does/not/exist/test.log"
        
        # Should not raise exception
        setup_logging(log_file=invalid_path)
        
        root_logger = logging.getLogger()
        # Should still have console handler
        assert len(root_logger.handlers) >= 1


class TestGetLogger:
    """Test get_logger function."""
    
    def setup_method(self):
        """Clean up logging state before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    @patch('scaling.utils.logging.config')
    def test_get_logger_basic(self, mock_config):
        """Test get_logger returns properly named logger."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        logger = get_logger("scaling.utils.config")
        assert logger.name == "scaling.utils.config"
        assert isinstance(logger, logging.Logger)
    
    @patch('scaling.utils.logging.config')
    def test_get_logger_module_name_conversion(self, mock_config):
        """Test get_logger converts module names correctly."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        logger = get_logger("config")
        assert logger.name == "scaling.config"
    
    @patch('scaling.utils.logging.config')
    def test_get_logger_logging_works(self, mock_config):
        """Test that logger actually logs messages."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        # Capture stdout
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            logger = get_logger("scaling.test")
            logger.info("Test message")
        
        output = captured_output.getvalue()
        assert "scaling.test" in output
        assert "Test message" in output
        assert "INFO" in output


class TestGetFileLogger:
    """Test get_file_logger function."""
    
    def setup_method(self):
        """Clean up logging state before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    @patch('scaling.utils.logging.config')
    def test_get_file_logger_creates_file(self, mock_config):
        """Test get_file_logger creates log file."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            logger = get_file_logger("scaling.test", tmp_path)
            logger.info("Test file message")
            
            # Check that file was created and has content
            assert os.path.exists(tmp_path)
            
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "scaling.test.file" in content
                assert "Test file message" in content
                
        finally:
            os.unlink(tmp_path)
    
    @patch('scaling.utils.logging.config')
    def test_get_file_logger_creates_directory(self, mock_config):
        """Test get_file_logger creates directory if it doesn't exist."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = os.path.join(tmp_dir, "logs", "test.log")
            
            logger = get_file_logger("scaling.test", log_file)
            logger.info("Test message")
            
            # Check that directory and file were created
            assert os.path.exists(os.path.dirname(log_file))
            assert os.path.exists(log_file)
    
    @patch('scaling.utils.logging.config')
    def test_get_file_logger_invalid_path_fallback(self, mock_config):
        """Test get_file_logger falls back to regular logger on invalid path."""
        mock_config.logging_level = 'info'
        mock_config.log_file_path = None
        
        invalid_path = "/invalid/path/test.log"
        
        logger = get_file_logger("scaling.test", invalid_path)
        
        # Should return a logger, but not necessarily a file logger
        assert isinstance(logger, logging.Logger)
        assert "scaling.test" in logger.name


class TestIntegration:
    """Integration tests for the logging system."""
    
    def setup_method(self):
        """Clean up logging state before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    @patch.dict(os.environ, {'LOGGING_LEVEL': 'debug', 'LOG_FILE_PATH': ''})
    def test_full_integration(self):
        """Test complete logging workflow."""
        # This test uses actual config loading
        from scaling.utils.config import config
        
        # Test that config is loaded correctly
        assert config.logging_level == 'debug'
        assert config.log_file_path is None or config.log_file_path == ''
        
        # Test logger creation and usage
        logger = get_logger("scaling.utils.test")
        assert logger.name == "scaling.utils.test"
        
        # Test that logging actually works
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
        
        output = captured_output.getvalue()
        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "scaling.utils.test" in output
    
    @patch.dict(os.environ, {'LOGGING_LEVEL': 'warning'})
    def test_level_filtering(self):
        """Test that log level filtering works correctly."""
        from scaling.utils.config import config
        
        logger = get_logger("scaling.test")
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
        
        output = captured_output.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
        assert "Error message" in output
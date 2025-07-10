"""
Example usage of the unified logging system.
This file demonstrates how to use the logging system in practice.
"""

import sys
import os
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling.utils.logging import get_logger, get_file_logger, setup_logging


def example_basic_usage():
    """Example of basic logger usage."""
    print("=== Basic Logger Usage ===")
    
    # Get a logger for this module
    logger = get_logger(__name__)
    
    # Log messages at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print()


def example_module_naming():
    """Example of how module naming works."""
    print("=== Module Naming Examples ===")
    
    # Different ways to create loggers
    logger1 = get_logger("scaling.utils.config")
    logger2 = get_logger("config")  # Will become scaling.config
    logger3 = get_logger("src.scaling.pipeline.analyzer")  # Will become scaling.pipeline.analyzer
    
    logger1.info("Message from scaling.utils.config")
    logger2.info("Message from scaling.config")
    logger3.info("Message from scaling.pipeline.analyzer")
    
    print()


def example_file_logging():
    """Example of logging to a file."""
    print("=== File Logging Example ===")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
        log_file = tmp.name
    
    try:
        # Get a file logger
        file_logger = get_file_logger("scaling.example", log_file)
        
        # Log some messages
        file_logger.info("This message goes to the file")
        file_logger.warning("This warning also goes to the file")
        file_logger.error("This error goes to the file too")
        
        # Also log to console
        console_logger = get_logger("scaling.example")
        console_logger.info("This message goes to console")
        
        print(f"Log file created at: {log_file}")
        
        # Read and display file contents
        with open(log_file, 'r') as f:
            content = f.read()
            print("File contents:")
            print(content)
            
    finally:
        # Clean up
        os.unlink(log_file)
    
    print()


def example_different_log_levels():
    """Example of different log levels."""
    print("=== Different Log Levels Example ===")
    
    # Setup logging with different levels
    print("Setting up with DEBUG level:")
    setup_logging(level='debug', force_reconfigure=True)
    
    logger = get_logger("scaling.level_test")
    logger.debug("Debug message (should appear)")
    logger.info("Info message (should appear)")
    logger.warning("Warning message (should appear)")
    
    print("\nSetting up with WARNING level:")
    setup_logging(level='warning', force_reconfigure=True)
    
    logger = get_logger("scaling.level_test")
    logger.debug("Debug message (should NOT appear)")
    logger.info("Info message (should NOT appear)")
    logger.warning("Warning message (should appear)")
    logger.error("Error message (should appear)")
    
    print()


def example_real_world_usage():
    """Example of real-world usage pattern."""
    print("=== Real-World Usage Pattern ===")
    
    # This is how you would typically use it in your modules
    logger = get_logger(__name__)
    
    logger.info("Starting political analysis process")
    
    try:
        # Simulate some processing
        logger.debug("Loading configuration")
        logger.info("Processing 100 documents")
        logger.info("Analysis complete")
        
        # Simulate a warning
        logger.warning("Some documents had incomplete data")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    
    logger.info("Process completed successfully")
    
    print()


if __name__ == "__main__":
    print("Logging System Usage Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_module_naming()
    example_file_logging()
    example_different_log_levels()
    example_real_world_usage()
    
    print("All examples completed!")
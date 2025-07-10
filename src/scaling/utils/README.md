# Scaling Utils

Utility modules for the Political Analysis Scaling project.

## Modules

### Config (`config.py`)

Centralized configuration management for the entire project.

#### Features
- **Environment Variable Loading**: Automatically loads from `.env` files
- **API Key Management**: Support for multiple LLM providers (OpenAI, Anthropic, Azure, Google, etc.)
- **Validation**: Check for required API keys and validate configuration
- **Provider Discovery**: Automatically detect available LLM providers

#### Usage
```python
from scaling.utils.config import config

# Check API key availability
if config.has_api_key('openai'):
    api_key = config.openai_api_key

# Get available providers
providers = config.get_available_providers()

# Validate required providers
config.validate_required_keys(['openai', 'anthropic'])
```

#### Environment Variables
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
GROK_API_KEY=...
NOVITA_API_KEY=...
LANGSMITH_API_KEY=lsv2_...
LOGGING_LEVEL=info
LOG_FILE_PATH=/path/to/logs/app.log
CALCULATE_COST=false
```

### Logging (`logging.py`)

Unified logging system with module-based naming and configurable output.

#### Features
- **Module-based Naming**: Automatic conversion to `scaling.module.submodule` format
- **Timestamped Messages**: Clear formatting with `YYYY-MM-DD HH:MM:SS` timestamps
- **Configurable Levels**: Support for all standard logging levels via environment variables
- **Dual Output**: Console and optional file logging
- **Third-party Silence**: Automatically reduces noise from external libraries

#### High-Level Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Module   │───▶│  get_logger()    │───▶│  Logger Instance│
│   (__name__)    │    │  - Name convert  │    │  - Formatted    │
│                 │    │  - Auto setup    │    │  - Timestamped  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  setup_logging() │
                       │  - Config load   │
                       │  - Handler setup │
                       │  - Level config  │
                       └──────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐        ┌──────────────┐
            │   Console    │        │   File       │
            │   Handler    │        │   Handler    │
            │   (stdout)   │        │   (optional) │
            └──────────────┘        └──────────────┘
```

#### Usage

**Basic Usage**
```python
from scaling.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Process started")
logger.warning("Something needs attention")
logger.error("An error occurred")
```

**Module Naming Examples**
```python
# File: src/scaling/utils/config.py
logger = get_logger(__name__)  # Creates: scaling.utils.config

# File: src/scaling/pipeline/analyzer.py  
logger = get_logger(__name__)  # Creates: scaling.pipeline.analyzer

# Manual naming
logger = get_logger("scaling.custom.module")
```

**File Logging**
```python
from scaling.utils.logging import get_file_logger

# Log to specific file
file_logger = get_file_logger(__name__, "/path/to/logs/analysis.log")
file_logger.info("This goes to the file")
```

**Custom Setup**
```python
from scaling.utils.logging import setup_logging

# Force reconfigure with specific settings
setup_logging(level='debug', log_file='/tmp/debug.log', force_reconfigure=True)
```

#### Log Format
```
2025-01-10 14:30:25 - scaling.utils.config - INFO - Configuration loaded successfully
2025-01-10 14:30:26 - scaling.pipeline.analyzer - WARNING - Missing some data points
2025-01-10 14:30:27 - scaling.pipeline.analyzer - ERROR - Analysis failed for document 123
```

#### Environment Configuration
The logging system automatically uses configuration from environment variables:

- `LOGGING_LEVEL`: Sets the global logging level (debug, info, warning, error, critical)
- `LOG_FILE_PATH`: Optional path for file output

#### Integration with Config
The logging system integrates seamlessly with the config module:

```python
# Logging automatically uses config settings
from scaling.utils.config import config
from scaling.utils.logging import get_logger

logger = get_logger(__name__)

# Uses config.logging_level and config.log_file_path automatically
logger.info(f"Available providers: {config.get_available_providers()}")
```

## Testing

Both modules have comprehensive pytest test suites:

```bash
# Test configuration
pytest tests/test_config.py -v

# Test logging
pytest tests/test_logging.py -v

# Run example
python tests/test_logging_example.py
```

## Design Principles

1. **Simplicity**: Easy to use with minimal setup
2. **Consistency**: Uniform naming and formatting across the project
3. **Flexibility**: Configurable via environment variables
4. **Integration**: Modules work together seamlessly
5. **Testing**: Comprehensive test coverage with pytest
6. **Documentation**: Clear usage examples and documentation
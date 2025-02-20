# 🤖 Single File Agents (SFA)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-0.1.0-green.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Build powerful, lean, and production-ready AI agents in single files. Scale your AI development without overinvestment.

## 🌟 Why Single File Agents?

In the age of rapidly evolving AI, staying agile while building robust solutions is crucial. Single File Agents (SFA) offers:

- 🚀 **Rapid Development**: Build complete AI agents in single files
- 🛡️ **Production-Ready**: Built-in validation, error handling, and monitoring
- 🔄 **Self-Verifying**: Closed-loop validation systems
- 📦 **Dependency Isolated**: UV-powered dependency management
- 🎯 **Domain-Focused**: Specialized agents for specific tasks
- ⚡ **Resource Efficient**: Memory-aware with compute controls
- 🔌 **Plug & Play**: Easy to clone and modify for new use cases

## 🎯 Perfect For

- 🔬 AI Researchers needing quick prototypes
- 🏢 Startups wanting to move fast without technical debt
- 🛠️ Engineers building domain-specific AI tools
- 🧪 Data Scientists requiring reproducible agents
- 🏗️ Teams needing production-ready AI solutions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/single-file-agents.git
cd single-file-agents

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new agent from template
uv run sfa_cookiecutter.py create query duckdb_agent \\
    --model openai \\
    --tools "list_tables,describe_table,run_query" \\
    --validation strict
```

### Basic Usage

```python
# Run your agent
uv run duckdb_agent.py -d your_database.db -p "list all tables" -c 5

# Example output:
🔍 Analyzing database...
📊 Found tables:
- users (15 columns)
- orders (8 columns)
- products (12 columns)
✅ Analysis complete!
```

## 🎨 Available Agent Templates

1. **Database Agents**
   ```bash
   # DuckDB Agent
   uv run sfa_cookiecutter.py create query duckdb_agent
   
   # SQLite Agent
   uv run sfa_cookiecutter.py create query sqlite_agent
   ```

2. **Analysis Agents**
   ```bash
   # Climate Analysis
   uv run sfa_cookiecutter.py create analysis climate_agent
   
   # Crop Analysis
   uv run sfa_cookiecutter.py create timeseries_analysis crop_agent
   ```

3. **Infrastructure Agents**
   ```bash
   # API Generator
   uv run sfa_cookiecutter.py create api_generator api_agent
   
   # Security Scanner
   uv run sfa_cookiecutter.py create security_scanner security_agent
   ```

## 🔥 Features

### 1. Built-in Validation Chain
```python
# Pre-execution validation
if not pre_execution_validation(data, "Validating input"):
    sys.exit(1)

# During-execution validation
if not during_execution_validation(data, "Monitoring execution"):
    sys.exit(1)

# Post-execution validation
if not post_execution_validation(data, "Verifying results"):
    sys.exit(1)
```

### 2. Memory Management
```python
with MemoryManager(threshold_mb=1000) as mm:
    # Your memory-intensive operations
    result = process_large_dataset(data)
    
    if not mm.check_memory():
        mm.cleanup()
```

### 3. Compute Control
```python
# Set maximum compute iterations
uv run agent.py -c 5 -p "your complex query"

# Background processing
with ThreadPoolExecutor() as executor:
    monitor_future = executor.submit(monitor_resources)
    result_future = executor.submit(process_data)
```

### 4. Rich Logging
```python
# Automatic progress and error logging
console.log("[blue]Processing[/blue] - Starting analysis...")
console.log("[red]Error[/red] - Failed to connect to database")
console.log("[green]Success[/green] - Analysis complete!")
```

## 📚 Documentation

### Agent Structure
```plaintext
single-file-agent/
├── Dependencies (inline UV metadata)
├── Documentation
│   ├── Purpose
│   ├── Examples
│   └── Usage
├── Tools
│   ├── Validation tools
│   ├── Core tools
│   └── Verification tools
├── Validation Chain
│   ├── Pre-execution
│   ├── During execution
│   └── Post-execution
└── Main Loop
    ├── Model configuration
    ├── Compute control
    └── Memory management
```

### Creating Custom Agents

1. **Define Your Interface**
```python
class CustomDataInterface(BaseModel):
    """Implement this interface for your data source"""
    required_fields: List[str]
    
    def load_data(self) -> pd.DataFrame:
        """Override this method to load your data"""
        raise NotImplementedError
```

2. **Add Validation Rules**
```python
def validate_data(self) -> bool:
    """Validates data meets requirements"""
    try:
        if not self.df is not None:
            return False
        return self.validate_schema() and self.validate_values()
    except Exception as e:
        console.log(f"[red]Data validation failed: {str(e)}[/red]")
        return False
```

3. **Implement Tools**
```python
def process_data(reasoning: str, data: Dict[str, Any]) -> str:
    """Process data with validation and monitoring"""
    try:
        with MemoryManager() as mm:
            # Your processing logic here
            result = transform_data(data)
            return result
    except Exception as e:
        console.log(f"[red]Processing failed: {str(e)}[/red]")
        return str(e)
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Astral](https://github.com/astral-sh) for the amazing UV tool
- [Rich](https://github.com/Textualize/rich) for beautiful terminal outputs
- [Pydantic](https://github.com/pydantic/pydantic) for robust data validation

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/single-file-agents&type=Date)](https://star-history.com/#yourusername/single-file-agents&Date)

---

<p align="center">
  Made with ❤️ by the SFA Team
</p>

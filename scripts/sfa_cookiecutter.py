# /// script
# dependencies = [
#   "cookiecutter>=2.5.0",
#   "rich>=13.7.0",
# ]
# ///

import os
import argparse
from rich.console import Console

console = Console()

SFA_TEMPLATES = {
    "analysis": {
        "base_tools": [
            # Pre-validation tools
            "validate_data_integrity",
            "validate_analysis_approach",
            
            # Core analysis tools
            "analyze_subset",
            "run_test_analysis",
            "run_final_analysis",
            
            # Verification tools
            "verify_results",
            "plot_verification",
            "generate_report"
        ],
        "dependencies": [
            "pandas",
            "numpy", 
            "openai/anthropic/gemini",
            "rich",
            "pydantic"
        ],
        "prompt_sections": [
            "<purpose>",
            "<instructions>",
            "<tools>",
            "<examples>",  # Optional but recommended
            "<user-request>"
        ],
        "validation_chain": {
            "pre_execution": [
                "data_quality_checks",
                "approach_validation",
                "expected_outcome_definition"
            ],
            "during_execution": [
                "test_on_subset",
                "verify_intermediate_results",
                "physical_bounds_check"
            ],
            "post_execution": [
                "results_verification",
                "confidence_assessment",
                "domain_knowledge_validation"
            ]
        }
    },
    
    "agent": {
        "base_tools": [
            # Planning tools
            "validate_request",
            "plan_actions",
            "estimate_resources",
            
            # Execution tools
            "execute_action",
            "monitor_progress",
            "handle_error",
            
            # Verification tools
            "verify_result",
            "validate_output",
            "generate_report"
        ],
        "dependencies": [
            "openai/anthropic/gemini",
            "rich",
            "pydantic"
        ],
        "prompt_sections": [
            "<purpose>",
            "<instructions>",
            "<tools>",
            "<examples>",  # Optional but recommended
            "<user-request>"
        ],
        "validation_chain": {
            "pre_execution": [
                "request_validation",
                "resource_check",
                "safety_check"
            ],
            "during_execution": [
                "progress_monitoring",
                "error_detection",
                "intermediate_validation"
            ],
            "post_execution": [
                "output_verification",
                "success_criteria_check",
                "impact_assessment"
            ]
        }
    },

    "query": {
        "base_tools": [
            # Query validation
            "validate_query",
            "test_query",
            "estimate_cost",
            
            # Execution tools
            "run_test_query",
            "run_final_query",
            
            # Verification tools
            "verify_results",
            "validate_output",
            "generate_report"
        ],
        "dependencies": [
            "openai/anthropic/gemini",
            "rich",
            "pydantic",
            "database-specific-lib"
        ],
        "prompt_sections": [
            "<purpose>",
            "<instructions>",
            "<tools>",
            "<examples>",
            "<user-request>"
        ],
        "validation_chain": {
            "pre_execution": [
                "query_validation",
                "schema_verification",
                "cost_estimation"
            ],
            "during_execution": [
                "test_on_subset",
                "performance_monitoring",
                "result_sampling"
            ],
            "post_execution": [
                "result_verification",
                "data_quality_check",
                "consistency_validation"
            ]
        }
    },

    "timeseries_analysis": {
        "description": "Analyze any time series data with built-in validation",
        "data_interface": """
class TimeSeriesDataInterface(BaseModel):
    \"\"\"Implement this interface for your data source\"\"\"
    timestamp_column: str
    value_column: str
    groupby_columns: Optional[List[str]]
    
    def load_data(self) -> pd.DataFrame:
        \"\"\"Override this method to load your data\"\"\"
        raise NotImplementedError
        
    def validate_schema(self) -> bool:
        \"\"\"Validates required columns exist\"\"\"
        return all(col in self.df.columns for col in [
            self.timestamp_column,
            self.value_column
        ])
        
    def validate_data(self) -> bool:
        \"\"\"Validates data meets requirements\"\"\"
        try:
            if not self.df is not None:
                return False
            return self.validate_schema() and self.validate_values()
        except Exception as e:
            console.log(f"[red]Data validation failed: {str(e)}[/red]")
            return False
            
    def validate_values(self) -> bool:
        \"\"\"Validates data values\"\"\"
        try:
            # Check for empty dataframe
            if self.df.empty:
                return False
                
            # Check for null values
            if self.df[self.timestamp_column].isnull().any():
                return False
                
            # Check for invalid timestamps
            try:
                pd.to_datetime(self.df[self.timestamp_column])
            except:
                return False
                
            # Check for numeric values in value column
            if not pd.api.types.is_numeric_dtype(self.df[self.value_column]):
                return False
                
            return True
        except Exception as e:
            console.log(f"[red]Value validation failed: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class ClimateDataInterface(TimeSeriesDataInterface):
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv('climate_data.csv')
        
class CropDataInterface(TimeSeriesDataInterface):
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv('crop_data.csv')
""",
        "dependencies": [
            "pandas",
            "numpy",
            "openai/anthropic/gemini",
            "rich",
            "pydantic"
        ],
        "validation_chain": {
            "pre_execution": [
                "data_quality_checks",
                "schema_validation",
                "time_range_validation"
            ],
            "during_execution": [
                "test_on_subset",
                "verify_intermediate_results",
                "trend_validation"
            ],
            "post_execution": [
                "results_verification",
                "statistical_validation",
                "domain_validation"
            ]
        }
    },
    
    "ml_prediction": {
        "description": "ML prediction agent with built-in validation chain",
        "data_interface": """
class PredictionDataInterface(BaseModel):
    \"\"\"Implement this interface for your prediction data\"\"\"
    feature_columns: List[str]
    target_column: str
    date_column: Optional[str]
    
    def load_training_data(self) -> pd.DataFrame:
        \"\"\"Override to load training data\"\"\"
        raise NotImplementedError
        
    def load_prediction_data(self) -> pd.DataFrame:
        \"\"\"Override to load data for predictions\"\"\"
        raise NotImplementedError
        
    def validate_features(self) -> bool:
        \"\"\"Validates features exist and are numeric\"\"\"
        return all(col in self.df.columns for col in self.feature_columns)
""",
        "example_implementation": """
class CropYieldPredictor(PredictionDataInterface):
    def load_training_data(self) -> pd.DataFrame:
        return pd.read_csv('historical_crop_data.csv')
        
    def load_prediction_data(self) -> pd.DataFrame:
        return pd.read_csv('current_crop_data.csv')
"""
    },

    "data_quality": {
        "description": "Data quality analysis with built-in validation",
        "data_interface": """
class DataQualityInterface(BaseModel):
    \"\"\"Implement this interface for data quality checks\"\"\"
    required_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    
    def load_data(self) -> pd.DataFrame:
        \"\"\"Override to load your data\"\"\"
        raise NotImplementedError
        
    def validate_datatypes(self) -> Dict[str, bool]:
        \"\"\"Validates column datatypes\"\"\"
        return {col: pd.api.types.is_numeric_dtype(self.df[col]) 
                for col in self.numeric_columns}
""",
        "example_implementation": """
class ClimateDataQuality(DataQualityInterface):
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv('climate_data.csv')
"""
    },

    "earth_engine": {
        "description": "Google Earth Engine analysis with local development support",
        "base_tools": [
            # Authentication and setup
            "validate_credentials",
            "initialize_gee",
            "setup_local_cache",
            
            # Data handling
            "load_gee_dataset",
            "export_to_local",
            "cache_results",
            
            # Analysis tools
            "run_gee_computation",
            "validate_results",
            "visualize_results"
        ],
        "dependencies": [
            "earthengine-api>=0.1.375",
            "geemap>=0.28.0",
            "pandas",
            "numpy",
            "rich",
            "pydantic",
            "folium>=0.14.0",
            "geopandas>=0.14.0"
        ],
        "data_interface": """
class GEEDataInterface(BaseModel):
    \"\"\"Interface for Google Earth Engine data handling\"\"\"
    dataset_id: str
    region_of_interest: Optional[Dict]
    start_date: Optional[str]
    end_date: Optional[str]
    scale: Optional[int] = 30  # Default Landsat resolution
    max_pixels: Optional[int] = 1e8
    
    def initialize_gee(self) -> bool:
        \"\"\"Initialize Earth Engine and authenticate\"\"\"
        try:
            import ee
            # Try to initialize without authentication
            try:
                ee.Initialize()
                console.log("[green]Successfully initialized Earth Engine[/green]")
                return True
            except ee.EEException:
                # If fails, authenticate
                ee.Authenticate()
                ee.Initialize()
                console.log("[green]Successfully authenticated and initialized Earth Engine[/green]")
                return True
        except Exception as e:
            console.log(f"[red]Failed to initialize Earth Engine: {str(e)}[/red]")
            return False
    
    def load_dataset(self) -> 'ee.ImageCollection':
        \"\"\"Load GEE dataset\"\"\"
        try:
            import ee
            collection = ee.ImageCollection(self.dataset_id)
            
            # Apply filters if provided
            if self.start_date and self.end_date:
                collection = collection.filterDate(self.start_date, self.end_date)
            
            if self.region_of_interest:
                geometry = ee.Geometry(self.region_of_interest)
                collection = collection.filterBounds(geometry)
            
            return collection
        except Exception as e:
            console.log(f"[red]Failed to load dataset: {str(e)}[/red]")
            raise
    
    def validate_data(self) -> bool:
        \"\"\"Validate GEE dataset access and parameters\"\"\"
        try:
            import ee
            # Check if dataset exists
            collection = ee.ImageCollection(self.dataset_id)
            info = collection.getInfo()
            
            # Validate dates if provided
            if self.start_date and self.end_date:
                start = ee.Date(self.start_date)
                end = ee.Date(self.end_date)
                if start.gt(end):
                    console.log("[red]Error: Start date is after end date[/red]")
                    return False
            
            # Validate region if provided
            if self.region_of_interest:
                geometry = ee.Geometry(self.region_of_interest)
                area = geometry.area().getInfo()
                if area <= 0:
                    console.log("[red]Error: Invalid region geometry[/red]")
                    return False
            
            return True
        except Exception as e:
            console.log(f"[red]Data validation failed: {str(e)}[/red]")
            return False
            
    def setup_local_cache(self) -> bool:
        \"\"\"Setup local caching for GEE results\"\"\"
        try:
            import os
            cache_dir = os.path.join(os.getcwd(), '.gee_cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            return True
        except Exception as e:
            console.log(f"[red]Failed to setup local cache: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class LandsatAnalysis(GEEDataInterface):
    def __init__(self):
        super().__init__(
            dataset_id='LANDSAT/LC08/C02/T1_L2',
            region_of_interest={
                'type': 'Polygon',
                'coordinates': [[[-122.5, 37.7], [-122.4, 37.7], 
                               [-122.4, 37.8], [-122.5, 37.8], [-122.5, 37.7]]]
            },
            start_date='2022-01-01',
            end_date='2022-12-31'
        )
        
class SentinelAnalysis(GEEDataInterface):
    def __init__(self):
        super().__init__(
            dataset_id='COPERNICUS/S2_SR',
            scale=10  # Sentinel-2 resolution
        )
""",
        "validation_chain": {
            "pre_execution": [
                "credentials_validation",
                "gee_initialization",
                "dataset_access"
            ],
            "during_execution": [
                "computation_monitoring",
                "memory_usage",
                "export_progress"
            ],
            "post_execution": [
                "result_validation",
                "local_backup",
                "visualization_check"
            ]
        }
    },

    "data_pipeline": {
        "description": "ETL/ELT pipeline automation with validation",
        "data_interface": """
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect
import pandas as pd

class TableSchema(BaseModel):
    \"\"\"Schema definition for a database table\"\"\"
    name: str
    columns: List[str]
    primary_key: Optional[List[str]] = None
    indexes: Optional[List[str]] = None

class SourceConfig(BaseModel):
    \"\"\"Configuration for data source\"\"\"
    connection: str
    schema: List[TableSchema]
    format: Optional[str] = 'sql'  # sql, csv, parquet, etc.

class TargetConfig(BaseModel):
    \"\"\"Configuration for data target\"\"\"
    connection: str
    schema: List[TableSchema]
    format: Optional[str] = 'sql'

class TransformationRule(BaseModel):
    \"\"\"Definition of a data transformation rule\"\"\"
    name: str
    type: str  # sql, python, etc.
    config: Dict[str, Any]

class ValidationRule(BaseModel):
    \"\"\"Definition of a data validation rule\"\"\"
    expectation_type: str
    kwargs: Dict[str, Any]

class DataPipelineInterface(BaseModel):
    \"\"\"Interface for ETL/ELT pipeline configuration\"\"\"
    source_config: SourceConfig
    target_config: TargetConfig
    transformation_rules: List[TransformationRule]
    validation_rules: Optional[List[ValidationRule]] = Field(default_factory=list)
    
    def load_data(self) -> Dict[str, Any]:
        \"\"\"Load pipeline configuration data\"\"\"
        try:
            config = {
                'source': self.source_config.dict(),
                'target': self.target_config.dict(),
                'transformations': [rule.dict() for rule in self.transformation_rules],
                'validations': [rule.dict() for rule in self.validation_rules]
            }
            return config
        except Exception as e:
            console.log(f"[red]Failed to load pipeline configuration: {str(e)}[/red]")
            raise
            
    def validate_source(self) -> bool:
        \"\"\"Validate source data configuration\"\"\"
        try:
            # Validate source connection
            engine = create_engine(self.source_config.connection)
            with engine.connect() as conn:
                inspector = inspect(engine)
                
                # Validate tables exist
                existing_tables = inspector.get_table_names()
                for table in self.source_config.schema:
                    if table.name not in existing_tables:
                        console.log(f"[red]Table {table.name} not found in source[/red]")
                        return False
                        
                    # Validate columns exist
                    columns = [col['name'] for col in inspector.get_columns(table.name)]
                    for col in table.columns:
                        if col not in columns:
                            console.log(f"[red]Column {col} not found in table {table.name}[/red]")
                            return False
                            
            return True
        except Exception as e:
            console.log(f"[red]Source validation failed: {str(e)}[/red]")
            return False
            
    def validate_transformations(self) -> bool:
        \"\"\"Validate transformation rules\"\"\"
        try:
            for rule in self.transformation_rules:
                # Validate SQL transformations
                if rule.type == 'sql':
                    if 'query' not in rule.config:
                        console.log(f"[red]Missing query in SQL transformation {rule.name}[/red]")
                        return False
                        
                    # Basic SQL syntax validation
                    query = rule.config['query'].lower()
                    if not any(keyword in query for keyword in ['select', 'insert', 'update', 'delete']):
                        console.log(f"[red]Invalid SQL query in transformation {rule.name}[/red]")
                        return False
                        
                # Validate Python transformations
                elif rule.type == 'python':
                    if 'function' not in rule.config:
                        console.log(f"[red]Missing function in Python transformation {rule.name}[/red]")
                        return False
                        
                    if not callable(rule.config['function']):
                        console.log(f"[red]Invalid Python function in transformation {rule.name}[/red]")
                        return False
                        
            return True
        except Exception as e:
            console.log(f"[red]Transformation validation failed: {str(e)}[/red]")
            return False
            
    def validate_schema(self) -> bool:
        \"\"\"Validate data schema compatibility\"\"\"
        try:
            # Validate source schema
            if not self.validate_source():
                return False
                
            # Validate target schema compatibility
            target_tables = {table.name: set(table.columns) for table in self.target_config.schema}
            source_tables = {table.name: set(table.columns) for table in self.source_config.schema}
            
            for table_name, target_cols in target_tables.items():
                if table_name not in source_tables:
                    console.log(f"[red]Missing source table: {table_name}[/red]")
                    return False
                    
                source_cols = source_tables[table_name]
                missing_cols = target_cols - source_cols
                if missing_cols:
                    console.log(f"[red]Missing columns in {table_name}: {missing_cols}[/red]")
                    return False
                    
            return True
        except Exception as e:
            console.log(f"[red]Schema validation failed: {str(e)}[/red]")
            return False
            
    def validate_data(self) -> bool:
        \"\"\"Comprehensive data validation\"\"\"
        try:
            # Schema validation
            if not self.validate_schema():
                return False
                
            # Transformation validation
            if not self.validate_transformations():
                return False
                
            # Setup monitoring
            if not self.setup_monitoring():
                return False
                
            return True
        except Exception as e:
            console.log(f"[red]Data validation failed: {str(e)}[/red]")
            return False
            
    def setup_monitoring(self) -> bool:
        \"\"\"Setup pipeline monitoring\"\"\"
        try:
            from great_expectations.data_context import DataContext
            
            context = DataContext.create(
                project_root_dir="./.great_expectations"
            )
            
            # Setup basic expectations
            suite = context.create_expectation_suite(
                expectation_suite_name="pipeline_validation"
            )
            
            # Add validation rules if provided
            for rule in self.validation_rules:
                suite.add_expectation(
                    expectation_type=rule.expectation_type,
                    kwargs=rule.kwargs
                )
                    
            context.save_expectation_suite(suite)
            return True
        except Exception as e:
            console.log(f"[red]Monitoring setup failed: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class PostgresToRedshiftPipeline(DataPipelineInterface):
    def __init__(self):
        super().__init__(
            source_config={
                'connection': 'postgresql://user:pass@localhost:5432/db',
                'schema': [
                    {
                        'name': 'users',
                        'columns': ['id', 'name', 'email']
                    }
                ]
            },
            target_config={
                'connection': 'redshift://user:pass@cluster.region.redshift.amazonaws.com:5439/db',
                'schema': [
                    {
                        'name': 'users',
                        'columns': ['id', 'name', 'email']
                    }
                ]
            },
            transformation_rules=[
                {
                    'name': 'clean_email',
                    'type': 'sql',
                    'config': {
                        'query': 'SELECT id, name, LOWER(email) as email FROM users'
                    }
                }
            ],
            validation_rules={
                'expectations': [
                    {
                        'expectation_type': 'expect_column_values_to_not_be_null',
                        'kwargs': {'column': 'email'}
                    }
                ]
            }
        )
""",
        "validation_chain": {
            "pre_execution": [
                "source_validation",
                "schema_validation",
                "transformation_validation"
            ],
            "during_execution": [
                "data_quality_monitoring",
                "performance_tracking",
                "error_detection"
            ],
            "post_execution": [
                "target_validation",
                "data_reconciliation",
                "pipeline_metrics"
            ]
        }
    },

    "llm_app_builder": {
        "description": "LLM application scaffolding and validation",
        "base_tools": [
            "validate_prompts",
            "test_completions",
            "monitor_costs",
            "validate_outputs",
            "handle_errors",
            "track_performance"
        ],
        "dependencies": [
            "langchain>=0.1.0",
            "openai>=1.63.0",
            "anthropic>=0.7.0",
            "instructor>=0.4.0",
            "guidance>=0.1.0",
            "rich>=13.7.0",
            "pydantic>=2.0.0",
            "tiktoken>=0.5.0"
        ],
        "data_interface": """
class LLMAppInterface(BaseModel):
    \"\"\"Interface for LLM application configuration\"\"\"
    model_config: Dict[str, Any]
    prompt_templates: Dict[str, str]
    validation_rules: Dict[str, Any]
    cost_limits: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    def validate_prompts(self) -> bool:
        \"\"\"Validate prompt templates\"\"\"
        try:
            import tiktoken
            
            # Get tokenizer based on model
            model_name = self.model_config.get('model', 'gpt-3.5-turbo')
            tokenizer = tiktoken.encoding_for_model(model_name)
            
            # Validate each prompt template
            for name, template in self.prompt_templates.items():
                # Check for required placeholders
                if not all(p in template for p in self.validation_rules.get('required_placeholders', [])):
                    console.log(f"[red]Missing required placeholders in template: {name}[/red]")
                    return False
                    
                # Check token length
                tokens = tokenizer.encode(template)
                max_tokens = self.validation_rules.get('max_prompt_tokens', 4000)
                if len(tokens) > max_tokens:
                    console.log(f"[red]Template {name} exceeds max token length[/red]")
                    return False
                    
            return True
        except Exception as e:
            console.log(f"[red]Prompt validation failed: {str(e)}[/red]")
            return False
            
    def validate_outputs(self, output: str, context: Dict[str, Any]) -> bool:
        \"\"\"Validate LLM outputs\"\"\"
        try:
            # Check output length
            if len(output) < self.validation_rules.get('min_output_length', 1):
                return False
                
            # Check for required patterns
            for pattern in self.validation_rules.get('required_patterns', []):
                if not re.search(pattern, output):
                    return False
                    
            # Check for forbidden content
            for pattern in self.validation_rules.get('forbidden_patterns', []):
                if re.search(pattern, output):
                    return False
                    
            return True
        except Exception as e:
            console.log(f"[red]Output validation failed: {str(e)}[/red]")
            return False
            
    def track_costs(self, usage: Dict[str, int]) -> bool:
        \"\"\"Track and validate API usage costs\"\"\"
        try:
            # Calculate costs based on usage
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            # Cost per 1k tokens (example rates)
            rates = {
                'gpt-4': {'prompt': 0.03, 'completion': 0.06},
                'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002}
            }
            
            model = self.model_config.get('model', 'gpt-3.5-turbo')
            rate = rates.get(model, rates['gpt-3.5-turbo'])
            
            cost = (prompt_tokens * rate['prompt'] + completion_tokens * rate['completion']) / 1000
            
            # Check against limits
            if cost > self.cost_limits.get('per_request', float('inf')):
                console.log(f"[red]Cost limit exceeded: ${cost:.4f}[/red]")
                return False
                
            return True
        except Exception as e:
            console.log(f"[red]Cost tracking failed: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class CustomerSupportBot(LLMAppInterface):
    def __init__(self):
        super().__init__(
            model_config={
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 500
            },
            prompt_templates={
                'support_response': \"\"\"
                Context: {context}
                Customer Query: {query}
                Previous Interaction: {history}
                
                Respond to the customer query professionally and empathetically.
                Include relevant information from the context.
                If technical support is needed, provide step-by-step instructions.
                \"\"\"
            },
            validation_rules={
                'required_placeholders': ['context', 'query', 'history'],
                'max_prompt_tokens': 3000,
                'min_output_length': 50,
                'required_patterns': [
                    r'(?i)thank.*you',  # Requires some form of "thank you"
                    r'(?i)help.*you.*further'  # Requires offer for further help
                ],
                'forbidden_patterns': [
                    r'(?i)password',  # Never include passwords
                    r'(?i)account.*number'  # Never include account numbers
                ]
            },
            cost_limits={
                'per_request': 0.10,  # Maximum $0.10 per request
                'daily': 10.0  # Maximum $10 per day
            }
        )
""",
        "validation_chain": {
            "pre_execution": [
                "prompt_validation",
                "model_validation",
                "cost_estimation"
            ],
            "during_execution": [
                "output_monitoring",
                "cost_tracking",
                "performance_monitoring"
            ],
            "post_execution": [
                "response_validation",
                "safety_check",
                "quality_metrics"
            ]
        }
    },

    "security_scanner": {
        "description": "Automated security testing and validation",
        "base_tools": [
            "scan_dependencies",
            "check_vulnerabilities",
            "test_endpoints",
            "validate_auth",
            "analyze_code",
            "generate_report"
        ],
        "dependencies": [
            "safety>=2.3.0",
            "bandit>=1.7.0",
            "python-owasp-zap-v2.4>=0.0.20",
            "pytest-security>=0.1.0",
            "cryptography>=41.0.0",
            "rich>=13.7.0",
            "pydantic>=2.0.0"
        ],
        "data_interface": """
class SecurityScanInterface(BaseModel):
    \"\"\"Interface for security scanning configuration\"\"\"
    target_config: Dict[str, Any]
    scan_rules: Dict[str, Any]
    compliance_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def validate_target(self) -> bool:
        \"\"\"Validate scan target configuration\"\"\"
        try:
            required_fields = ['type', 'location']
            if not all(f in self.target_config for f in required_fields):
                return False
                
            # Validate based on target type
            target_type = self.target_config['type']
            if target_type == 'web_app':
                return self._validate_web_target()
            elif target_type == 'codebase':
                return self._validate_codebase()
            elif target_type == 'dependencies':
                return self._validate_dependencies()
                
            return False
        except Exception as e:
            console.log(f"[red]Target validation failed: {str(e)}[/red]")
            return False
            
    def _validate_web_target(self) -> bool:
        \"\"\"Validate web application target\"\"\"
        try:
            import requests
            url = self.target_config['location']
            response = requests.head(url, timeout=10)
            return response.status_code < 400
        except Exception:
            return False
            
    def _validate_codebase(self) -> bool:
        \"\"\"Validate codebase target\"\"\"
        try:
            path = self.target_config['location']
            return os.path.exists(path)
        except Exception:
            return False
            
    def _validate_dependencies(self) -> bool:
        \"\"\"Validate dependencies target\"\"\"
        try:
            path = self.target_config['location']
            return os.path.exists(path) and any(
                f.endswith(('.txt', '.json', '.toml', '.lock'))
                for f in os.listdir(path)
            )
        except Exception:
            return False
            
    def setup_scan(self) -> bool:
        \"\"\"Setup security scanning environment\"\"\"
        try:
            # Setup scan directory
            scan_dir = os.path.join(os.getcwd(), '.security_scan')
            os.makedirs(scan_dir, exist_ok=True)
            
            # Initialize scan configuration
            config = {
                'target': self.target_config,
                'rules': self.scan_rules,
                'compliance': self.compliance_requirements,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(scan_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
                
            return True
        except Exception as e:
            console.log(f"[red]Scan setup failed: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class WebAppSecurityScan(SecurityScanInterface):
    def __init__(self):
        super().__init__(
            target_config={
                'type': 'web_app',
                'location': 'https://example.com',
                'auth': {
                    'type': 'bearer',
                    'token': os.getenv('API_TOKEN')
                }
            },
            scan_rules={
                'vulnerability_scan': True,
                'auth_test': True,
                'injection_test': True,
                'xss_test': True,
                'csrf_test': True
            },
            compliance_requirements={
                'standards': ['OWASP Top 10', 'PCI DSS'],
                'risk_level': 'high',
                'required_checks': [
                    'sql_injection',
                    'xss',
                    'csrf',
                    'auth_bypass'
                ]
            }
        )
""",
        "validation_chain": {
            "pre_execution": [
                "target_validation",
                "credentials_check",
                "compliance_check"
            ],
            "during_execution": [
                "scan_monitoring",
                "vulnerability_tracking",
                "performance_impact"
            ],
            "post_execution": [
                "findings_validation",
                "false_positive_check",
                "report_generation"
            ]
        }
    },

    "api_generator": {
        "description": "FastAPI/OpenAPI generator with validation",
        "data_interface": """
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine
import re

class ModelField(BaseModel):
    \"\"\"Field definition for API model\"\"\"
    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    unique: bool = False
    default: Optional[Any] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError('Invalid field name format')
        return v
        
    @validator('type')
    def validate_type(cls, v):
        valid_types = {'str', 'int', 'float', 'bool', 'datetime', 'date', 'dict', 'list'}
        if v.lower() not in valid_types:
            raise ValueError(f'Invalid type. Must be one of: {valid_types}')
        return v.lower()

class Relationship(BaseModel):
    \"\"\"Model relationship definition\"\"\"
    type: str  # one_to_one, one_to_many, many_to_many
    model: str
    back_populates: Optional[str] = None
    cascade: Optional[str] = None
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = {'one_to_one', 'one_to_many', 'many_to_many'}
        if v not in valid_types:
            raise ValueError(f'Invalid relationship type. Must be one of: {valid_types}')
        return v

class APIModel(BaseModel):
    \"\"\"Model definition for API\"\"\"
    name: str
    fields: List[ModelField]
    description: Optional[str] = None
    relationships: Optional[List[Relationship]] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', v):
            raise ValueError('Model name must start with uppercase letter')
        return v
        
    @validator('fields')
    def validate_fields(cls, v):
        field_names = [f.name for f in v]
        if len(field_names) != len(set(field_names)):
            raise ValueError('Duplicate field names found')
        return v

class Parameter(BaseModel):
    \"\"\"API endpoint parameter\"\"\"
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = {'path', 'query', 'header', 'cookie'}
        if v not in valid_types:
            raise ValueError(f'Invalid parameter type. Must be one of: {valid_types}')
        return v

class RequestBody(BaseModel):
    \"\"\"API endpoint request body\"\"\"
    model: str
    required: bool = True
    description: Optional[str] = None

class Endpoint(BaseModel):
    \"\"\"Endpoint definition for API\"\"\"
    path: str
    method: str
    response_model: str
    description: Optional[str] = None
    auth_required: bool = False
    parameters: Optional[List[Parameter]] = None
    request_body: Optional[RequestBody] = None
    
    @validator('path')
    def validate_path(cls, v):
        if not v.startswith('/'):
            raise ValueError('Path must start with /')
        return v
        
    @validator('method')
    def validate_method(cls, v):
        valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
        if v not in valid_methods:
            raise ValueError(f'Invalid HTTP method. Must be one of: {valid_methods}')
        return v

class AuthConfig(BaseModel):
    \"\"\"Authentication configuration\"\"\"
    auth_type: str  # jwt, oauth2, basic
    secret_key: str
    token_expire_minutes: Optional[int] = 30
    refresh_token_expire_days: Optional[int] = 7
    
    @validator('auth_type')
    def validate_auth_type(cls, v):
        valid_types = {'jwt', 'oauth2', 'basic'}
        if v not in valid_types:
            raise ValueError(f'Invalid auth type. Must be one of: {valid_types}')
        return v
        
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v

class DatabaseConfig(BaseModel):
    \"\"\"Database configuration\"\"\"
    url: str
    type: str  # postgresql, mysql, sqlite, oracle
    pool_size: Optional[int] = 5
    max_overflow: Optional[int] = 10
    
    @validator('type')
    def validate_db_type(cls, v):
        valid_types = {'postgresql', 'mysql', 'sqlite', 'oracle'}
        if v not in valid_types:
            raise ValueError(f'Invalid database type. Must be one of: {valid_types}')
        return v
        
    @validator('pool_size', 'max_overflow')
    def validate_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Must be positive')
        return v

class APIGeneratorInterface(BaseModel):
    \"\"\"Interface for API generation configuration\"\"\"
    api_name: str
    models: List[APIModel]
    endpoints: List[Endpoint]
    auth_config: Optional[AuthConfig] = None
    database_config: Optional[DatabaseConfig] = None
    
    def load_data(self) -> Dict[str, Any]:
        \"\"\"Load API configuration data\"\"\"
        try:
            config = {
                'api_name': self.api_name,
                'models': [model.dict() for model in self.models],
                'endpoints': [endpoint.dict() for endpoint in self.endpoints]
            }
            
            if self.auth_config:
                config['auth_config'] = self.auth_config.dict()
            if self.database_config:
                config['database_config'] = self.database_config.dict()
                
            return config
        except Exception as e:
            console.log(f"[red]Failed to load API configuration: {str(e)}[/red]")
            raise
            
    def validate_schema(self) -> bool:
        \"\"\"Validate API schema\"\"\"
        try:
            # Validate models
            model_names = set()
            for model in self.models:
                # Check for duplicate model names
                if model.name in model_names:
                    console.log(f"[red]Duplicate model name: {model.name}[/red]")
                    return False
                model_names.add(model.name)
                
                # Validate relationships if present
                if model.relationships:
                    for rel in model.relationships:
                        if rel.model not in model_names:
                            console.log(f"[red]Invalid relationship target in model {model.name}: {rel.model}[/red]")
                            return False
                            
            # Validate endpoints
            endpoint_paths = set()
            for endpoint in self.endpoints:
                # Check for duplicate paths
                if endpoint.path in endpoint_paths:
                    console.log(f"[red]Duplicate endpoint path: {endpoint.path}[/red]")
                    return False
                endpoint_paths.add(endpoint.path)
                
                # Validate response model exists
                if endpoint.response_model not in model_names:
                    console.log(f"[red]Invalid response model: {endpoint.response_model}[/red]")
                    return False
                    
                # Validate request body model if present
                if endpoint.request_body and endpoint.request_body.model not in model_names:
                    console.log(f"[red]Invalid request body model: {endpoint.request_body.model}[/red]")
                    return False
                    
            return True
        except Exception as e:
            console.log(f"[red]Schema validation failed: {str(e)}[/red]")
            return False
            
    def validate_data(self) -> bool:
        \"\"\"Validate API configuration data\"\"\"
        try:
            # Schema validation
            if not self.validate_schema():
                return False
                
            # Validate database connection if configured
            if self.database_config:
                try:
                    engine = create_engine(self.database_config.url)
                    with engine.connect():
                        pass
                except Exception as e:
                    console.log(f"[red]Database connection failed: {str(e)}[/red]")
                    return False
                    
            # Validate endpoint paths
            for endpoint in self.endpoints:
                # Check path parameters match model fields
                path_params = [p.strip('{}') for p in endpoint.path.split('/') if p.startswith('{')]
                model = next(m for m in self.models if m.name == endpoint.response_model)
                model_fields = {f.name for f in model.fields}
                
                for param in path_params:
                    if param not in model_fields:
                        console.log(f"[red]Path parameter {param} not found in model {model.name}[/red]")
                        return False
                        
                # Validate parameter types if present
                if endpoint.parameters:
                    for param in endpoint.parameters:
                        if param.type == 'path' and param.name not in path_params:
                            console.log(f"[red]Path parameter {param.name} not found in path[/red]")
                            return False
                            
            return True
        except Exception as e:
            console.log(f"[red]Data validation failed: {str(e)}[/red]")
            return False
""",
        "example_implementation": """
class UserAPIGenerator(APIGeneratorInterface):
    def __init__(self):
        super().__init__(
            api_name='User API',
            models=[
                {
                    'name': 'User',
                    'fields': [
                        {'name': 'id', 'type': 'int', 'primary_key': True},
                        {'name': 'email', 'type': 'str', 'unique': True},
                        {'name': 'password', 'type': 'str', 'exclude': True},
                        {'name': 'is_active', 'type': 'bool', 'default': True}
                    ]
                },
                {
                    'name': 'UserCreate',
                    'fields': [
                        {'name': 'email', 'type': 'str'},
                        {'name': 'password', 'type': 'str'}
                    ]
                }
            ],
            endpoints=[
                {
                    'path': '/users',
                    'method': 'POST',
                    'response_model': 'User',
                    'auth_required': False,
                    'description': 'Create new user'
                },
                {
                    'path': '/users/me',
                    'method': 'GET',
                    'response_model': 'User',
                    'auth_required': True,
                    'description': 'Get current user'
                }
            ],
            auth_config={
                'type': 'jwt',
                'algorithm': 'HS256',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7
            },
            database_config={
                'url': 'postgresql://user:pass@localhost:5432/db',
                'type': 'postgresql'
            }
        )
""",
        "validation_chain": {
            "pre_execution": [
                "model_validation",
                "endpoint_validation",
                "auth_validation"
            ],
            "during_execution": [
                "code_generation",
                "test_generation",
                "doc_generation"
            ],
            "post_execution": [
                "api_testing",
                "schema_validation",
                "security_check"
            ]
        }
    }
}

TOOL_TEMPLATES = {
    "validation_tool": """
def validate_{{name}}(reasoning: str, {{params}}) -> str:
    \"\"\"{{description}}\"\"\"
    try:
        # Validation logic
        console.log(f"[blue]Validation[/blue] - {{name}} - Reasoning: {reasoning}")
        return result
    except Exception as e:
        console.log(f"[red]Error in validation: {str(e)}[/red]")
        return str(e)
    """,
    
    "execution_tool": """
def execute_{{name}}(reasoning: str, {{params}}) -> str:
    \"\"\"{{description}}\"\"\"
    try:
        # Execution logic
        console.log(f"[blue]Execution[/blue] - {{name}} - Reasoning: {reasoning}")
        return result
    except Exception as e:
        console.log(f"[red]Error in execution: {str(e)}[/red]")
        return str(e)
    """,
    
    "verification_tool": """
def verify_{{name}}(reasoning: str, {{params}}) -> str:
    \"\"\"{{description}}\"\"\"
    try:
        # Verification logic
        console.log(f"[blue]Verification[/blue] - {{name}} - Reasoning: {reasoning}")
        return result
    except Exception as e:
        console.log(f"[red]Error in verification: {str(e)}[/red]")
        return str(e)
    """
}

PROMPT_TEMPLATES = {
    "base_prompt": """<purpose>
    You are a world-class expert at {{purpose_description}}.
    Your goal is to {{goal_description}}.
</purpose>

<instructions>
    <instruction>Follow this closed-loop process:
        1. Validate inputs and approach
        2. Test on small subset/sample
        3. Verify results meet criteria
        4. Generate final output only after validation
    </instruction>
    {{additional_instructions}}
</instructions>

<tools>
    {{tool_descriptions}}
</tools>

<examples>
    {{examples}}
</examples>

<user-request>
    {{user_request}}
</user-request>
"""
}

AGENT_PATTERNS = {
    "compute_scaling": {
        "description": "Patterns for scaling compute usage effectively",
        "features": [
            "parallel_tool_calls",  # Allow multiple tools to run simultaneously
            "compute_loop_control",  # Max iterations and stopping conditions
            "reasoning_effort",      # Configurable reasoning depth
            "model_routing"          # Route to different models based on complexity
        ]
    },
    
    "tool_design": {
        "description": "Best practices for tool interface design",
        "features": [
            "strict_mode",          # Enforce schema validation
            "poka_yoke",           # Make errors impossible by design
            "absolute_paths",      # Use absolute paths to avoid ambiguity
            "natural_formats",     # Keep formats close to what models expect
            "thinking_space"       # Give models enough tokens to plan
        ]
    },

    "verification_patterns": {
        "description": "Patterns for result verification",
        "types": {
            "pre_verification": {
                "description": "Verify before main execution",
                "tools": [
                    "validate_inputs",
                    "check_assumptions",
                    "estimate_resources"
                ]
            },
            "parallel_verification": {
                "description": "Run parallel verifications",
                "tools": [
                    "run_parallel_checks",
                    "cross_validate_results",
                    "compare_approaches"
                ]
            },
            "domain_verification": {
                "description": "Verify against domain knowledge",
                "tools": [
                    "check_physical_bounds",
                    "validate_relationships",
                    "verify_patterns"
                ]
            }
        }
    },

    "prompt_engineering": {
        "description": "Best practices for prompt engineering",
        "sections": {
            "tool_documentation": {
                "required_fields": [
                    "name",
                    "description",
                    "input_schema",
                    "example_usage",
                    "edge_cases",
                    "boundaries"
                ]
            },
            "instruction_clarity": {
                "patterns": [
                    "step_by_step_thinking",
                    "explicit_validation_points",
                    "clear_success_criteria"
                ]
            }
        }
    }
}

TOOL_TEMPLATES.update({
    "parallel_tool": r'''def {{name}}(reasoning: str, {{params}}, parallel_calls: int = 1) -> str:
    """{{description}}
    Supports parallel execution for increased compute utilization.
    """
    try:
        results = []
        with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
            futures = [executor.submit(single_execution, p) for p in partitioned_params]
            results = [f.result() for f in futures]
        console.log(f"[blue]Parallel Execution[/blue] - {{name}} - Calls: {parallel_calls}")
        return aggregate_results(results)
    except Exception as e:
        console.log(f"[red]Error in parallel execution: {str(e)}[/red]")
        return str(e)''',

    "strict_validation_tool": r'''def {{name}}(reasoning: str, {{params}}) -> str:
    """{{description}}
    Implements strict validation with schema enforcement.
    """
    try:
        # Validate against schema
        validated_params = ValidationSchema(**{{params}})
        
        # Execute with validated params
        result = execute_validated(validated_params)
        
        console.log(f"[blue]Strict Validation[/blue] - {{name}} - Reasoning: {reasoning}")
        return result
    except ValidationError as e:
        console.log(f"[red]Schema validation failed: {str(e)}[/red]")
        return str(e)'''
})

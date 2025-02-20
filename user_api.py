# /// script
# dependencies = [
#   'openai>=1.63.0',
#   'rich>=13.7.0',
#   'pydantic>=2.0.0',
#   'concurrent.futures>=3.0.0'
# ]
# ///

"""
User Api Single File Agent

Example Usage:
    uv run user_api.py -d <data_path> -p "your analysis request" -c 5
"""

import os
import sys
import json
import gc
import argparse
import pandas as pd
import numpy as np
import psutil
from typing import List, Dict, Optional, Tuple, Any
from rich.console import Console
from rich.panel import Panel
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# Initialize rich console
console = Console()

# Data Interface Models
class ModelField(BaseModel):
    """Field definition for API model"""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    unique: bool = False
    default: Optional[Any] = None

class Relationship(BaseModel):
    """Model relationship definition"""
    type: str  # one_to_one, one_to_many, many_to_many
    model: str
    back_populates: Optional[str] = None
    cascade: Optional[str] = None

class APIModel(BaseModel):
    """Model definition for API"""
    name: str
    fields: List[ModelField]
    description: Optional[str] = None
    relationships: Optional[List[Relationship]] = None

class Parameter(BaseModel):
    """API endpoint parameter"""
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None

class RequestBody(BaseModel):
    """API endpoint request body"""
    model: str
    required: bool = True
    description: Optional[str] = None

class Endpoint(BaseModel):
    """Endpoint definition for API"""
    path: str
    method: str
    response_model: str
    description: Optional[str] = None
    auth_required: bool = False
    parameters: Optional[List[Parameter]] = None
    request_body: Optional[RequestBody] = None

class AuthConfig(BaseModel):
    """Authentication configuration"""
    auth_type: str  # jwt, oauth2, basic
    secret_key: str
    token_expire_minutes: Optional[int] = 30
    refresh_token_expire_days: Optional[int] = 7

class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str
    type: str  # postgresql, mysql, sqlite, oracle
    pool_size: Optional[int] = 5
    max_overflow: Optional[int] = 10

class APIGeneratorInterface(BaseModel):
    """Interface for API generation configuration"""
    api_name: str
    models: List[APIModel]
    endpoints: List[Endpoint]
    auth_config: Optional[AuthConfig] = None
    database_config: Optional[DatabaseConfig] = None
    
    def load_data(self) -> Dict[str, Any]:
        """Load API configuration data"""
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
        """Validate API schema"""
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
        """Validate API configuration data"""
        try:
            # Schema validation
            if not self.validate_schema():
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

# Memory management
class MemoryManager:
    """Manages memory usage during execution"""
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def check_memory(self) -> bool:
        """Check if memory usage is below threshold"""
        return self.get_memory_usage() < self.threshold_mb
        
    def cleanup(self):
        """Force garbage collection"""
        gc.collect()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Validation chain implementation
def pre_execution_validation(data: Dict, reasoning: str) -> bool:
    """Pre-execution validation checks"""
    try:
        console.log(f"[blue]Pre-execution validation[/blue] - Reasoning: {reasoning}")
        with MemoryManager() as mm:
            for check in ['model_validation', 'endpoint_validation', 'auth_validation']:
                if not globals()[f"validate_{check}"](data, f"Running {check}"):
                    return False
            return True
    except Exception as e:
        console.log(f"[red]Pre-execution validation failed: {str(e)}[/red]")
        return False

def during_execution_validation(data: Dict, reasoning: str) -> bool:
    """During-execution validation checks"""
    try:
        console.log(f"[blue]During-execution validation[/blue] - Reasoning: {reasoning}")
        with MemoryManager() as mm:
            for check in ['code_generation', 'test_generation', 'doc_generation']:
                if not globals()[f"validate_{check}"](data, f"Running {check}"):
                    return False
            return True
    except Exception as e:
        console.log(f"[red]During-execution validation failed: {str(e)}[/red]")
        return False

def post_execution_validation(data: Dict, reasoning: str) -> bool:
    """Post-execution validation checks"""
    try:
        console.log(f"[blue]Post-execution validation[/blue] - Reasoning: {reasoning}")
        with MemoryManager() as mm:
            for check in ['api_testing', 'schema_validation', 'security_check']:
                if not globals()[f"validate_{check}"](data, f"Running {check}"):
                    return False
            return True
    except Exception as e:
        console.log(f"[red]Post-execution validation failed: {str(e)}[/red]")
        return False

# Tool implementations with error handling and progress logging

def generate_models(reasoning: str, data: Dict[str, Any]) -> str:
    """Generate API models from configuration"""
    try:
        # Validate against schema
        validated_params = ValidationSchema(**data)
        
        # Execute with validated params
        result = execute_validated(validated_params)
        
        console.log(f"[blue]Strict Validation[/blue] - generate_models - Reasoning: {reasoning}")
        return result
    except ValidationError as e:
        console.log(f"[red]Schema validation failed: {str(e)}[/red]")
        return str(e)
    
def create_endpoints(reasoning: str, data: Dict[str, Any]) -> str:
    """Create API endpoints from configuration"""
    try:
        # Validate against schema
        validated_params = ValidationSchema(**data)
        
        # Execute with validated params
        result = execute_validated(validated_params)
        
        console.log(f"[blue]Strict Validation[/blue] - create_endpoints - Reasoning: {reasoning}")
        return result
    except ValidationError as e:
        console.log(f"[red]Schema validation failed: {str(e)}[/red]")
        return str(e)
    
def add_authentication(reasoning: str, data: Dict[str, Any]) -> str:
    """Add authentication to API endpoints"""
    try:
        # Validate against schema
        validated_params = ValidationSchema(**data)
        
        # Execute with validated params
        result = execute_validated(validated_params)
        
        console.log(f"[blue]Strict Validation[/blue] - add_authentication - Reasoning: {reasoning}")
        return result
    except ValidationError as e:
        console.log(f"[red]Schema validation failed: {str(e)}[/red]")
        return str(e)
    

def execute_with_compute_control(func, *args, max_compute: int = 5, **kwargs):
    """Execute function with compute control"""
    try:
        with MemoryManager() as mm:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result()
    except Exception as e:
        console.log(f"[red]Compute control error: {str(e)}[/red]")
        return None

# Resource monitoring
def monitor_resources():
    """Monitor system resources during execution"""
    try:
        mm = MemoryManager()
        while True:
            memory_usage = mm.get_memory_usage()
            if memory_usage > mm.threshold_mb:
                console.log(f"[yellow]High memory usage detected: {memory_usage:.2f} MB[/yellow]")
                mm.cleanup()
            time.sleep(1)
    except Exception as e:
        console.log(f"[red]Resource monitoring error: {str(e)}[/red]")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="User Api")
    parser.add_argument("-d", "--data", required=True, help="Path to data file")
    parser.add_argument("-p", "--prompt", required=True, help="The analysis request")
    parser.add_argument("-c", "--compute", type=int, default=5, help="Maximum compute loops")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    args = parser.parse_args()

    try:
        with MemoryManager() as mm:
            # Pre-execution validation
            if not pre_execution_validation({"data_path": args.data}, "Validating input data"):
                sys.exit(1)

            # Execute with compute control and monitoring
            with ThreadPoolExecutor() as executor:
                monitor_future = executor.submit(monitor_resources)
                
                # Main execution with validation
                if not during_execution_validation({}, "Monitoring execution"):
                    sys.exit(1)
                    
                # Your implementation here
                pass

            # Post-execution validation
            if not post_execution_validation({}, "Verifying results"):
                sys.exit(1)

    except Exception as e:
        console.log(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
    finally:
        gc.collect()  # Final cleanup

if __name__ == "__main__":
    main()

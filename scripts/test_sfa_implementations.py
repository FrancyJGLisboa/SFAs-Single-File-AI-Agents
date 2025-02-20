# /// script
# dependencies = [
#   "rich>=13.7.0",
#   "pytest>=7.4.0"
# ]
# ///

import os
import subprocess
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Callable

console = Console()

class SFAVerifier:
    """Verifies Single File Agent implementations against documentation principles."""
    
    def __init__(self):
        self.test_cases = [
            {
                "name": "DuckDB Query Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "query", "duckdb_test",
                    "--model", "openai",
                    "--tools", "list_tables,describe_table,sample_table,run_test_query,run_final_query",
                    "--validation", "strict"
                ],
                "output_file": "duckdb_test.py",
                "verifications": ["validation_chain", "tool_design"]
            },
            {
                "name": "Climate Analysis Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "analysis", "climate_test",
                    "--model", "anthropic",
                    "--tools", "validate_data_integrity,analyze_temperature_trends,analyze_precipitation,verify_results",
                    "--validation", "strict"
                ],
                "output_file": "climate_test.py",
                "verifications": ["compute_scaling", "validation_chain"]
            },
            {
                "name": "Crop Analysis Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "timeseries_analysis", "crop_test",
                    "--model", "openai",
                    "--tools", "query_yield,analyze_crop_stage,correlation_analysis",
                    "--validation", "strict"
                ],
                "output_file": "crop_test.py",
                "verifications": ["data_interface", "validation_chain"]
            },
            {
                "name": "Data Pipeline Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "data_pipeline", "etl_test",
                    "--model", "openai",
                    "--tools", "validate_source,transform_data,load_data,monitor_quality",
                    "--validation", "strict"
                ],
                "output_file": "etl_test.py",
                "verifications": ["data_interface", "validation_chain", "compute_scaling"]
            },
            {
                "name": "LLM App Builder Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "llm_app_builder", "chat_app",
                    "--model", "anthropic",
                    "--tools", "validate_prompts,test_completions,monitor_costs",
                    "--validation", "strict"
                ],
                "output_file": "chat_app.py",
                "verifications": ["tool_design", "validation_chain"]
            },
            {
                "name": "Security Scanner Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "security_scanner", "web_scan",
                    "--model", "openai",
                    "--tools", "scan_dependencies,test_endpoints,validate_auth",
                    "--validation", "strict"
                ],
                "output_file": "web_scan.py",
                "verifications": ["validation_chain", "compute_scaling"]
            },
            {
                "name": "API Generator Agent",
                "command": [
                    "uv", "run", "scripts/sfa_cookiecutter.py", "create", "api_generator", "user_api",
                    "--model", "openai",
                    "--tools", "generate_models,create_endpoints,add_authentication",
                    "--validation", "strict"
                ],
                "output_file": "user_api.py",
                "verifications": ["tool_design", "data_interface"]
            }
        ]
        
    def verify_validation_chain(self, content: str) -> Dict[str, bool]:
        """Verifies proper validation chain implementation."""
        checks = {
            "pre_execution": "pre_execution" in content or "validate_" in content,
            "during_execution": "during_execution" in content or "test_" in content,
            "post_execution": "post_execution" in content or "verify_" in content,
            "reasoning_parameter": "reasoning: str" in content,
            "error_handling": "try:" in content and "except" in content,
            "progress_logging": "console.log" in content,
            "strict_mode": "strict" in content
        }
        return checks

    def verify_tool_design(self, content: str) -> Dict[str, bool]:
        """Verifies tool design follows best practices."""
        checks = {
            "reasoning_parameter": "reasoning: str" in content,
            "error_handling": "try:" in content and "except" in content,
            "progress_logging": "console.log" in content,
            "schema_validation": "validate_schema" in content or "pydantic" in content,
            "strict_mode": "strict" in content,
            "parallel_execution": "ThreadPoolExecutor" in content,
            "resource_monitoring": "monitor_resources" in content
        }
        return checks

    def verify_compute_scaling(self, content: str) -> Dict[str, bool]:
        """Verifies compute scaling implementation."""
        checks = {
            "parallel_execution": "parallel" in content or "ThreadPoolExecutor" in content,
            "compute_control": "compute_iterations" in content or "max_compute" in content,
            "resource_monitoring": "monitor_resources" in content,
            "error_handling": "try:" in content and "except" in content,
            "progress_tracking": "console.log" in content,
            "memory_management": "gc.collect" in content or "__del__" in content
        }
        return checks

    def verify_data_interface(self, content: str) -> Dict[str, bool]:
        """Verifies data interface implementation."""
        checks = {
            "interface_definition": "class" in content and "Interface" in content,
            "data_loading": "load_data" in content,
            "data_validation": "validate_data" in content,
            "schema_validation": "validate_schema" in content,
            "error_handling": "try:" in content and "except" in content,
            "type_safety": "BaseModel" in content or "pydantic" in content,
            "monitoring": "console.log" in content
        }
        return checks

    def run_verification(self, test_case: Dict) -> Dict:
        """Runs verification for a single test case."""
        try:
            # Create SFA using cookiecutter
            result = subprocess.run(
                test_case["command"],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Verify file exists
            if not os.path.exists(test_case["output_file"]):
                console.log(f"[red]Command output: {result.stdout}[/red]")
                console.log(f"[red]Command error: {result.stderr}[/red]")
                return {
                    "status": "failed",
                    "error": f"File not created: {test_case['output_file']}\nOutput: {result.stdout}\nError: {result.stderr}"
                }
            
            # Read and verify content
            with open(test_case["output_file"]) as f:
                content = f.read()
            
            results = {}
            for verification in test_case["verifications"]:
                verify_func = getattr(self, f"verify_{verification}")
                results[verification] = verify_func(content)
            
            return {"status": "success", "results": results}
        
        except subprocess.CalledProcessError as e:
            console.log(f"[red]Command output: {e.output}[/red]")
            console.log(f"[red]Command error: {e.stderr}[/red]")
            return {
                "status": "failed",
                "error": f"Command failed: {e.cmd}\nOutput: {e.output}\nError: {e.stderr}"
            }
        except Exception as e:
            console.log(f"[red]Error details: {str(e)}[/red]")
            return {
                "status": "failed",
                "error": f"Unexpected error: {str(e)}"
            }

    def run_all_tests(self) -> None:
        """Runs all test cases and displays results."""
        results_table = Table(title="SFA Implementation Test Results")
        results_table.add_column("Agent Type", style="cyan")
        results_table.add_column("Verification", style="magenta")
        results_table.add_column("Check", style="yellow")
        results_table.add_column("Result", style="green")

        for test_case in self.test_cases:
            console.print(f"\n[bold blue]Testing {test_case['name']}...[/bold blue]")
            
            result = self.run_verification(test_case)
            
            if result["status"] == "failed":
                console.print(f"[red]{result['error']}[/red]")
                continue

            for verification, checks in result["results"].items():
                for check, passed in checks.items():
                    results_table.add_row(
                        test_case["name"],
                        verification,
                        check,
                        "✅" if passed else "❌"
                    )

        console.print("\n")
        console.print(results_table)

def main():
    """Main test execution."""
    console.print("[bold green]Starting SFA Implementation Tests[/bold green]")
    console.print("Verifying implementations against documentation principles...")
    
    verifier = SFAVerifier()
    verifier.run_all_tests()

if __name__ == "__main__":
    main() 
"""Sandbox implementation for executing GPU code safely."""
import ast
import contextlib
import io
import sys
import time
from typing import Any, Tuple

class GPUSandbox:
    """Sandbox for executing GPU code safely."""
    
    def __init__(self):
        self._allowed_modules = {
            'torch', 'numpy', 'random', 'math', 'typing', 'enum',
            'matplotlib', 'os', 'datetime', 'itertools', 'collections'
        }
        self._allowed_functions = {
            'range', 'len', 'max', 'min', 'sum', 'all', 'any',
            'enumerate', 'zip', 'map', 'filter', 'sorted',
            'random', 'randint', 'shuffle', 'choice', 'sample',
            'randrange', 'uniform', 'gauss', 'expovariate',
            'triangular', 'betavariate', 'gammavariate', 'lognormvariate',
            'normalvariate', 'vonmisesvariate', 'paretovariate',
            'weibullvariate', 'getrandbits', 'randbytes'
        }
    
    def _is_safe_import(self, node: ast.Import) -> bool:
        """Check if an import statement is safe."""
        for name in node.names:
            if name.name not in self._allowed_modules:
                print(f"Blocked import of module: {name.name}", file=sys.stderr)
                return False
        return True
    
    def _is_safe_import_from(self, node: ast.ImportFrom) -> bool:
        """Check if an import from statement is safe."""
        if node.module not in self._allowed_modules:
            print(f"Blocked import from module: {node.module}", file=sys.stderr)
            return False
        for name in node.names:
            if name.name not in self._allowed_functions:
                print(f"Blocked import of function: {name.name} from {node.module}", file=sys.stderr)
                return False
        return True
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if the code is safe to execute."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if not self._is_safe_import(node):
                        return False
                elif isinstance(node, ast.ImportFrom):
                    if not self._is_safe_import_from(node):
                        return False
                elif isinstance(node, ast.Call):
                    # Check for potentially dangerous function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in self._allowed_functions:
                            print(f"Blocked function call: {node.func.id}", file=sys.stderr)
                            return False
            return True
        except SyntaxError as e:
            print(f"Syntax error in code: {e}", file=sys.stderr)
            return False
    
    def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any,
        timeout_seconds: int,
    ) -> Tuple[Any, bool]:
        """Execute the program safely and return its output."""
        if not self._is_safe_code(program):
            return None, False
        
        # Create a new namespace for execution
        namespace = {}
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                # Execute the program in the new namespace
                exec(program, namespace)
                
                # Get the function to run
                func = namespace.get(function_to_run)
                if func is None:
                    print(f"Function {function_to_run} not found in program", file=sys.stderr)
                    return None, False
                
                # Run the function with timeout
                start_time = time.time()
                result = func(*test_input if isinstance(test_input, tuple) else (test_input,))
                elapsed = time.time() - start_time
                
                if elapsed > timeout_seconds:
                    print(f"Function execution timed out after {timeout_seconds} seconds", file=sys.stderr)
                    return None, False
                
                return result, True
                
        except Exception as e:
            print(f"Error in sandbox: {str(e)}", file=sys.stderr)
            print(f"Program:\n{program}", file=sys.stderr)
            return None, False 
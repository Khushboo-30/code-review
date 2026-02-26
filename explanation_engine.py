import ast

class ExplanationEngine:
    def __init__(self):
        # Heuristics for simple risk highlights (unsafe Python practices)
        self.risky_functions = ['exec', 'eval', 'subprocess', 'system', 'popen']
        self.risky_modules = ['pickle', 'os', 'subprocess', 'sys']

    def highlight_risky_lines(self, code_snippet):
        highlights = []
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.risky_functions:
                        highlights.append((node.lineno, f"Use of potentially unsafe function: `{node.func.id}`"))
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in self.risky_functions:
                        highlights.append((node.lineno, f"Use of potentially unsafe method: `.{node.func.attr}`"))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.risky_modules:
                            highlights.append((node.lineno, f"Import of potentially unsafe module: `{alias.name}`"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.risky_modules:
                        highlights.append((node.lineno, f"Import from potentially unsafe module: `{node.module}`"))
        except SyntaxError:
            # Fallback to simple string matching for C/C++ or invalid Python
            lines = code_snippet.split('\n')
            c_risky_funcs = ['gets', 'strcpy', 'sprintf', 'scanf', 'system', 'popen']
            for i, line in enumerate(lines):
                for func in c_risky_funcs:
                    if func + "(" in line:
                        highlights.append((i + 1, f"Use of potentially unsafe C function: `{func}`"))
        return highlights

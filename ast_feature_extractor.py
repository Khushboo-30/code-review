import ast

class ASTFeatureExtractor:
    def __init__(self):
        # We extract counts of specific AST node types as simple structural features
        self.feature_names = [
            "FunctionDef", "ClassDef", "Return", "Delete", "Assign", "For", "While", 
            "If", "With", "Raise", "Try", "Assert", "Import", "Call", "Compare"
        ]

    def extract_features(self, code_snippet):
        features = {name: 0 for name in self.feature_names}
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                node_type = type(node).__name__
                if node_type in features:
                    features[node_type] += 1
        except SyntaxError:
            # If code is invalid or not Python (e.g. C), return zeroes
            # In a full multi-language implementation, we would use tree-sitter here
            pass
        
        # Convert to list
        return [features[name] for name in self.feature_names]

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

CSS_QUERIES = {
    "rulesets": """
        (rule_set) @rule_set
    """,
    "selectors": """
        (selectors) @selectors
        (class_selector) @class_selector
        (tag_name) @tag_name
    """,
    "declarations": """
        (declaration) @declaration
    """,
    "at_rules": """
        (media_statement) @media_statement
    """,
    "comments": """
        (comment) @comment
    """,
    "values": """
        (plain_value) @plain_value
        (integer_value) @integer_value
        (float_value) @float_value
        (string_value) @string_value
    """,
}

class CSSTreeSitterParser:
    """A CSS-specific parser using tree-sitter, encapsulating language-specific logic."""

    def __init__(self, generic_parser_wrapper):
        self.generic_parser_wrapper = generic_parser_wrapper
        self.language_name = generic_parser_wrapper.language_name
        self.language = generic_parser_wrapper.language
        self.parser = generic_parser_wrapper.parser

        self.queries = {
            name: self.language.query(query_str)
            for name, query_str in CSS_QUERIES.items()
        }

    def _get_node_text(self, node) -> str:
        return node.text.decode('utf-8')

    def _get_parent_context(self, node, types=('rule_set', 'at_rule', 'media_statement')):
        """Get parent context for CSS constructs like nested rules."""
        curr = node.parent
        while curr:
            if curr.type in types:
                # For CSS, we'll use the selector text as context
                if curr.type == 'rule_set':
                    selectors_node = curr.child_by_field_name('selectors')
                    if selectors_node:
                        return self._get_node_text(selectors_node), curr.type, curr.start_point[0] + 1
                elif curr.type in ('at_rule', 'media_statement'):
                    name_node = curr.child_by_field_name('name')
                    if name_node:
                        return self._get_node_text(name_node), curr.type, curr.start_point[0] + 1
            curr = curr.parent
        return None, None, None

    def _calculate_complexity(self, node):
        """Calculate complexity for CSS rules (based on selectors and properties)."""
        complexity_nodes = {
            "selectors", "simple_selector", "class_selector", "id_selector",
            "type_selector", "attribute_selector", "pseudo_class_selector",
            "pseudo_element_selector", "declaration"
        }
        count = 1
        
        def traverse(n):
            nonlocal count
            if n.type in complexity_nodes:
                count += 1
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return count

    def _get_docstring(self, node):
        """Extract comments as documentation for CSS rules."""
        # Look for comments near the node
        if node.parent:
            for child in node.parent.children:
                if child.type == 'comment' and child.start_point[0] < node.start_point[0]:
                    return self._get_node_text(child)
        return None

    def parse(self, file_path: Path, is_dependency: bool = False) -> Dict:
        """Parses a CSS file and returns its structure in a standardized dictionary format."""
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        tree = self.parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node

        # CSS doesn't have traditional functions, so we'll map CSS constructs to the expected structure
        functions = self._find_css_functions(root_node)  # CSS functions like calc(), var(), etc.
        classes = self._find_css_classes(root_node)     # CSS classes and selectors
        imports = self._find_css_imports(root_node)     # @import statements
        function_calls = self._find_css_function_calls(root_node)  # CSS function calls
        variables = self._find_css_variables(root_node)  # CSS custom properties

        return {
            "file_path": str(file_path),
            "functions": functions,
            "classes": classes,
            "variables": variables,
            "imports": imports,
            "function_calls": function_calls,
            "is_dependency": is_dependency,
            "lang": self.language_name,
        }

    def _find_css_functions(self, root_node):
        """Find CSS functions like calc(), var(), url(), etc."""
        functions = []
        
        # Look for call_expression nodes directly
        def find_call_expressions(node):
            if node.type == 'call_expression':
                # Find the function name
                function_name_node = None
                for child in node.children:
                    if child.type == 'function_name':
                        function_name_node = child
                        break
                
                if function_name_node:
                    func_name = self._get_node_text(function_name_node)
                    
                    # Find the property this function is used in
                    property_name = None
                    parent_declaration = node.parent
                    while parent_declaration and parent_declaration.type != 'declaration':
                        parent_declaration = parent_declaration.parent
                    
                    if parent_declaration:
                        for child in parent_declaration.children:
                            if child.type == 'property_name':
                                property_name = self._get_node_text(child)
                                break
                    
                    if property_name:
                        context, context_type, _ = self._get_parent_context(parent_declaration)
                        
                        func_data = {
                            "name": f"{property_name}_{func_name}",
                            "line_number": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "args": [property_name],
                            "source": self._get_node_text(node),
                            "source_code": self._get_node_text(node),
                            "docstring": self._get_docstring(node),
                            "cyclomatic_complexity": self._calculate_complexity(node),
                            "context": context,
                            "context_type": context_type,
                            "class_context": None,
                            "decorators": [],
                            "lang": self.language_name,
                            "is_dependency": False,
                        }
                        functions.append(func_data)
            
            for child in node.children:
                find_call_expressions(child)
        
        find_call_expressions(root_node)
        return functions

    def _find_css_classes(self, root_node):
        """Find CSS classes, selectors, and at-rules."""
        classes = []
        query = self.queries.get('rulesets')
        if not query:
            return []

        for match in query.captures(root_node):
            capture_name = match[1]
            node = match[0]

            if capture_name == 'rule_set':
                # Find selectors node
                selectors_node = None
                for child in node.children:
                    if child.type == 'selectors':
                        selectors_node = child
                        break
                
                if selectors_node:
                    selector_text = self._get_node_text(selectors_node)
                    context, _, _ = self._get_parent_context(node)
                    
                    class_data = {
                        "name": selector_text,
                        "line_number": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "bases": [],  # CSS doesn't have inheritance like OOP
                        "source": self._get_node_text(node),
                        "docstring": self._get_docstring(node),
                        "context": context,
                        "decorators": [],
                        "lang": self.language_name,
                        "is_dependency": False,
                    }
                    classes.append(class_data)

        # Also capture at-rules as "classes"
        at_rules_query = self.queries.get('at_rules')
        if at_rules_query:
            for match in at_rules_query.captures(root_node):
                capture_name = match[1]
                node = match[0]

                if capture_name == 'media_statement':
                    # Extract media query text
                    media_text = self._get_node_text(node)
                    context, _, _ = self._get_parent_context(node)
                    
                    class_data = {
                        "name": media_text.split('{')[0].strip(),  # Get the media query part
                        "line_number": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "bases": [],
                        "source": self._get_node_text(node),
                        "docstring": self._get_docstring(node),
                        "context": context,
                        "decorators": [],
                        "lang": self.language_name,
                        "is_dependency": False,
                    }
                    classes.append(class_data)

        return classes

    def _find_css_imports(self, root_node):
        """Find CSS @import statements."""
        imports = []
        # Look for import statements manually since they might not be captured by our query
        def find_imports_recursive(node):
            if node.type == 'import_statement':
                # Extract the URL from the import statement
                import_text = self._get_node_text(node)
                # Simple extraction of URL from @import "url" or @import url("url")
                import re
                url_match = re.search(r'@import\s+(?:url\()?["\']([^"\']+)["\']?\)?', import_text)
                if url_match:
                    import_url = url_match.group(1)
                    import_data = {
                        "name": import_url,
                        "full_import_name": import_url,
                        "line_number": node.start_point[0] + 1,
                        "alias": None,
                        "context": self._get_parent_context(node)[:2],
                        "lang": self.language_name,
                        "is_dependency": False,
                    }
                    imports.append(import_data)
            
            for child in node.children:
                find_imports_recursive(child)
        
        find_imports_recursive(root_node)
        return imports

    def _find_css_function_calls(self, root_node):
        """Find CSS function calls like calc(), var(), etc."""
        calls = []
        
        # Look for call_expression nodes directly
        def find_call_expressions(node):
            if node.type == 'call_expression':
                # Find the function name
                function_name_node = None
                for child in node.children:
                    if child.type == 'function_name':
                        function_name_node = child
                        break
                
                if function_name_node:
                    func_name = self._get_node_text(function_name_node)
                    
                    call_data = {
                        "name": func_name,
                        "full_name": f"{func_name}()",
                        "line_number": node.start_point[0] + 1,
                        "args": [],
                        "inferred_obj_type": None,
                        "context": self._get_parent_context(node),
                        "class_context": None,
                        "lang": self.language_name,
                        "is_dependency": False,
                    }
                    calls.append(call_data)
            
            for child in node.children:
                find_call_expressions(child)
        
        find_call_expressions(root_node)
        return calls

    def _find_css_variables(self, root_node):
        """Find CSS custom properties (variables)."""
        variables = []
        query = self.queries.get('declarations')
        if not query:
            return []

        for match in query.captures(root_node):
            capture_name = match[1]
            node = match[0]

            if capture_name == 'declaration':
                # Find property name and value
                property_name = None
                value_text = ""
                
                for child in node.children:
                    if child.type == 'property_name':
                        property_name = self._get_node_text(child)
                    elif child.type in ('plain_value', 'integer_value', 'float_value', 'string_value'):
                        value_text += self._get_node_text(child)
                
                if property_name and value_text:
                    # Check if it's a CSS custom property (starts with --)
                    if property_name.startswith('--'):
                        context, _, _ = self._get_parent_context(node)
                        class_context, _, _ = self._get_parent_context(node, types=('rule_set',))
                        
                        variable_data = {
                            "name": property_name,
                            "line_number": node.start_point[0] + 1,
                            "value": value_text,
                            "type": "css-custom-property",
                            "context": context,
                            "class_context": class_context,
                            "lang": self.language_name,
                            "is_dependency": False,
                        }
                        variables.append(variable_data)

        return variables


def pre_scan_css(files: list[Path], parser_wrapper) -> dict:
    """Scans CSS files to create a map of selectors and at-rules to their file paths."""
    imports_map = {}
    query_str = """
        (rule_set selectors: (selectors) @selectors)
        (media_statement) @media_statement
    """
    query = parser_wrapper.language.query(query_str)
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = parser_wrapper.parser.parse(bytes(f.read(), "utf8"))
            
            for capture, _ in query.captures(tree.root_node):
                name = capture.text.decode('utf-8')
                if name not in imports_map:
                    imports_map[name] = []
                imports_map[name].append(str(file_path.resolve()))
                
            # Also look for import statements manually
            def find_imports_recursive(node):
                if node.type == 'import_statement':
                    import_text = node.text.decode('utf-8')
                    import re
                    url_match = re.search(r'@import\s+(?:url\()?["\']([^"\']+)["\']?\)?', import_text)
                    if url_match:
                        import_url = url_match.group(1)
                        if import_url not in imports_map:
                            imports_map[import_url] = []
                        imports_map[import_url].append(str(file_path.resolve()))
                
                for child in node.children:
                    find_imports_recursive(child)
            
            find_imports_recursive(tree.root_node)
            
        except Exception as e:
            logger.warning(f"Tree-sitter pre-scan failed for {file_path}: {e}")
    return imports_map

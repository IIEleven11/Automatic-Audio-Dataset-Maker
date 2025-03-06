import pkg_resources
import ast
import os
from pathlib import Path

def get_imports_from_file(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def get_project_imports():
    imports = set()
    for path in Path('.').rglob('*.py'):
        if '.venv' not in str(path):
            imports.update(get_imports_from_file(path))
    return imports

def get_installed_packages():
    return {pkg.key for pkg in pkg_resources.working_set}

def read_requirements():
    with open('requirements.txt') as f:
        return {line.split('==')[0].lower() for line in f if line.strip() and not line.startswith('#')}

project_imports = get_project_imports()
installed_packages = get_installed_packages()
requirements = read_requirements()

unused_packages = requirements - project_imports
print("\nPotentially unused packages in requirements.txt:")
for package in sorted(unused_packages):
    print(f"- {package}")

print("\nNote: Some packages might be indirect dependencies or used in non-import ways.")
#!/usr/bin/env python3
"""
Script to convert Python file with cell markers to Jupyter notebook format.
"""

import json
import re
import sys
from pathlib import Path

def convert_python_to_notebook(python_file_path, output_notebook_path=None):
    """
    Convert a Python file with # %% cell markers to Jupyter notebook format.
    
    Args:
        python_file_path: Path to the Python file
        output_notebook_path: Path for the output notebook (optional)
    """
    
    if output_notebook_path is None:
        output_notebook_path = python_file_path.replace('.py', '.ipynb')
    
    # Read the Python file
    with open(python_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into cells
    cell_pattern = r'# %% \[(markdown|code)\](.*?)(?=# %% \[|$)'
    cells = re.findall(cell_pattern, content, re.DOTALL)
    
    # If no cells found, try alternative pattern
    if not cells:
        cell_pattern = r'# %% \[(markdown|code)\](.*?)(?=# %%|$)'
        cells = re.findall(cell_pattern, content, re.DOTALL)
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Process each cell
    for cell_type, cell_content in cells:
        cell_content = cell_content.strip()
        
        if cell_type == 'markdown':
            # Markdown cell
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": cell_content
            }
        else:
            # Code cell
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_content
            }
        
        notebook["cells"].append(cell)
    
    # Write the notebook
    with open(output_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {python_file_path} to {output_notebook_path}")
    print(f"📊 Created {len(notebook['cells'])} cells")

def main():
    """Main function to convert Python files to notebooks."""
    
    # List of files to convert
    files_to_convert = [
        ("notebooks/01_exploratory_data_analysis.py", "notebooks/01_exploratory_data_analysis.ipynb"),
        ("notebooks/02_feature_engineering.py", "notebooks/02_feature_engineering.ipynb"),
        ("notebooks/03_model_evaluation.py", "notebooks/03_model_evaluation.ipynb")
    ]
    
    for python_file, output_notebook in files_to_convert:
        # Check if file exists
        if not Path(python_file).exists():
            print(f"❌ Python file not found: {python_file}")
            continue
        
        # Convert the file
        try:
            convert_python_to_notebook(python_file, output_notebook)
            print(f"🎉 Successfully created notebook: {output_notebook}")
        except Exception as e:
            print(f"❌ Error converting file {python_file}: {e}")
            continue

if __name__ == "__main__":
    main() 
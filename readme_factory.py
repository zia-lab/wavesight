#!/usr/bin/env python3

import os
import ast

readme_header = '''# wavesight

`wavesight` is a set of tools that may be used to the folowing ends:

* to find the modes in idealized optical waveguides,
* estimate the launch fields once they undergo refraction across a planar interface,
* and calculate the propagation of these fields across a homogeneous space using an implementation of the Rayleigh-Sommerfeld diffraction integral.

<center> <img src="./img/aMode.png" style="width: 100%;"/> </center>

'''

excluding = ['readme_factory.py']

def extract_function_data(node):
    """Extract the function name, parameters and docstring."""
    name = node.name
    params = [arg.arg for arg in node.args.args]
    docstring = ast.get_docstring(node)
    
    return name, params, docstring

def extract_from_source(source):
    """Extract function data from the source code."""
    tree = ast.parse(source)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    function_data = {extract_function_data(f)[0]: {"params": extract_function_data(f)[1], "docstring": extract_function_data(f)[2]} for f in functions}

    return function_data

def extract_from_directory(directory, exclude=[]):
    """Extract function data from all .py files in a directory."""
    function_data = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.py') and (filename not in exclude):
            with open(os.path.join(directory, filename), 'r') as f:
                source = f.read()
                function_data[filename] = extract_from_source(source)
    
    return function_data

def format_markdown(function_data):
    """Formats the function data as markdown."""
    markdown = ""
    
    for file, functions in function_data.items():
        markdown += f"## File: {file}\n"
        
        for name, data in functions.items():
            markdown += f"### {name}({',  '.join(data['params'])})\n"
            if data['docstring']:
                markdown += f"```Docstring:\n{data['docstring']}\n```\n"
            else:
                markdown += ""
    
    return markdown

def main():
    directory = os.getcwd()
    function_data = extract_from_directory(directory, exclude=excluding)
    markdown = format_markdown(function_data)
    readme_bits = [readme_header, markdown]
    readme = "\n".join(readme_bits)
    with open("README.md", "w") as f:
        f.write(readme)

if __name__ == "__main__":
    main()

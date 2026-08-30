import os
from jsmin import jsmin

def concatenate_and_minify_js_files(directory, output_file):
    # Get a list of all JavaScript files in the directory and its subdirectories
    js_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.js'):
                js_files.append(os.path.join(root, file))

    # Concatenate all JavaScript files into a single string
    concatenated_code = ''
    for js_file in js_files:
        with open(js_file, 'r') as f:
            concatenated_code += f.read() + '\n'

    # Minify the concatenated code
    minified_code = jsmin(concatenated_code)

    # Write the minified code to the output file
    with open(output_file, 'w') as f:
        f.write(minified_code)

# Example usage
directory_to_concatenate = './src/js'
output_minified_file = './dist/js/analytic.min.js'
concatenate_and_minify_js_files(directory_to_concatenate, output_minified_file)

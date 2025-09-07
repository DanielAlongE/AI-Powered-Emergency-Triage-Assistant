import os
import sys

# Get the absolute path of the directory containing the current script
# which is '/my_project/module_b/'
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory, which is '/my_project/'
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now, you can import from the sibling directory 'module_a'
from playground import whisper_debug


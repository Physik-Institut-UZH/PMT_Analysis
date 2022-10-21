"""
Test configuration file. Add the root dir to the python path.
"""
import sys
from os.path import dirname as d
from os.path import abspath
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)

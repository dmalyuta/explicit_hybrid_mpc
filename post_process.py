"""
Data post-processing.
"""

import sys
sys.path.append('./lib/')

import pickle

import global_vars

data = []
with open(global_vars.STATISTICS_FILE,'rb') as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
print(len(data))
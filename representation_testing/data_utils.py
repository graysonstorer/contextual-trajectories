import numpy as np
import pandas as pd

# data = pd.read_csv('../Corpus/modified_stimulus.csv') # for pyCharm
data = pd.read_csv('Corpus/modified_stimulus.csv') # for command line

gp = data[data['Ambiguity'] == 'A']
ngp = data[data['Ambiguity'] == 'U']
gp = gp['Stimulus']
ngp = ngp['Stimulus']

print("gp looks like this: ", gp.head())
print("ngp looks like this: ", ngp.head())
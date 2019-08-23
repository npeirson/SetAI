import numpy as np
import pandas as pd
import itertools

data = pd.DataFrame()

colors = ['red','blue','blue','blue','red','red','blue','green','green','red','blue','green']
counts = ['two','one','two','three','one','one','two','one','one','two','one','two']
shades = ['partial','full','full','full','partial','empty','full','full','partial','partial','partial','full']
shapes = ['squiggle','oval','oval','oval','oval','oval','squiggle','diamond','squiggle','oval','diamond','oval']

data['colors'] = colors
data['counts'] = counts
data['shades'] = shades
data['shapes'] = shapes

'''
column_permutations = list(itertools.permutations(['colors','counts','shades','shapes']))
for permutation in column_permutations:
	if (data.duplicated(permutation[:-1]).value_counts()[1] >= 3):
		print('WIN')
		break
'''
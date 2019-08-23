import os
import pandas as pd


def get_set(path):
	data = pd.DataFrame()
	for dirpath, subdirs, files in os.walk(path):
		for file in files:
			filepath = os.path.join(dirpath,file)
			strpath = str(filepath)
			strpath = strpath.replace('/','\\').split('\\') # windows specific
			class_num = strpath[1]
			class_color = strpath[2]
			class_shape = strpath[3]
			class_shade = strpath[4]
			row = pd.Series({'filepath':filepath,
							 'class_num':class_num,
							 'class_color':class_color,
							 'class_shape':class_shape,
							 'class_shade':class_shade})
			data = data.append(row,ignore_index=True)
	return data

get_set('dataset/')
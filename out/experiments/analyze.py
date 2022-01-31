import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np


#post_name = '-normal.csv'
post_name = '.csv'


#pre_name  = 'glove.840B.300d_taskA/'
#pre_name  = 'tweet200d_taskB/'
#pre_name  = 'glove.840B.300d_taskB/'
#pre_name  = 'recursive_A/'
#pre_name  = 'recursive-LSTM-A-v2/'
pre_name  = 'B final/'



onlyfiles = [f for f in listdir(pre_name) if isfile(join(pre_name, f))]

data = []
data.append(pd.read_csv(pre_name + str(1) + post_name, delimiter=',', header=None))
for i in range(1,7):
    data.append(pd.read_csv(pre_name + str(i) + post_name, delimiter=',', skiprows=1, header=None))


result = pd.concat(data, axis=0, ignore_index=True)

result.to_csv(pre_name + 'results' + post_name)
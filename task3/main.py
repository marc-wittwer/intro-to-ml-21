import pandas as pd

#
# read data (as a pd.DataFrame)
#
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Split to have sequence and active (label) separated
train_sequence = train_data['Sequence']
train_active = train_data['Active']

test_sequence = test_data['Sequence']


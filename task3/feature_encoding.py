import pandas as pd

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Split to have sequence (Each protein on its own) and active (label) separated
train_sequence = train_data['Sequence'].apply(lambda x: pd.Series(list(x)))
train_active = train_data['Active']

test_sequence = test_data['Sequence'].apply(lambda x: pd.Series(list(x)))

# Map Aminoacids to numbers according to feature encoding

# Label 'U' does not exist in training/test data
feature_encoding_dict = {
        'R': 1, 'H': 2, 'K': 3, 'D': 4, 'E': 5,
        'S': 6, 'T': 7, 'N': 8, 'Q': 9, 'C': 10, 'U': 11, 'G': 12, 'P': 13,
        'A': 14, 'I': 15, 'L': 16, 'M': 17, 'F': 18, 'W': 19, 'Y': 20, 'V': 21
        }

train_sequence = train_sequence.stack().map(feature_encoding_dict).unstack()
test_sequence = test_sequence.stack().map(feature_encoding_dict).unstack()

# Save to file
train_sequence.to_csv('data/train_encoded_features.csv', index=False)
test_sequence.to_csv('data/test_encoded_features.csv', index=False)
train_active.to_csv('data/train_labels.csv', index=False)

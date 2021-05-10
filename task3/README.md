**Instructions:**

1. Run the `feature_encoding.py` script
2. Run the `main.py` script
3. See output file `prediction.csv`

**Explanation:**

The code in `feature_encoding.py` separates the train labels from the
train features. The train and test features (four-letter strings describing 
the mutations) are split into separate columns and encoded using integer encoding. 
The resulting data is stored in the files `train_encoded_features.csv`,
`train_labels.csv` and `test_encoded_features.csv`.

The pre-processed data files are imported in `main.py` where classifiers
to predict the protein activity are trained.
The `GridSearchCV` class is used to evaluate the performance of different 
classifiers (e.g. `RandomForestClassifier` and `HistGradientBoostingClassifier`) 
with different sets of hyper parameters. The classifiers and hyper parameter 
settings to be compared are defined in the `grid_search_params` variable.

All classifiers are evaluated using 5-fold cross validation and the F1 score.
The best one is selected to predict the activity of each mutation in the test
set. Finally, the predictions are stored in `predictions.csv`.

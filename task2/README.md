Instructions:

1. Make sure that all the packages from the `requirements.txt`
   are installed
2. Run the `feature_extraction.py` script
3. Run the `main.py` script
4. See output file `prediction.csv`

Explanation:

The code in `feature_extraction.py` extracts the following
features from the time-series data for each patient.

- Median per column
- Standard deviation per column
- Minimum value per column
- Maximum per column
- Number of Nan values per column

The result gets stored in the files `train_extracted_features.csv`,
`train_extracted_labels.csv` and `test_extracted_features.csv`.

These data files get imported in `main.py` where the models 
for each label will be trained. For the classification task
1 and 2 the SVM classifier `SVC()` was used.

First, the missing values in the data set were imputed using
the column mean strategy. Then, the data was normalized 
with a `StandardScaler` to have mean of 0 and std of 1. To 
speed up the training uncorrelated columns were dropped by
using a selector. As a last step the SVC model was fitted
to the training data. The best hyperparameters for the 
model were found by using grid search with internal 
cross-validation `GridSearchCV`. The  code for the grid 
search can be found in `grid_search.py`. 

For the regression task 3, the missing values were also imputed
with the mean strategy and afterwards normalized with the
`StandardScaler`. A linear regression with Lasso regularization
was used for to fit the model to the training data. The lambdas
were optimized using `GridSearchCV`.
 
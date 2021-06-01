# Task 4

The goal of this task is to decide for a triplet ABC of food images, whether B or
C is more similar in taste to A.

Our approach to this task was extracting image features for all the test and train 
images using a pretrained vision model (``mobilenet_v3_small`` trained on ``ImageNet``),
formulating the task as a binary classification problem and training a simple
neural network to solve it.

The input to our neural net is a feature vector containing concatenated features 
for images A, B and C. The output is 1 if B is more similar to A and 0 if C is.

## Reproduce predictions
In order to reproduce our hand-in predictions, make sure the files 
`train_triplets.txt` and `test_triplets.txt` as well as the food images (in 
their own `food` directory) are located in a subdirectory ``data``.

- run ``feature_extraction.py``
- run ``main.py``

The file ``predictions.csv`` containing our predictions on the test data
will be saved in the ``data`` folder.
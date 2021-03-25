import torch
import torch.nn as nn
import pandas as pd
import numpy as numpy
import tables
from torch.autograd import Variable


# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 120, 50, 45324, 50

#Import data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")


# Split the training data into y and X.
X_train = train.drop(["y"], axis=1)
y_train = train[["y"]]
# Split test data into X.
X_test = test



#Generate numpy arrays
x=train.iloc[:,1:121].values
y=train.iloc[:,0].values
x_test=test.iloc[:,:].values



#Bring into final right from

x = Variable(torch.from_numpy(x))
y = Variable(torch.from_numpy(y))
x_test = Variable(torch.from_numpy(x_test))

type(y)
y = y.to(dtype=torch.long)
type(y)

print(type(y))

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())

# Construct the loss function
criterion = torch.nn.CrossEntropyLoss()

# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Gradient Descent
for epoch in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()

final_pred = model(x_test)

#Prepare submission file.
submission= pandas.DataFrame(columns=['Id', 'y'])
i=0
for x in final_pred:
    submission.loc[i] = [str(i+45324),(y_pred[i])]
    i = i + 1
df = pandas.DataFrame(["Id", "y"])
submission.appendr3et45r(df)

submission.to_csv("submission_torch.csv", index=False, header=True)
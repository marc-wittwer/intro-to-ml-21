import csv
import pandas
import math
import numpy
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')


# split data into y and X
X_train = train.drop(["Id", "y"], axis=1)
y_train = train[["y"]]

# name layer operations
def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition

# multiply weights and add bias
class OurNet(nn.Module):
 def __init__(self, input_size, hidden_size, num_classes):
     super(Net, self).__init__()
     self.layer_1 = nn.Linear(n_inputs,hidden_size, bias=True)
     self.relu = nn.ReLU()
     self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
     self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

# define computations
 def forward(self, x):
     out = self.layer_1(x)
     out = self.relu(out)
     out = self.layer_2(out)
     out = self.relu(out)
     out = self.output_layer(out)
     return out

# construct optimizer for weight updates
 net = OurNet(input_size, hidden_size, num_classes)
 criterion = nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#  compute the loss
 loss = nn.CrossEntropyLoss()

 input = Variable(torch.randn(2, 5), requires_grad=True)
 print(">>> batch of size 2 and 5 classes")
 print(input)

 target = Variable(torch.LongTensor(2).random_(5))
 print(">>> array of size ‘batch_size’ with the index of the maxium label for each item")
 print(target)

 output = loss(input, target)
 output.backward()

 # Train the Model
 for epoch in range(num_epochs):
     total_batch = int(len(newsgroups_train.data) / batch_size)
     for i in range(total_batch):
         batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
         articles = Variable(torch.FloatTensor(batch_x))
         labels = Variable(torch.FloatTensor(batch_y))

         # Forward + Backward + Optimize
         optimizer.zero_grad()  # zero the gradient buffer
     outputs = net(articles)
     loss = criterion(outputs, labels)
     loss.backward()
     optimizer.step()

     print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
           % (epoch + 1, num_epochs, i + 1, len(newsgroups_train.data) // batch_size, loss.data[0]))
     break
break

# Test the Model
     correct = 0
     total = 0
     total_test_data = len(newsgroups_test.target)
     batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
     articles = Variable(torch.FloatTensor(batch_x_test))
     labels = Variable(torch.LongTensor(batch_y_test))
     outputs = net(articles)
     _, predicted = torch.max(outputs.data, 1)
     total += labels.size(0)
     correct += (predicted == labels).sum()

 # source of code: Déborah Mesquita: https://www.deborahmesquita.com/2017-11-05/how-pytorch-gives-the-big-picture-with-deep-learning
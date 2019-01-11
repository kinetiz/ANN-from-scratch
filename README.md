# ANN-from-scratch
This project is to code up the neural network object from scratch with python. Only Numpy library is required.
To make it simple, Logistic or Sigmoid function is used as activation function for all nodes.
The ANN object is configurable which allows users to configure hidden layer size, no. of hidden layer, learning rate and limited iteration

# Summary of the code in training part
1) Feedforward - Calculate the initial predicted result by multiplying the input by weights and biases with non-linear transformation by activation fucntion

2) Calculate loss function (average square error) and print out to update the progress from each traning step

3) Backpropagation
* Calculate gradients of all weights and biases by backpropagating error recursively from last layer backward to the first layer. Thus, the backpropagation function is recursive where the loop finishes at the first layer.
* Use the gradients to update the weights and biases

4) Repeat 1) - 3) until convergence(no more improvement from the last training step, loss does not decrease) or limited iteration is reached

# Testing part
Use XOR dataset, where non-linear estimator is required to achieve the prediction, for testing the ANN object

Please feel free to change the configurations eg. hidden size, no. of hidden layer, learning rate, limited iteration

Note: Accuracy might drop from parameter changes where you need to fine-tune other parameters to find a proper configuration

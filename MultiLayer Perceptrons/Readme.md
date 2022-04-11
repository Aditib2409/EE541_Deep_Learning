Implemented a MultiLayer Perceptron using Standard Python Libraries, Numpy and MatPlotLib. Trained the network on MNIST dataset.

1. Dataset
The dataset used in this question is the MNIST classification dataset. In this problem, we build a simple MLP based neural network from scratch and conduct a detail implementation of forward propagation and a backward propagation. the training dataset is split into a validation set of size 10,000.

2. Network Architecture
In this problem, I have build a network with two hidden layers and an ouptut layer. The following is the layout of each layer
  1. 1st hidden layer - 200 neurons, “RELU”/ “TANH” activation
  2. 2nd hidden layer - 100 neurons, “RELU”/“TANH” activation
  3. output layer - 10 neurons, “SOFTMAX” activation

3. Cost Function
After the forward propagation, the cost function is calculated using ‘cross entropy’.

4. Optimizer
For the optimizer the Stochastic Gradient Descent optimization method is used. According to which the weights and biases are updated as follows:
w(l) = w(l) − η(δ(l)a(l−1))b(l) = b(l) − η(δ(l))
A set of three different learning rates were used - [0.1, 0.05, 0.01]. For the training phase , the data images were sent as a batch of size 500. And the training phase was run for a total of 50 epochs, before each of which the training dataset is shuffled. After every batch, the gradients from the back propagation phase are averaged and finally the parameters are computed for each epoch and the optimal parameters are computed x
During each epoch, both training accuracy and validation accuracy is computed and plotted as follows. As seen the validation set is doing slightly better with the model but has accuracy very close to the training set. This can be seen in the following plots shown in the next section
Here I have not used any regularizer as the gradients did not seem to explode.

5. Parameter Initialization
The weights are initialized as normal distribution with mean = 0 and standard deviation as 1/(out_size). For the biases, a normal distribution was used for initialization.
There is no Batch Normalization done in the problem.

In the training stage, there is learning rate decay also included where after the 20 epochs the learning rate is reduced by 2 and after 40 epochs it is reduced by 4 times.
A total of 6 following configurations were used -
1. ReLU layer actiavtion, Learning rate = 0.1 
2. ReLU layer activation, Learning rate = 0.05 
3. ReLU layer activation, Learning rate = 0.01 
4. tanh layer activation, Learning rate = 0.1
5. tanh layer activation, Learning rate = 0.05 
6. tanh layer activation, Learning rate = 0.01

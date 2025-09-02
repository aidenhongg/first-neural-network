I started this project after watching 3Blue1Brown's neural network series: 

[![Watch the video](https://img.youtube.com/vi/aircAruvnKk/mqdefault.jpg)](https://youtu.be/aircAruvnKk)

The example network he uses is an MLP containing 4 layers - 2 hidden layers with 16 neurons each. It predicts the digits shown in a 28 x 28 picture of peoples' handwriting, so the input and output layer each have 784 and 10 neurons. This project is that example brought to life.

For any layer L:

Let $W_{L}$ and $b_{L}$ be the weights matrix and bias vector.

Let $z_{L}$ be the raw weighted sum.

Let $n_{L}$ be the output neurons. 

The raw weighted sum used was:
$z_{L} = W_{L}n_{L-1} + b_{L}$

The activation function (ReLU) used was:
$ReLU(x) = \max(0, x)$

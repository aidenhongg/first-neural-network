## I started this project after watching 3Blue1Brown's neural network series: 

[![Watch the video](https://img.youtube.com/vi/aircAruvnKk/mqdefault.jpg)](https://youtu.be/aircAruvnKk)

The example network he uses is an MLP containing 4 layers - 2 hidden layers with 16 neurons each. It predicts the digits shown in a 28 x 28 images of peoples' handwriting, so the input and output layer each have 784 and 10 neurons. This project is that example brought to life.

To sum up his video series, a neural network is a large function that uses large matrix operations and differentiable activation functions. In training, it uses derivatives to find in what direction its parameters must shift to bring its outputs as close to the ideal outputs as possible. 

## Summary: 
This exact model has 4 layers.

  - **L1** - The input layer, with 784 neurons so it can accept flattened and vectorized 28 x 28 images.
  - **L2 / L3** - Hidden layers with 16 neurons each and ReLU activation. 
  - **L4** - Output layer with 10 neurons, each representing a digit. SoftMax activation.

Parameters are randomly set on instantiation using He initialization. The cost function is Cross Categorical-Entropy (CCE) and gradient steps are calculated with an ADAM optimizer. 

Optimized default hyperparameters can be found here. These were used to train the current saved model, which has a 95.11% raw accuracy and ~0.224 cost.

## How to use:
There are three public modules that users may run.
#### run_interactive.py
The most user-friendly. Loads the saved model and prints random entries from the testing set. Human users can test their prediction accuracy against the machine. 
  - If you delete the default saved model, you must train and save a new one using `run_model.py`.
#### run_model.py
Instantiates and trains the model.
  - `LOAD_SEED`: if `True`, load the seed and fix parameter initialization.
  - `SEED`: set the seed that `LOAD_SEED` uses.
  - `LOAD_MODEL`: if `True`, load the model in `./_IO/output/model`.
  - `SAVE_MODEL`: if `True`, overwrite the model in  `./_IO/output/model` and save the current model after training.
#### run_tuner.py
For testing purposes only. Iteratively creates 1500 different models using 150 unique hyperparameters and 10 different random seeds each. 
  - The lowest-cost model for each set of hyperparameters, along with the correlated seed and raw accuracy are recorded in `./_IO/output/cost_by_hyperparameter.csv`.

**All hyperparameters and flags** can be edited in `hyperparameters_flags.py`

## Process:

Because of my interest in math, I did my best to calculate the derivatives myself and building my network using mostly only NumPy and the data import functions from python-mnist. My math can be found here.

I tried following 3Blue1Brown's example as closely as possible - using ReLU on every layer and Mean Square Error (MSE) as the loss function. However, there were some complications I came across in the process that required me to make some changes:
  - It's unclear how parameters are randomly initialized. **If unbounded, training can be wildly inconsistent and inefficient**. Thus I looked up initialization methods and found **He initialization**, which uses a normal distribution with a standard deviation based on each layer's neuron count.

  - **ReLU has no bounded maximum value**, meaning there's no way to create a constant ideal output vector to compare against the model's predictions. After more searching I used **SoftMax** on the last layer, which was naturally **followed by changing the loss function from MSE to CCE**.
    
  - He's **ambiguous** on **how** and **when** the parameters are updated after calculating the gradients, as well as how we can **avoid getting stuck** in shallow local minima.
      1. I started by shuffling the data between epochs to **introduce stochasity**.
         
      2. I found I could alternatively **use an Exponential Weighted Moving Average (EWMA) to weight gradients based on recency**, instead of a simple raw average. This led me to add one EWMA and rig up what was essentially a RMSProp, but due to being implemented the same way as my raw average, it was **updated on every example and reset after every batch.**
         
      3. My model performance became **incredibly low** (40% accuracy at best), so I went looking for another optimizer and found ADAM. My model still struggled with ADAM.
     
      4. It turned out EWMAs are **NOT an alternative to the raw average** of gradients in a batch; they should **persist even across epochs.** This was the most crucial change that I implemented - average performance shot up to **~85%** raw accuracy after.
   
      5. I was confused on whether EWMA should be updated after each batch or each example. My grad student friend **Rick Chaki** generously informed me that updates happening between examples can be more performant if the batches **have been sorted such that they each contain similar examples.** Because I was shuffling my dataset entirely randomly, I decided to **update it between batches only.**

These challenges demonstrate that **although there are conventional methods** established for neural network activation and gradient stepping, **there are many details** making up these conventions that are **subject to change** depending on the **context** of the dataset and application.

## Tuning:



For any layer L:

Let $W_{L}$ and $b_{L}$ be the weights matrix and bias vector.

Let $z_{L}$ be the raw weighted sum.

Let $n_{L}$ be the output neurons. 

The raw weighted sum:
$z_{L} = W_{L}n_{L-1} + b_{L}$

The hidden layers' activation function (ReLU), given a scalar x:

$ReLU(x) = \max(0, x)$

The final layer's activation function (Softmax), given a vector v with entries $v_i$:

$Softmax(v_i) = \frac{e^{v_i}}{\sum_j e^{v_j}}$

The loss function (Cross Categorical-Entropy - CCE):

$L = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})$


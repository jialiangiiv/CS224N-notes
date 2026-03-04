# Week1-lec2

## Optimziation: Gradient Descent

- $J(\theta)$ is the cost function we want to minimize
- $\nabla_{\theta} J(\theta) = \left[ \frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \dots, \frac{\partial J}{\partial \theta_n} \right] $
  - The negative gradient ($-\nabla$) tells you: in which direction the altitude decreases most rapidly.

Stochastic gradient descent (SGD)

## GloVe

Core idea: word vector learning should be with ratios of co-occurrence probabilities rather than the probabilities themselves.

Formula: 
$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})$
It means that the dot product of two word vectors (plus biases) should equal the log of how often those words co-occur.

Objective function:
$J = \sum_{i,j} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij}) \right)^2$
$f(X_{ij})$ is a weighting function for corner cases.

## Word Vector Evaluation

intrisic evaluation
1. word vector analogies
2. meaning similarity

extrinsic evaluation: downstream tasks
1. named entity recognition

## Word Sense

Word ambiguity: one word can have multiple meanings/senses.
To solve this, we use weighting.

## Named Entity Recognition (NER)

In our example, $y$ is a label with only two classes: location or non-location.

## Note

- A hyperparameter is a setting you choose before training a machine learning model.
  - such as learning rate, batch size, etc.


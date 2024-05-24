* Create your own library, have it respond to at least 10 datasets.
* write method and evaluation
* Random forests with time series.
* Want data
* Mamba neural network. Edge.
* XG Boost with time series
* TFT
* RNN
* Time series models - arima, seasonal moving average.
* Sampling is key.

I outlined a roadmap of most central ML concepts in another answer. Spend about two weeks following this:

(I’ve written about many of these topics in various answers on Quora; I linked the most relevant ones for quick reference.)

Day 1:

Basic terminology:
Most common settings: Supervised setting, Unsupervised setting, Semi-supervised setting, Reinforcement learning.
Most common problems: Classification (binary & multiclass), Regression, Clustering.
Preprocessing of data: Data normalization.
Concepts of hypothesis sets, empirical error, true error, complexity of hypotheses sets, regularization, bias-variance trade-off, loss functions, cross-validation.
Day 2:

Optimization basics:
Terminology & Basic concepts: Convex optimization, Lagrangian, Primal-dual problems, Gradients & subgradients,  
ℓ
1
  and  
ℓ
2
  regularized objective functions.
Algorithms: Batch gradient descent & stochastic gradient descent, Coordinate gradient descent.
Implementation: Write code for stochastic gradient descent for a simple objective function, tune the step size, and get an intuition of the algorithm.
Day 3:

Classification:
Logistic Regression
Support vector machines: Geometric intuition, primal-dual formulations, notion of support vectors, kernel trick, understanding of hyperparameters, grid search.
Online tool for SVM: Play with this online SVM tool (scroll down to “Graphic Interface”) to get some intuition of the algorithm.
Day 4:

Regression:
Ridge regression
Clustering:
k-means & Expectation-Maximization algorithm.
Top-down and bottom-up hierarchical clustering.
Day 5:

Bayesian methods:
Basic terminology: Priors, posteriors, likelihood, maximum likelihood estimation and maximum-a-posteriori inference.
Gaussian Mixture Models
Latent Dirichlet Allocation: The generative model and basic idea of parameter estimation.
Day 6:

Graphical models:
Basic terminology: Bayesian networks, Markov networks / Markov random fields.
Inference algorithms: Variable elimination, Belief propagation.
Simple examples: Hidden Markov Models. Ising model.
Days 7–8:

Neural Networks:
Basic terminology: Neuron, Activation function, Hidden layer.
Convolutional neural networks: Convolutional layer, pooling layer, Backpropagation.
Memory-based neural networks: Recurrent Neural Networks, Long-short term memory.
Tutorials: I’m familiar with this Torch tutorial (you’ll want to look at  
1_supervised
  directory). There might be other tutorials in other deep learning frameworks.
Day 9:

Miscellaneous topics:
Decision trees
Recommender systems
Markov decision processes
Multi-armed bandits
Day 10: (Budget day)

You can use the last day to catch up on anything left from previous days, or learn more about whatever topic you found most interesting / useful for your future work.
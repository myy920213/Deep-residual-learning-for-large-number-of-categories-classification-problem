# Deep-residual-learning-for-large-number-of-categories-classification-problem

Training deep model on large data set with plenty of classes could meet difficulty of probability vibration in softmax layer, the huge gap between batch size and categories makes classes seen in each iteration extremely unbalance. Two strategies are applied to overcome this problem, (1) Take a sub-sample set of all the classes in softmax partition function to reduce the influence on negative classes. (2) Sparsifying the probability distribution with sparsemax loss function

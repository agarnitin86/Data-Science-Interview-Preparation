## Deep Learning

## How does dropout work in deep learning algorithms?
1. [Dropout Regularization - Practical Aspects of Deep Learning | Coursera](https://www.coursera.org/lecture/deep-neural-network/dropout-regularization-eM33A)
1. [Why Dropout is so effective in Deep Neural Network? | Towards Data Science](https://towardsdatascience.com/introduction-to-dropout-to-regularize-deep-neural-network-8e9d6b1d4386)

## What is the vanishing & exploding gradient problem? How to identify it? How is it solved?
1. [The Vanishing/Exploding Gradient Problem in Deep Neural Networks | by Kurtis Pykes | Towards Data Science](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11)
1. [The Vanishing Gradient Problem. The Problem, Its Causes, Its… | by Chi-Feng Wang | Towards Data Science](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

In a network of n hidden layers, n derivatives will be multiplied together. If the derivatives are large then the gradient will increase exponentially as we propagate down the model until they eventually explode, and this is what we call the problem of exploding gradient. Alternatively, if the derivatives are small then the gradient will decrease exponentially as we propagate through the model until it eventually vanishes, and this is the vanishing gradient problem.

**Identifying Exploding gradient:**

- The model is not learning much on the training data therefore resulting in a poor loss.
- The model will have large changes in loss on each update due to the models instability.
- The models loss will be NaN during training.
- Model weights grow exponentially and become very large when training the model.
- The model weights become NaN in the training phase.

**Identifying Vanishing gradient:**

- The model will improve very slowly during the training phase and it is also possible that training stops very early, meaning that any further training does not improve the model.
- The weights closer to the output layer of the model would witness more of a change whereas the layers that occur closer to the input layer would not change much (if at all).
- Model weights shrink exponentially and become very small when training the model.
- The model weights become 0 in the training phase.

**Solving gradient problem:**
1. Reducing the amount of Layers
1. Choosing a small learning rate so that there are no large updates in the gradients
1. Gradient Clipping (Exploding Gradients)
1. Weight Initialization
1. use other activation functions, such as ReLU, which doesn’t cause a small derivative.
1. Residual networks
1. Add batch normalization layers
1. Use LSTM Networks

## What is Gradient Clipping?
1. [Introduction to Gradient Clipping Techniques with Tensorflow | cnvrg.io](https://cnvrg.io/gradient-clipping/#:~:text=%20Gradient%20clipping%20can%20be%20applied%20in%20two,by%20value%202%20Clipping%20by%20norm%20More%20)
1. [Understanding Gradient Clipping (and How It Can Fix Exploding Gradients Problem) - neptune.ai](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)

Gradient clipping involves forcing the gradients to a certain number when they go above or below a defined threshold. 

**Types of Clipping techniques**
Gradient clipping can be applied in two common ways:

**Clipping by value** : If a gradient exceeds some threshold value, we clip that gradient to the threshold. If the gradient is less than the lower limit then we clip that too, to the lower limit of the threshold. The algorithm is as follows:

    g ← ∂C/∂W
    if ‖g‖ ≥ max_threshold or ‖g‖ ≤ min_threshold then
        g ← threshold (accordingly)
    end if
where max\_threshold and min\_threshold are the boundary values and between them lies a range of values that gradients can take. g, here is the gradient, and  ‖g ‖ is the norm of g. 

**Clipping by norm**: We clip the gradients by multiplying the unit vector of the gradients with the threshold. 
The algorithm is as follows:

    g ← ∂C/∂W
    if ‖g‖ ≥ threshold then
        g ← threshold * g/‖g‖
    end if
where the threshold is a hyperparameter, g is the gradient, and  ‖g ‖ is the norm of g. Since g/‖ g‖ is a unit vector, after rescaling the new g will have norm equal to threshold. Note that if  ‖g‖ < c, then we don’t need to do anything,

Gradient clipping ensures the gradient vector g has norm at most equal to threshold. 

## Compare Activation Functions
1. [Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax | by Mukesh Chaudhary | Medium](https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5)

## Compare Loss Functions
1. [Loss Function | Loss Function In Machine Learning (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/)![](Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.030.png)

## Compare Optimizers
[Line to coursera](https://www.coursera.org/learn/deep-neural-network/lecture/qcogH/mini-batch-gradient-descent)

1. Gradient Descent
1. Stochastic Gradient Descent - Same as Gradient descent, but for each sample
1. Mini Batch Gradient Descent - Same as Gradient descent, but for each mini batch
1. Gradient Descent with Momentum - Each update is not made with the current value of gradient, but with the exponential moving average of the gradients:                                   

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.031.png)

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.032.png)

5. Root Mean Square Propagation (RMSProp):

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.033.png)

6. Adaptive Moment Estimate (Adam)

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.034.png)

## How to reduce overfitting

1. Adding more data 
2. Data Augmentation 
3. Dropout
4. Early Stopping

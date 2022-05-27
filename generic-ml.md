# Data-Science-Interview-Preparation
ISLR = Introduction to Statistical Learning  
ESLR = Elements of Statistical Learning

[clickme](https://github.com/agarnitin86/Data-Science-Interview-Preparation/blob/main/classification.md)

## Generative vs Discriminative Models
## Difference b/w generative & discriminative models
1. [Generative vs. Discriminative Machine Learning Models - Unite.AI](https://www.unite.ai/generative-vs-discriminative-machine-learning-models/)
1. [machine learning - What is the difference between a generative and a discriminative algorithm? - Stack Overflow](https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-a-discriminative-algorithm)


|**Generative**|**Discriminative**|
| :-- | :-- |
|Generative models aim to capture the actual distribution of the classes in the dataset.|Discriminative models model the decision boundary for the dataset classes.|
|Generative models predict the joint probability distribution – p(x,y) – utilizing [Bayes Theorem](https://www.unite.ai/what-is-bayes-theorem/).|<p>Discriminative models learn the conditional probability – p(y|x).</p><p></p>|
|Generative models are computationally expensive compared to discriminative models.|<p>Discriminative models are computationally cheap compared to generative models.</p><p></p>|
|Generative models are useful for unsupervised machine learning tasks.|Discriminative models are useful for supervised machine learning tasks.|
|<p>Generative models are impacted by the presence of outliers more than discriminative models.</p><p></p>|<p>Discriminative models have the advantage of being more robust to outliers, unlike the generative models.</p><p>Discriminative models are more robust to outliers compared to generative models.</p>|
|E.g. Linear Discriminant Analysis, HMM, Bayesian Networks|E.g. SVM, Logistic regression, Decision Trees, Random Forests|
</details>

# Parametric & Non-Parametric models
## Difference b/w parametric & non-parametric models
***[Source: ISLR Page-21]*:** 


Parametric methods involve a two-step model-based approach.

1. First, we make an assumption about the functional form, or shape of f. For example, one very simple assumption is that f is linear in X:

<img src="https://latex.codecogs.com/svg.image?f(x)=\beta_0&plus;\beta_1x_1&plus;\beta_2x_2&plus;\cdots&plus;\beta_px_p\cdots\cdots(2.4)">

This is a linear model, which will be discussed extensively in Chapter 3. Once we have assumed that f is linear, the problem of estimating f is greatly simplified. Instead of having to estimate an entirely arbitrary p-dimensional function f(X), one only needs to estimate the p+1 coefficients <img src="https://latex.codecogs.com/svg.image?\beta_0,&space;\beta_1x_1,&space;\beta_2x_2,&space;\cdots,&space;\beta_px_p"> 

2. After a model has been selected, we need a procedure that uses the training data to fit or train the model. In the case of the linear model (2.4), we need to estimate the parameters <img src="https://latex.codecogs.com/svg.image?\beta_0,&space;\beta_1x_1,&space;\beta_2x_2,&space;\cdots,&space;\beta_px_p">. That is, we want to find values of these parameters such that,
<img src="https://latex.codecogs.com/svg.image?Y\approx\beta_0&plus;\beta_1x_1&plus;\beta_2x_2&plus;\cdots&plus;\beta_px_p">

Non-parametric methods do not make explicit assumptions about the functional form of *f*. Instead, they seek an estimate of *f* that gets as close to the data points as possible without being too rough or wiggly. 

Some examples of parametric and non-parametric models:

|**Parametric Models**|**Non-parametric Models**|
| :-: | :-: |
|Linear regression|KNN|
|Logistic regression|Decision Trees, Random Forests|
## Advantages/disadvantages of Parametric/Non-Parametric models
***[Source: ISLR Page-23]*:** 

Non-parametric approaches can have a major advantage over parametric approaches: by avoiding the assumption of a particular functional form for *f*, they have the potential to accurately fit a wider range of possible shapes for f. Any parametric approach brings with it the possibility that the functional form used to estimate f is very different from the true f, in which case the resulting model will not fit the data well. 

In contrast, non-parametric approaches completely avoid this danger, since essentially no assumption about the form of f is made. But non-parametric approaches do suffer from a major disadvantage: since they do not reduce the problem of estimating f to a small number of parameters, a very large number of observations (far more than is typically needed for a parametric approach) is required in order to obtain an accurate estimate for f.
# Bias & Variance
## What do we mean by the variance and bias of a statistical learning method? 
**Variance** refers to the amount by which *f* would change if we estimated it using a different training data set. Since the training data are used to fit the statistical learning method, different training data sets will result in a different *f* . But ideally the estimate for *f* should not vary too much between training sets. However, if a method has high variance, then small changes in the training data can result in large changes in *f*. In general, more flexible statistical methods have higher variance.

On the other hand, 

**Bias** refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. For example, linear regression assumes that there is a linear relationship between *Y* and <img src="https://latex.codecogs.com/svg.image?X_1,X_2,\cdots,X_p">. It is unlikely that any real-life problem truly has such a simple linear relationship, and so performing linear regression will undoubtedly result in some bias in the estimate of *f*.

## How to handle data imbalance:
1. Under sampling/Oversampling
1. SMOTE
1. Better evaluation metric – like Lift, ROC curves, PR Curves
1. Cost sensitive learning : [Cost-Sensitive Learning for Imbalanced Classification (machinelearningmastery.com)](https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/)
1. Class weight balancing : [How To Dealing With Imbalanced Classes in Machine Learning (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/)
1. Weighted loss function: [Handling Class Imbalance by Introducing Sample Weighting in the Loss Function | by Ishan Shrivastava | GumGum Tech Blog | Medium](https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4)
1. One Class SVM

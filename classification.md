#
# Classification
## Some approaches for classification
1. Using linear regression of a Indicator Matrix
1. Linear Discriminant Analysis
1. Quadratic Discriminant Analysis
1. Regularized Discriminant Analysis ***[Source: ESLR Page-112]*** ![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.007.png) 
1. Logistic Regression

## Resources to understand MLE estimation for Logistic Regression
1. [lecture05.pdf (zstevenwu.com)](https://zstevenwu.com/courses/s20/csci5525/resources/slides/lecture05.pdf)
1. [Logit.dvi (rutgers.edu)](https://stat.rutgers.edu/home/pingli/papers/Logit.pdf)
1. [ADAfaEPoV (cmu.edu)](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf)
1. [A Gentle Introduction to Logistic Regression With Maximum Likelihood Estimation (machinelearningmastery.com)](https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation/)
1. [Logistic Regression and Maximum Likelihood Estimation Function | by Puja P. Pathak | CodeX | Medium](https://medium.com/codex/logistic-regression-and-maximum-likelihood-estimation-function-5d8d998245f9)

## Difference b/w Logistic Regression & Linear Discriminant Analysis
***[Source: ISLR Page-151]***


|**Logistic Regression**|**Linear DA**|
| :-- | :-- |
|Parameters <img src="https://latex.codecogs.com/svg.image?\beta_0,\beta_1"> are estimated using Maximum Likelihood estimation|Parameters are estimated using estimated mean & variance from normal distribution|
|Decision boundary- Linear|Decision boundary- Linear|
|logistic regression can outperform LDA if the Gaussian assumptions are not met|LDA assumes that the observations are drawn from a Gaussian distribution with a common covariance matrix in each class, and so can provide some improvements over logistic regression when this assumption approximately holds.|

## Difference b/w Linear & Quadratic Discriminant Analysis
***[Source: ESLR Page-109]*** and [lecture9-stanford](https://web.stanford.edu/class/stats202/content/lec9.pdf)


|**Linear DA**|**Quadratic DA**|
| :-: | :-: |
|All the classes have common covariance matrix Σk = Σ ∀ *k*|Each class has its own covariance matrix, Σk|
|Decision boundary- Linear|Decision boundary- Quadratic|
|Discriminant Function![](Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.008.png)|<p>Discriminant Function</p><p>![](Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.009.png)</p>|
|Since covariance matrices is common for all classes no such problem|Since separate covariance matrices must be computed for each class, when p (#Features) is large, number of parameters increases dramatically.|
|[Source: ISLR Page 142] LDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vector and a common variance σ2|[Source: ISLR Page 142] LDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vector and covariance matrix Σk|
|With p predictors, estimating a covariance matrix requires estimating p(p+1)/2 parameters. |With p predictors and K classses, estimating a covariance matrix requires estimating K.p(p+1)/2 parameters|
|LDA is a much less flexible classifier|QDA is a more flexible classifier|
|Can have low variance high bias||

#
# Logistic Regression

## Write the Logistic Function


## Log odds

![](Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.010.png)

## How to fit a logistic regression model?
A.2. Although we could use (non-linear) least squares to fit the logistic model , the more general method of maximum likelihood is preferred, since it has better statistical properties.

## Classification Evaluation Metrics
1. Sensitivity = Recall = True Positive Rate
   1. ` `TPTP+FN
1. Specificity = True Negative Rate
   1. TNTN+FP
1. Precision
   1. TPTP+FP
1. False Positive Rate
   1. FPFP+TN

A.2. Although we could use (non-linear) least squares to fit the logistic model , the more general 
## What happens when the classes are well separated in Logistic Regression?
A.3. When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.

<https://stats.stackexchange.com/questions/224863/understanding-complete-separation-for-logistic-regression>

<https://stats.stackexchange.com/questions/239928/is-there-any-intuitive-explanation-of-why-logistic-regression-will-not-work-for>

[Source ESLR, Page 128] If the data in a two-class logistic regression model can be perfectly separated by a hyperplane, the maximum likelihood estimates of the parameters are undefined (i.e., infinite; see Exercise 4.5). The LDA coefficients for the same data will be well defined, since the marginal likelihood will not permit these degeneracies.
## Compare SVM & Logistic Regression
[Source ISLR, Page 357]SVM loss function is exactly zero for observations for which yi(β0 + β1xi1 + ... + βpxip) ≥ 1; these correspond to observations that are on the correct side of the margin. In contrast, the loss function for logistic regression is not exactly zero anywhere. But it is very small for observations that are far from the decision boundary. Due to the similarities between their loss functions,  logistic regression and the support vector classifier often give very similar results. When the classes are well separated, SVMs tend to behave better than logistic regression; in more overlapping regimes, logistic regression is often preferred.


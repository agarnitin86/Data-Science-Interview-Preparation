#
# Classification
## Some approaches for classification
1. Using linear regression of a Indicator Matrix
1. Linear Discriminant Analysis
1. Quadratic Discriminant Analysis
1. Regularized Discriminant Analysis ***[Source: ESLR Page-112]*** ![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.007.png) 
1. Logistic Regression

## What is log odds?
1. [Log Oddds Definition](https://www.statisticshowto.com/log-odds/#:~:text=Taking%20the%20logarithm%20of%20the,p%2F(1%2Dp)%5D)
2. [What and Why of Log Odds](https://towardsdatascience.com/https-towardsdatascience-com-what-and-why-of-log-odds-64ba988bf704)

The odds ratio is the probability of success/probability of failure. As an equation, that’s P(A)/P(-A), where P(A) is the probability of A, and P(-A) the probability of ‘not A’ (i.e. the complement of A).

Taking the logarithm of the odds ratio gives us the log odds of A, which can be written as

log(A) = log(P(A)/P(-A)),
Since the probability of an event happening, P(-A) is equal to the probability of an event not happening, 1 – P(A), we can write the log odds as

<img src="https://latex.codecogs.com/svg.image?log\frac{p}{(1-p)}">

Where:
p = the probability of an event happening
1 – p = the probability of an event not happening

## MLE Estimation for Logistic Regression
Although we could use (non-linear) least squares to fit the logistic model , the more general method of maximum likelihood is preferred, since it has better statistical properties. In logistic regression, we use the logistic function,
<img src="https://latex.codecogs.com/svg.image?p(X)=\frac{\exp^{\beta_0&plus;\beta_1X}}{1&plus;{\exp^{\beta_0&plus;\beta_1X}}}">
<img src="https://latex.codecogs.com/svg.image?\begin{aligned}p(X)&=\frac{\exp^{\beta_0&plus;\beta_1X}}{1&plus;{\exp^{\beta_0&plus;\beta_1X}}}\cdots\cdots(1)&space;\\1-p(X)&=1-\frac{\exp^{\beta_0&plus;\beta_1X}}{1&plus;{\exp^{\beta_0&plus;\beta_1X}}}&space;\\&=\frac{1&plus;\exp^{\beta_0&plus;\beta_1X}-\exp^{\beta_0&plus;\beta_1X}}{1&plus;{\exp^{\beta_0&plus;\beta_1X}}}&space;\\&=\frac{1}{1&plus;{\exp^{\beta_0&plus;\beta_1X}}}\cdots\cdots(2)&space;\end{aligned}">

Dividing (1) by (2) & taking log on both sides,

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}\frac{p(X)}{1-p(X)}&=\exp^{\beta_0&plus;\beta_1X}\\log\frac{p(X)}{1-p(X)}&=\beta_0&plus;\beta_1X\end{aligned}">

Let us make following parametric assumption:

<img src="https://latex.codecogs.com/svg.image?y_i|x_i&space;=&space;Bern(\sigma(w^Tx_i))\\where,&space;\\\sigma(z)&space;=&space;\frac{1}{1&plus;\exp{(-z)}}=\frac{\exp{(z)}}{1&plus;\exp{(z)}}">

MLE is used to find the model parameters while maximizing, 

***P(observed data|model parameters)***

For Logistic Regression, we need to find the model parameter **w** that maximizes conditional probability,

<img src="https://latex.codecogs.com/svg.image?=P(y_1,x_1,\cdots\cdots,y_n,x_n&space;|&space;\textbf{w})\\=\operatorname*{argmax}_w&space;&space;P(y_1,x_1,\cdots\cdots,y_n,x_n&space;|&space;\textbf{w})\\=\operatorname*{argmax}_w&space;&space;\prod_i^n&space;P(y_i,x_i&space;|&space;\textbf{w})&space;\cdots\cdots&space;(Independence)\\=\operatorname*{argmax}_w&space;&space;\prod_i^n&space;P(y_i&space;|&space;x_i,&space;\textbf{w})P(x_i&space;|&space;\textbf{w})\\=\operatorname*{argmax}_w&space;&space;\prod_i^n&space;P(y_i&space;|&space;x_i,&space;\textbf{w})P(x_i)\cdots\cdots(x_i\,is\,independent\,of\,\textbf{w})\\=\operatorname*{argmax}_w&space;&space;\prod_i^n&space;P(y_i&space;|&space;x_i,&space;\textbf{w})\cdots\cdots(P(x_i)\,does\,not&space;\,depend\,on\,\textbf{w})\\=\operatorname*{argmax}_w&space;&space;\prod_i^n&space;\sigma{(w^Tx_i)}^{y_i}(1-{\sigma{(w^Tx_i)}})^{1-y_i}\\\\Equivalently,we\,would\,like\,to\,find\,the\,\textbf{w}\,to\,maximize\,the\,log\,likelihood:\\\\=\ln&space;\prod_i^n{\sigma{(w^Tx_i)}^{y_i}(1-{\sigma{(w^Tx_i)}})^{1-y_i}}\\=\sum_{i}^{n}&space;\ln({\sigma{(w^Tx_i)}^{y_i}(1-{\sigma{(w^Tx_i)}})^{1-y_i}})\\=\sum_{i}^{n}&space;\ln{\sigma{(w^Tx_i)}^{y_i}&plus;\ln(1-{\sigma{(w^Tx_i)}})^{1-y_i}}\\&space;=\sum_{i}^{n}&space;({y_i}\ln{\sigma{(w^Tx_i)}&plus;({1-y_i})\ln(1-{\sigma{(w^Tx_i)}})})\\&space;&space;">

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
| :-- | :-- |
|All the classes have common covariance matrix Σ<sub>k</sub> = Σ ∀ *k*|Each class has its own covariance matrix, Σ<sub>k</sub>|
|Decision boundary- Linear|Decision boundary- Quadratic|
|Discriminant Function![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.008.png)|<p>Discriminant Function</p><p>![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.009.png)</p>|
|Since covariance matrices is common for all classes no such problem|Since separate covariance matrices must be computed for each class, when p (#Features) is large, number of parameters increases dramatically.|
|***[Source: ISLR Page-142]*** LDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vector and a common variance σ<sup>2</sup>|***[Source: ISLR Page-142]*** QDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vector and covariance matrix Σ<sub>k</sub>|
|With p predictors, estimating a covariance matrix requires estimating p(p+1)/2 parameters. |With p predictors and K classses, estimating a covariance matrix requires estimating K.p(p+1)/2 parameters|
|LDA is a much less flexible classifier|QDA is a more flexible classifier|
|Can have low variance high bias|Can have high variance low bias|

## What happens when the classes are well separated in Logistic Regression?
A.3. When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.

<https://stats.stackexchange.com/questions/224863/understanding-complete-separation-for-logistic-regression>

<https://stats.stackexchange.com/questions/239928/is-there-any-intuitive-explanation-of-why-logistic-regression-will-not-work-for>

[Source ESLR, Page 128] If the data in a two-class logistic regression model can be perfectly separated by a hyperplane, the maximum likelihood estimates of the parameters are undefined (i.e., infinite; see Exercise 4.5). The LDA coefficients for the same data will be well defined, since the marginal likelihood will not permit these degeneracies.

## Compare SVM & Logistic Regression
[Source ISLR, Page 357]SVM loss function is exactly zero for observations for which yi(β0 + β1xi1 + ... + βpxip) ≥ 1; these correspond to observations that are on the correct side of the margin. In contrast, the loss function for logistic regression is not exactly zero anywhere. But it is very small for observations that are far from the decision boundary. Due to the similarities between their loss functions,  logistic regression and the support vector classifier often give very similar results. When the classes are well separated, SVMs tend to behave better than logistic regression; in more overlapping regimes, logistic regression is often preferred.


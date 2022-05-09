# Data-Science-Interview-Preparation
ISLR = Introduction to Statistical Learning  
ESLR = Elements of Statistical Learning

## Generative vs Discriminative Models
## Difference b/w generative & discriminative models
1. [Generative vs. Discriminative Machine Learning Models - Unite.AI](https://www.unite.ai/generative-vs-discriminative-machine-learning-models/)
1. [machine learning - What is the difference between a generative and a discriminative algorithm? - Stack Overflow](https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-a-discriminative-algorithm)


|**Generative**|**Discriminative**|
| :-: | :-: |
|Generative models aim to capture the actual distribution of the classes in the dataset.|Discriminative models model the decision boundary for the dataset classes.|
|Generative models predict the joint probability distribution – p(x,y) – utilizing [Bayes Theorem](https://www.unite.ai/what-is-bayes-theorem/).|<p>Discriminative models learn the conditional probability – p(y|x).</p><p></p>|
|Generative models are computationally expensive compared to discriminative models.|<p>Discriminative models are computationally cheap compared to generative models.</p><p></p>|
|Generative models are useful for unsupervised machine learning tasks.|Discriminative models are useful for supervised machine learning tasks.|
|<p>Generative models are impacted by the presence of outliers more than discriminative models.</p><p></p>|<p>Discriminative models have the advantage of being more robust to outliers, unlike the generative models.</p><p>Discriminative models are more robust to outliers compared to generative models.</p>|
|E.g. Linear Discriminant Analysis, HMM, Bayesian Networks|E.g. SVM, Logistic regression, Decision Trees, Random Forests|

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

#
# Linear Regression
## Difference b/w TSS (Total Sum of Squares) & RSS (Residual Sum of Square)
<img src="https://latex.codecogs.com/svg.image?RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2\cdots\cdots(3.16)">

<img src="https://latex.codecogs.com/svg.image?TSS=\sum{(y_i-\bar{y})^2}">

**TSS** measures the total variance in the response **Y** and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, **RSS** measures the amount of variability that is left unexplained after performing the regression. Hence, TSS-RSS measures the amount of variability in the response that is explained (or removed) by performing the regression 
## Difference b/w R2 and Adjusted R2
***[Source: ISLR Page-212]*:**


|**R2**|**Adjusted R2**|
| :-- | :-- |
|<img src="https://latex.codecogs.com/svg.image?R^2=\frac{TSS-RSS}{TSS}=\frac{1-RSS}{TSS}">|<img src="https://latex.codecogs.com/svg.image?{Adjusted&space;R^2}=1-\frac{{RSS}/{(n-d-1)}}{{TSS}/{(n-1)}}">, where n = number of data points, d = model with d variables|
|<img src="https://latex.codecogs.com/svg.image?R^2"> will increase as more variables are added to the model.|<p>Maximizing the <img src="https://latex.codecogs.com/svg.image?Adjusted&space;R^2"> is equivalent to minimizing <img src="https://latex.codecogs.com/svg.image?{RSS}/{(n-d-1)}">. While RSS always decreases as the number of variables in the model increases, RSS/(n-d-1) may increase or decrease, due to the presence of d in the denominator.</p><p>The intuition behind the <img src="https://latex.codecogs.com/svg.image?Adjusted&space;R^2"> is that, once all of the correct variables have been included in the model, adding additional noise variables will lead to only a very small decrease in RSS. Since adding noise variables leads to an increase in d, such variables will lead to an increase in RSS/(n-d-1), and consequently a decrease in the <img src="https://latex.codecogs.com/svg.image?Adjusted&space;R^2">. Therefore, in theory, the model with the largest <img src="https://latex.codecogs.com/svg.image?Adjusted&space;R^2"> will have only correct variables and no noise variables.</p>|



## LR-Assumptions & Verification
1. [7 Classical Assumptions of Ordinary Least Squares (OLS) Linear Regression - Statistics By Jim](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/)
1. [Verifying the Assumptions of Linear Regression in Python and R | by Eryk Lewinson | Towards Data Science](https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0#:~:text=Verifying%20the%20Assumptions%20of%20Linear%20Regression%20in%20Python,the%20context%20of%20linear%20regression.%20More%20items...%20)

Following are the assumptions:
1. Random error <img src="https://latex.codecogs.com/svg.image?\varepsilon"> has <img src="https://latex.codecogs.com/svg.image?E(\varepsilon)=0">
1. <img src="https://latex.codecogs.com/svg.image?\varepsilon"> is independent of X
1. **[Source: ISLR Page-86]**: Two of the most important assumptions state that the relationship between the predictors and response are additive and linear. The additive assumption means that the effect of changes in a predictor <img src="https://latex.codecogs.com/svg.image?X_j"> on the response Y is independent of the values of the other predictors. The linear assumption states that the change in the response Y due to a one-unit change in <img src="https://latex.codecogs.com/svg.image?X_j"> is constant, regardless of the value of <img src="https://latex.codecogs.com/svg.image?X_j">
1. **[Source: ISLR Page-93]**: An important assumption of the linear regression model is that the error terms, ε1,ε2, …,εn, are uncorrelated.
1. **[Source: ISLR Page-95]**: Error terms have a constant variance, <img src="https://latex.codecogs.com/svg.image?Var(\varepsilon_i)=\sigma^2">. Unfortunately, it is often the case that the variances of the error terms are non-constant. For instance, the variances of the error terms may increase with the value of the response. One can identify non-constant variances in the errors, or heteroscedasticity, from the presence of a funnel shape in the residual plot.
## Implication of the assumption that errors are independent & identically distributed
Because of this assumption, we average squared errors uniformly in our Expected Prediction error criterion. If the errors were dependent, then, weightage of each error might have been different in the error function.
## Difference b/w Collineraity & Multi Collinearity 
1. [terminology - What is collinearity and how does it differ from multicollinearity? - Cross Validated (stackexchange.com)](https://stats.stackexchange.com/questions/254871/what-is-collinearity-and-how-does-it-differ-from-multicollinearity) 
1. [Correlation vs Collinearity vs Multicollinearity – Quantifying Health](https://quantifyinghealth.com/correlation-collinearity-multicollinearity/)

**Collinearity** is a linear association between two explanatory variables. Multicollinearity in a multiple regression model are highly linearly related associations between two or more explanatory variables. 

In case of perfect **multicollinearity**, the design matrix X has less than full rank, and therefore the moment matrix <img src="https://latex.codecogs.com/svg.image?X^TX"> cannot be matrix inverted. Under these circumstances, for a general linear model <img src="https://latex.codecogs.com/svg.image?y=x\beta+\epsilon">, the ordinary least-squares estimator <img src="https://latex.codecogs.com/svg.image?\beta_{OLS}={(X^TX)}^{-1}X^Ty"> does not exist.
## How to assess Multicollinearity?
***[Source: ISLR Page-101]*** Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the variance inflation factor (VIF). The VIF is variance inflation factor, the ratio of the variance of βj when fitting the full model divided by the variance of βj if fit on its own. The smallest possible value for VIF is 1, which indicates the complete absence of collinearity
## How to assess the quality of Linear Regression model?
***[Source: ISLR Page-68]***. The quality of a linear regression fit is typically assessed using two related quantities: the residual standard error (RSE) and the <img src="https://latex.codecogs.com/svg.image?R^2"> statistic.


![A picture containing text, clock, watch Description automatically generated](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.001.png)

where,

![A picture containing chart Description automatically generated](images/ds-interview/RSS.png)

The RSE provides an absolute measure of lack of fit of the model (3.5) to the data. But since it is measured in the units of Y, it is not always clear what constitutes a good RSE. The <img src="https://latex.codecogs.com/svg.image?R^2"> statistic provides an alternative measure of fit. It takes the form of a proportion—the proportion of variance explained—and so it always takes on a value between 0 and 1, and is independent of the scale of Y.

To calculate <img src="https://latex.codecogs.com/svg.image?R^2">, we use the formula, 

![Text Description automatically generated](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.002.png)

Where, 

<img src="https://latex.codecogs.com/svg.image?TSS=\sum{(y_i-\bar{y})^2}"> is the total sum of squares, and RSS is defined in (3.16). TSS measures the total variance in the response Y, and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, RSS measures the amount of variability that is left unexplained after performing the regression. Hence, TSS − RSS measures the amount of variability in the response that is explained (or removed) by performing the regression, and <img src="https://latex.codecogs.com/svg.image?R^2"> measures the proportion of variability in Y that can be explained using X. 
## Regression approaches in order of linearity
***[Source: ISLR Page-266]***


|**Regression Approach**|**Explanation**|
| :-: | :-- |
|Linear Regression||
|Polynomial Regression|Polynomial regression extends the linear model by adding extra predictors, obtained by raising each of the original predictors to a power.|
|Step Functions|Step functions cut the range of a variable into K distinct regions in order to produce a qualitative variable. This has the effect of fitting a piecewise constant function.|
|Regression Splines|Regression splines are more flexible than polynomials and step functions, and in fact are an extension of the two. They involve dividing the range of X into K distinct regions. Within each region, a polynomial function is fit to the data.|
|Smoothing Splines|Smoothing splines are similar to regression splines, but arise in a slightly different situation. Smoothing splines result from minimizing a residual sum of squares criterion subject to a smoothness penalty.|
|Local Regression|Local regression is similar to splines, but differs in an important way. The regions are allowed to overlap, and indeed they do so in a very smooth way.|
|Generalized Additive Models|Generalized additive models allow us to extend the methods above to deal with multiple predictors.|

## Approaches for Subset selection
***[Source: ESLR Page-57]***

1. Best subset selection
1. Forward stepwise selection
1. Backward stepwise selection
1. Hybrid (Forward + Backward stepwise) selection
1. Forward stagewise regression


|Best subset|<p>1. For each k ∈ {0, 1, 2, . . . , p}, where p is #predictors, the subset of size k that gives smallest residual sum of squares.</p><p>2. Typically, we choose the smallest model that minimizes an estimate of the expected prediction error.</p>|
| :- | :-- |
|Forward stepwise|<p>1. Is a greedy algorithm </p><p>2. Starts with the intercept, and,</p><p>3. then sequentially adds into the model the predictor that most improves the fit. With many candidate predictors, this might seem like a lot of computation; however, clever updating algorithms can exploit the QR decomposition for the current fit to rapidly establish the next candidate </p><p>4. for large p, we cannot compute the best subset sequence, but we can always compute the forward stepwise sequence(even when p ≫ N ).</p><p>5. forward stepwise is a more constrained search, and will have lower variance, but perhaps more bias than best subset</p>|
|Backward stepwise|<p>1. starts with the full model, then,</p><p>2. Sequentially deletes the predictor that has the least impact on the fit. </p><p>3. The candidate for dropping is the variable with the smallest Z-score. Backward selection can only be used when N > p, while forward stepwise can always be used.</p>|
|Hybrid selection|<p>consider both forward and backward moves at each step, and select</p><p>the “best” of the two.</p>|
|Forward stagewise|<p>1. more constrained than forward stepwise regression.</p><p>2. Starts like forward-stepwise regression, with an intercept equal to ȳ, and centered predictors with coefficients initially all 0.</p><p>3. At each step, the algorithm identifies the variable most correlated with the current residual. </p><p>4. It then computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current coefficient for that variable. This is continued till none of the variables have correlation with the residuals—i.e. the least-squares fit when N > p.</p>|
## What is confidence interval and prediction interval in Linear Regression?
## Derive equations for Least squares in vector & matrix notation
## Can we use Linear Regression for binary classification?

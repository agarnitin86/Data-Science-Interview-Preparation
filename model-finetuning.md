## Cross Validation

## Compare LOOCV(Leave One Out Cross Validation) with K-Fold CV

***[Source: ISLR Page-180]*** 

|LOOCV|K-Fold CV|
| :-- | :-- |
|LOOCV has the potential to be expensive to implement, since the model has to be fit n times. This can be very time consuming if n is large, and if each individual model is slow to fit.|Computationally less expensive|
|each training set contains n − 1 observations - has lower bias|each training set contains (k − 1)n/k observations—fewer than in the LOOCV approach - has higher bias|
|LOOCV has higher variance than does k-fold CV with k<n. When we perform LOOCV, we are in effect averaging the outputs of n fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other|<p>When we perform k-fold CV with k<n, we are averaging the outputs of k fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller.</p><p>Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from k-fold CV.</p>|

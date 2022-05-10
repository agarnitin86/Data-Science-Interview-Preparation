#
# KNN
## Effect of K on training error
***[Source: ESLR Page 15]*** Error on the training data should be approximately an increasing function of k, and will always be 0 for k = 1. 
It appears that k-nearest-neighbor fits have a single parameter, the number of neighbors k, compared to the p parameters in least-squares fits. Although this is the case, we will see that the effective number of parameters of k-nearest neighbors is N/k and is generally bigger than p, and decreases with increasing k. To get an idea of why, note that if the neighborhoods were nonoverlapping, there would be N/k neighborhoods and we would fit one parameter (a mean) in each neighborhood. It is also clear that we cannot use sum-of-squared errors on the training

set as a criterion for picking k, since we would always pick k = 1.
## How does bias & variance vary for KNN with the choice of K?
***[Source: ESLR Page 40]*** The choice of K has a drastic effect on the KNN classifier obtained. When K = 1, the decision boundary is overly flexible and finds patterns in the data that don’t correspond to the Bayes decision boundary. This corresponds to a classifier that has low bias but very high variance. As K grows, the method becomes less flexible and produces a decision boundary that is close to linear. This corresponds to a low-variance but high-bias classifier. 
Just as in the regression setting, there is not a strong relationship between the training error rate and the test error rate. With K = 1, the KNN training error rate is 0, but the test error rate may be quite high. In general, as we use more flexible classification methods, the training error rate will decline but the test error rate may not. 

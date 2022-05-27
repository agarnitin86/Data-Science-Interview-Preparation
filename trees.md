## Tree Based Models

## Gini Index

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.025.png)

Here <img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}"> represents the proportion of training observations in the m<sup>th</sup> region that are from the kth class. a measure of total variance across the K classes. It is not hard to see that the Gini index takes on a small value if all of the  <img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}">’s are close to zero or one. For this reason the Gini index is referred to as a measure of node purity—a small value indicates that a node contains predominantly observations from a single class. 

## Cross-Entropy

Since 0 ≤ <img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}"> ≤ 1, it follows that 0 ≤ −<img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}"> log <img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}">. One can show that the cross- entropy will take on a value near zero if the <img src="https://latex.codecogs.com/svg.image?\hat{p}_{mk}">’s are all near zero or near one. Therefore, like the Gini index, the cross-entropy will take on a small value if the mth node is pure. In fact, it turns out that the Gini index and the cross-entropy are quite similar numerically.  
![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.026.png)

## Why Bagging reduces over-fitting or variance?

***[Source: ISLR Page-316]*** Given a set of n independent observations Z<sub>1</sub> , . . . , Z<sub>n</sub>, each with variance σ<sup>2</sup>, the variance of the mean <img src="https://latex.codecogs.com/svg.image?\hat{Z}"> of the observations is given by σ<sup>2</sup>/n. In other words, averaging a set of observations reduces variance.

Explanation of why the above happens: [https://en.wikipedia.org/wiki/Variance#Properties](https://en.wikipedia.org/wiki/Variance#Properties)

Hence a natural way to reduce the variance and hence increase the prediction accuracy of a statistical learning method is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions. In other words, we could calculate <img src="https://latex.codecogs.com/svg.image?\hat{f}^1(x),\hat{f}^2(x),\cdots,\hat{f}^B(x)"> using **B** separate training sets, and average them in order to obtain a single low-variance statistical learning model, given by,

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.027.png)

Of course, this is not practical because we generally do not have access to multiple training sets. Instead, we can bootstrap, by taking repeated samples from the (single) training data set. In this approach we generate B different bootstrapped training data sets. We then train our method on the bth bootstrapped training set in order to get <img src="https://latex.codecogs.com/svg.image?\hat{f}^{*B}(x)">, and finally average all the predictions, to obtain

![](images/ds-interview/Aspose.Words.b499c499-1cb7-4a67-bd0b-7d4698e5c020.028.png)

## OOB Error Estimation

***[Source: ISLR Page-317]*** The key to bagging is that trees are repeatedly fit to bootstrapped subsets of the observations. One can show that on average, each bagged tree makes use of around two-thirds of the observations. The remaining one-third of the observations not used to fit a given bagged tree are referred to as the out-of-bag (OOB) observations. We can predict the response for the ith observation using each of the trees in which that observation was OOB. This will yield around B/3 predictions for the ith observation. In order to obtain a single prediction for the ith observation, we can average these predicted responses (if regression is the goal) or can take a  majority vote (if classification is the goal). This leads to a single OOB prediction for the ith observation. An OOB prediction can be obtained in this way for each of the n observations, from which the overall OOB MSE (for a regression problem) or classification error (for a classification  problem) can be computed. The resulting OOB error is a valid estimate of the test error for the  bagged model, since the response for each observation is predicted using only the trees that were not fit using that observation.

## How Random Forests ensure that trees are decorrelated?

***[Source: ISLR Page-320]*** Random forests provide an improvement over bagged trees by way of a random small tweak  that decorrelates the trees. As in bagging, we build a number  forest of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors. A fresh sample of √m predictors is taken at each split, and typically we choose m ≈ p—that is, the number of predictors considered at each split is approximately equal to the square root of the total number of predictors (4 out of the 13 for the Heart data). In other words, in building a random forest, at each split in the tree, the algorithm is not even allowed to consider a majority of the available predictors. This may sound crazy, but it has a clever rationale. Suppose that there is one very strong predictor in the data set, along with a number of other moderately strong predictors. Then in the collection of bagged trees, most or all of the trees will use this strong predictor in the top split. Consequently, all of the bagged trees will look quite similar to each other. Hence the predictions from the bagged trees will be highly correlated. Unfortunately, averaging many highly correlated quantities does not lead to as large of a reduction in variance as averaging many uncorrelated quantities. In particular, this means that bagging will not lead to a substantial reduction in variance over a single tree in this setting. Random forests overcome this problem by forcing each split to consider only a subset of the predictors. Therefore, on average (p − m)/p of the splits will not even consider the strong predictor, and so other predictors will have more of a chance. We can think of this process as decorrelating the trees, thereby making the average of the resulting trees less variable and hence more reliable. The main difference between bagging and random forests is the choice of predictor subset size m. For instance, if a random forest is built using m = p, then this amounts simply to bagging.

## Does Random Forest overfit if we increase the number of trees?

***[Source: ISLR Page-321]*** As with bagging, random forests will not overfit if we increase B, so in practice we use a value of B sufficiently large for the error rate to have settled down.

## Does Boosting overfit if we increase the number of trees?

***[Source: ISLR Page-323]*** Unlike bagging and random forests, boosting can overfit if B is too large, although this overfitting tends to occur slowly if at all. We use cross-validation to select B. (B is the number of trees)

## Compare DT, Bagging, RF, Boosting



|Model|Description|
| :-- | :-- |
|Bagging|We build a number of decision trees on bootstrapped training samples using all the predictors|
|RF|<p>We build a number of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose m ≈ √p. </p><p>Why does RF reduce more variance as compared to Bagging? Suppose that there is one very strong predictor in the data set, along with a number of other moderately strong predictors. Then in the collection of bagged trees, most or all of the trees will use this strong predictor in the top split. Consequently, all of the bagged trees will look quite similar to each other. Hence the predictions from the bagged trees will be highly correlated. Unfortunately, averaging many highly correlated quantities does not lead to as large of a reduction in  variance as averaging many uncorrelated quantities. In particular, this means that bagging will not lead to a substantial reduction in variance  over a single tree in this setting. Random forests overcome this  problem by forcing each split to consider only a subset of the predictors. Therefore, on average (p − m)/p of the splits will not even consider the strong predictor, and so other predictors will have more of a chance. We can think of this process as decorrelating the trees, thereby making the average of the resulting trees less variable and hence more reliable.</p>|
|Boosting|Boosting works similar to Bagging, except that the trees are grown sequentially: each tree is grown using information from previously grown trees. Boosting does not involve bootstrap sampling; instead each tree is fit on a modified version of the original data set.|

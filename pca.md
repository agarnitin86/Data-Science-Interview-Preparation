## PCA

## Does scaling affect PCA outcome?

***[Source: ISLR Page-381]*** If in a dataset variables are measured on different scales, then the variable with large values might have high variance. If we perform PCA on the unscaled variables, then the first principal component loading vector will have a very large loading for this high variance variable, since that variable has by far the highest variance. 

It is undesirable for the principal components obtained to depend on an arbitrary choice of scaling, and hence, we typically scale each variable to have standard deviation one before we perform PCA. 

## What are loading vectors and score vectors?

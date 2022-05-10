# SVM

## What is Hyperplane?
***[Source: ISLR Page-68]***: In a p-dimensional space, a hyperplane is a flat affine subspace of hyperplane dimension p −1 For instance, in two dimensions, a hyperplane is a flat one-dimensional subspace—in other words, a line. In three dimensions, a hyperplane is a flat two-dimensional subspace—that is, a plane. In p > 3 dimensions, it can be hard to visualize a hyperplane, but the notion of a
(p − 1)-dimensional flat subspace still applies. The mathematical definition of a hyperplane is quite simple. In two dimensions, a hyperplane is defined by the equation

![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.003.png)

In p-dimensional setting,

![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.004.png)

If a point <img src="https://latex.codecogs.com/svg.image?X=(X_1,X_2,\cdots,X_p)^T"> in p-dimensional space (i.e. a vector of length p) satisfies above eq., then X lies on the hyperplane.
Now, suppose that X does not satisfy the eq; rather,

![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.005.png), then, X lies to one side of the hyperplane

On the other hand, If, 

![](images/ds-interview/Aspose.Words.95ba44c8-92c8-4d90-8a97-630964b6dcab.006.png) then, X lies to the other side of the hyperplane.

Some Resource on equation of line  <https://math.stackexchange.com/questions/2533114/equation-of-a-hyperplane-in-two-dimensions>

## Variations of SVM
1. Maximal Margin Classifier
1. Support Vector Classifier
1. Support Vector Machines

## Comparison b/w Maximal Margin Classifier, Support Vector Classifier, Support Vector Machine


|MM Classifier|SVC|SVM|
| :- | :- | :- |
|Used when separating hyperplane exist.|generalization of the maximal margin classifier to the non-separable case|Generalization of SVC to non-separable & non-linear cases using Kernels|
|When separating hyperplane does not exist, the optimization problem has no solution with M > 0|Uses soft margin to identify hyperplane that almost separates the classes|Uses soft margin to identify hyperplane that almost separates the classes|
|An observation can only be on the right side of the margin, and the hyperplane|An observation can be not only on the wrong side of the margin, but also on the wrong side of the hyperplane.|An observation can be not only on the wrong side of the margin, but also on the wrong side of the hyperplane.|
||Only observations that either lie on the margin or that violate the margin will affect the hyperplane, and hence the classifier obtained.|–Same as SVC–|
||Changing the position of an observation that lies strictly on the correct side of the margin would not change the classifier at all, provided that its position remains on the correct side of the margin|–Same as SVC–|
||Observations that lie directly on the margin, or on the wrong side of the margin for their class, are known as support vectors|–Same as SVC–|
||When the tuning parameter C is large, then the margin is wide, many observations violate the margin, and so there are many support vectors. In this case, many observations are involved in determining the hyperplane. This classifier has low variance and potentially high bias. When C is small ->> fewer support vector ->> High variance, low bias||


## How does SVM select support vectors?

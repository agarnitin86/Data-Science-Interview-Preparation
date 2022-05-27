## Exploratory Data Analysis 

## Explain Boxplots
[Understanding Boxplots. The image above is a boxplot. A boxplot… | by Michael Galarnyk | Towards Data Science](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

[Explaining the 68-95-99.7 rule for a Normal Distribution | by Michael Galarnyk | Towards Data Science](https://towardsdatascience.com/understanding-the-68-95-99-7-rule-for-a-normal-distribution-b7b7cbf760c2)

## Explain QQ plots

[Q-Q plot - Ensure Your ML Model is Based on the Right Distribution (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2021/09/q-q-plot-ensure-your-ml-model-is-based-on-the-right-distributions/)

[Q-Q Plots Explained. Explore the powers of Q-Q plots. | by Paras Varshney | Towards Data Science](https://towardsdatascience.com/q-q-plots-explained-5aa8495426c0)

[Q–Q plot - Wikipedia](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)

## Techniques for Missing value imputation
1. Descriptive statistics like mean, median, mode, or constant value
1. Using regression/classification to predict the missing values (sklearn IterativeImputer)
1. Regression techniques are used to interpolate/extrapolate missing values. [Interpolation vs. Extrapolation: What's the Difference? - Statology](https://www.statology.org/interpolation-vs-extrapolation/)

## Feature Scaling and Normalization
[About Feature Scaling and Normalization (sebastianraschka.com)](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

1. Z-score = x-mean(x)/std(x) : mean = 0, std = 1 for the new data. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
1. Min max scaling = (X - Xmin)/(Xmax - Xmin) : range of new data = 0,1

## Transformation
1. Reduce the skewness of data using log,exponential, square root
1. Skewness refers to the presence of outliers in the data stretching towards right or left of the normal distribution
1. Kurtosis is the measure of sharpness of the peak of the normal distribution
1. Use **spatial sign** for multiple predictors: 

## Handling Outliers
**Detection:**

  
1. Using z-score
1. Using box plots
 
        Q1 = np.percentile(data, 25, interpolation = 'midpoint')
        Q2 = np.percentile(data, 50, interpolation = 'midpoint')
        Q3 = np.percentile(data, 75, interpolation = 'midpoint')
        IQR = Q3-Q1
        Outlier = Q1 – 1.5IQR and Q3 + 1.5QIQR
        print('Interquartile range is', IQR)


**Treatment**:

1. Drop the outlier records
1. Cap your outlier data
1. Assign a new value
1. Try a new transformation

## Encoding Categorical Variables
1. **One hot encoding**:  how to do it for large vectors? [How to Handle Categorical Features | by Ashutosh Sahu | Analytics Vidhya | Medium](https://medium.com/analytics-vidhya/how-to-handle-categorical-features-ab65c3cf498e)
1. **One hot encoding with Multiple Categories**: In this technique, instead of creating the new column for every category, they limit creating the new column for 10 most frequent categories.
1. **Ordinal Number Encoding:** In this technique, each unique category value is given an integer value. For instance, “red” equals 1, “green” equals 2 and “blue” equals 3.
1. **Count or Frequency Encoding:** In this technique we will substitute the categories by the count of the observations that show that category in the dataset
1. **Target Guided Ordinal Encoding:** 
   1. Choose a categorical variable.
   1. Take the aggregated mean of the categorical variable and apply it to the target variable.
   1. Assign higher integer values or a higher rank to the category with the highest mean.
1. **Mean Ordinal Encoding:** Replace the category with the obtained mean value instead of assigning integer values to it.
1. **Probability Ratio Encoding:** This technique is suitable for classification problems only when the target variable is binary(Either 1 or 0 or True or False). In this technique, we will substitute the category value with the probability ratio i.e. P(1)/P(0).
   1. Using the categorical variable, evaluate the probability of the Target variable (where the output is True or 1).
   1. Calculate the probability of the Target variable having a False or 0 output.
   1. Calculate the probability ratio i.e. P(True or 1) / P(False or 0).
   1. Replace the category with a probability ratio.
1. **Weight of Evidence** Explained later

1. **Label Encoding**

## Weight of Evidence & Information Value 
[Weight of Evidence (WOE) and Information Value (IV) Explained (listendata.com)](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html#:~:text=The%20WOE%20should%20be%20monotonic%2C%20i.e.%20either%20growing,smoothing%20-%20the%20fewer%20bins%2C%20the%20more%20smoothing.)

## Multivariate analysis

1. Add additional variables to the chart using hue
1. Add additional variables to the chart using columns
1. Using FacetGrid

## Common Analytics Functions 

First_value, last_value, nth_value, lead, lag, rank, dense_rank, cume_dist, percent_value

[Spark Window Functions with Examples - Spark by {Examples} (sparkbyexamples.com)](https://sparkbyexamples.com/spark/spark-sql-window-functions/)

[Top 5 SQL Analytic Functions Every Data Analyst Needs to Know | by Dario Radečić | Towards Data Science](https://towardsdatascience.com/top-5-sql-analytic-functions-every-data-analyst-needs-to-know-3f32788e4ebb)

[SQL Functions | SQL Functions For Data Analysis (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2020/07/sql-functions-for-data-analysis-tasks/)

[Built-in Functions - Spark 3.2.1 Documentation (apache.org)](https://spark.apache.org/docs/latest/sql-ref-functions-builtin.html#aggregate-functions)

## Which algorithms are sensitive to feature scaling and normalization
## What is Variable Clustering?

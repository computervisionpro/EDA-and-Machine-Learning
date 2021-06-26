# EDA-and-Machine-Learning

## Overview

This notebook models a given data set using the classification technique. It also uses various preprocessing & Exploratory data analysis steps.


# Requirements

Libraries used and their versions:

1. Numpy=1.19.2
2. Pandas=1.1.3
3. Matplotlib=3.3.2
4. Seaborn=0.11.0
5. Statsmodels=0.12.0
6. Scikit-Learn=0.23.2

-------

The notebook contains 5 major sections:
1. Data preprocessing and EDA
2. Feature Engineering
3. Model building
4. Final model performance using best hyperparameters
5. Prediction of test data


# Data preprocessing and EDA

In this step, I have first eliminated the first column with serial numbers. Then used `df.shape` to check the dimensions. The `df.describe()` showed the range for all features are different, for example max value of `X1` is 4.34 but for `X57` its more than 10K. Hence, I have used scaling. There were no null values present and all of the features were numerical, hence no encoding was required. A class imbalance was found using df['Y'].value_counts() method, 


<img width="307" alt="class_imbalance" src="https://user-images.githubusercontent.com/40919247/123505679-17e46700-d67e-11eb-896a-7a6985244107.png">


There are many methods to tackle class imbalance like collecting more data, duplication of data for which labels are less or deleting data for which labels are more etc. Here. we will using weighting technique wherein we will assign more weights to label 1 and less weights to label 0 while training, using the formula:

`weight = total_samples / (n_classes * class_samples)`


On checking linear correlation between different features, it was found that features X32 and X34 were highly correlated with each other. Also, few features like in the middle region also seem to be correlated with each other to certain extent.

<img width="391" alt="corr" src="https://user-images.githubusercontent.com/40919247/123506231-c38eb680-d680-11eb-8dfc-fd2a250d6dc4.png">

I tried removing outliers using `IQR` technique, but it did not work for the data set, as masking lead to many missing values.



# Feature Engineering

In this first step I did was calculate Variance Inflation Factor of all the features. From this, I found that features X32 and X34 had a very high VIF, which strengthens our previous claim in EDA step. Hence, we will be removing one of them while and checking the VIF again and repeat till there are no features with VIF > 5. 
I also scaled the data, as mentioned in the EDA step. After that we use ANOVA using `f_classif` function to get p-values. Here, X38 & X12 have p-values > 0.05, so that should be removed.

Then I used SelectKBest() function to get first 35 most significant features, to be used for modeling (by trial and error). Those feature were:




# Model Building

For this I considered 2 methods:
- Logistic Regression
- KNN


For Logistic regression, we considered 4 iterations with different features. In First,  I considered all features including those with high VIF and p-value and in Second iteration, dropped X38 and X12, as they had high p-values, for comparison. In third, I used 35 most significant features and got validation f1-score of 0.91


Before doing iteration 4, I dropped X38, X12 due to high p_value also X32 due to high VIF and recheck VIF. It was found none of the entries now have VIF > 5. We preoceed to model with this data and found validation f1-score of 0.90, less than iteration 3. 

We will neglect first two, as they had insignificant and high VIF features and take model of Iteration 3 for further consideration.

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


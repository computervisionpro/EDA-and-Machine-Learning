# EDA-and-Machine-Learning

## Overview
------

This notebook models a given data set using the classification technique. It also uses various preprocessing & Exploratory data analysis steps.

# Requirements
------

1. Numpy=1.19.2
2. Pandas=1.1.3
3. Matplotlib=3.3.2
4. Statsmodels=0.12.0
5. Scikit-Learn=0.23.2


The notebook contains 5 major sections:
1. Data preprocessing and EDA
2. Feature Engineering
3. Model building
4. Final model performance using best hyperparameters
5. Prediction of test data


# Data preprocessing and EDA

In this step, I have first eliminated the first column with serial numbers. Then used `df.shape` to check the dimensions. The `df.describe()` showed the range for all features are different, for example max value of `X1` is 4.34 but for `X57` its more than 10K. Hence, I have used scaling. There were no null values present and all of the features were numerical, hence no encoding was required. A class imbalance was found using df['Y'].value_counts() method, 


<img width="307" alt="class_imbalance" src="https://user-images.githubusercontent.com/40919247/123505679-17e46700-d67e-11eb-896a-7a6985244107.png">



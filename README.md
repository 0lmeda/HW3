# CSCI 3329 Homework #3
**Student:** Jose Olmeda
**Dataset:** Predict Students' Dropout and Academic Success

## DataSet
**Source:** UCI Machine Learning Repository
**Samples:** 4,424
**Features:** 36
**Classes:** 3 - Dropout, Enrolled, Graduate

## Algorithm Comparison
10-fold cross validation repeated 100 times (1,000 evaluations per model).

| Algorithm | Mean Accuracy | Std |
|-----------|---------------|-----|
| Linear Classifier | 0.7040 | 0.0269 |
| Logistic Regression | 0.7653 | 0.0192 |
| KNN | 0.6966 | 0.0203 |
| Gaussian NB | 0.6786 | 0.0224 |
| Neural Network | 0.7114 | 0.0212 |

Logistic Regression perfomed best. The decision boundaries between the rhee classes appear mostly linear, which favors this model.
Gaussian NB was the weakest one, because it assumes all features are independent, which in this case it is not true at all.
KNN Struggled because with 36 features, distance calculations become less reliable.
Neural Network did not fully converge, limiting its perfomance.

In summary, the gap between logistic regression and the others is larger than their standard deviations, only confirming that there's a reak difference and not random vaqriation.

## Feature Selection
**Method:** Forward Selection using 'SequentialFeatureSelector'.

Exhaustive search was not used because with m=36 features there are over
68 billion possible subsets and the pc would take a long time. 
Forward Selection was chosen instead, which adds one feature at a
time and costs roughly O(m^2) evaluations.

Note: 'n_repeats' was reduced from 100 to 10 during feature selection
due to computational constrains.

| Algorithm | Features Selected | Mean Accuracy | Std |
|-----------|-------------------|---------------|-----|
| Linear Classifier | 18 | 0.7096 | 0.0345 |
| Logistic Regression | 18 | 0.7729 | 0.0188 |
| KNN | 18 | 0.7477 | 0.0186 |
| Gaussian NB | 18 | 0.7335 | 0.0215 |
| Neural Network | 18 | 0.7705 | 0.0195 |

## 5. Discussion

| Algorithm | Part 2 | Part 3 | Change |
|-----------|--------|--------|--------|
| Linear Classifier | 0.7040 | 0.7096 | +0.0056 |
| Logistic Regression | 0.7653 | 0.7729 | +0.0076 |
| KNN | 0.6966 | 0.7477 | +0.0511 |
| Gaussian NB | 0.6786 | 0.7335 | +0.0549 |
| Neural Network | 0.7114 | 0.7705 | +0.0591 |

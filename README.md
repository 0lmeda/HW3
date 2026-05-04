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

Algorithm           | Mean Accuracy  | Std |
Linear Classifier   |    0.7040      | 0.0269 |
Logistic Regression |    0.7653      | 0.0192 |
KNN                 |    0.6966      | 0.0203 |
Gaussian NB         |    0.6786      | 0.0224 |
Neural Network      |    0.7114      | 0.0212 |

Logistic Regression perfomed best. The decision boundaries between the rhee classes appear mostly linear, which favors this model.
Gaussian NB was the weakest one, because it assumes all features are independent, which in this case it is not true at all.
KNN Struggled because with 36 features, distance calculations become less reliable.
Neural Network did not fully converge, limiting its perfomance.

In summary, the gap between logistic regression and the others is larger than their standard deviations, only confirming that there's a reak difference and not random vaqriation.

## Feature Selection

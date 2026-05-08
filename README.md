# CSCI 3329 Homework #3

**Student:** Jose Olmeda

**Dataset:** Predict Students' Dropout and Academic Success

## Dataset
**Source:** UCI Machine Learning Repository

**Samples:** 4,424

**Features:** 36

**Classes:** 3 - Dropout, Enrolled, Graduate


A college collected data on their students to predict who will drop out, stay enrolled, or graduate.
The data includes things like the student's background, demographics, and their grades after the first and second semester.
The dataset has 3 possible outcomes — Dropout, Enrolled, or Graduate — but it's unbalanced, meaning one outcome appears way more than the others (in this case, Graduate is ~50% of the data).

## Preprocessing

The dataset was loaded from a semicolon-separated CSV file. There were no missing values, so no rows were removed. There was no 'id' column to drop.

All 36 features are already numbers, so no encoding was needed it. The target column had text ables like Dropout, Enrolled, and Graduate, so 'labelEncoder' was used to turn those into numbers.

StandScaler was applied to all 36 features so they are on the same scale.

`StandardScaler` was applied to all 36 features so they are on the same scale.
This is especially important for KNN and the Neural Network, which perform poorly when features have very different ranges.

| Step | What was done |
|------|---------------|
| Missing values | None found  no rows removed |
| ID column | Dropped if present |
| Feature encoding | Not needed  all features are already numeric |
| Target encoding | LabelEncoder converted text labels to integers |
| Scaling | StandardScaler applied to all 36 features |


## Algorithm Comparison
10-fold cross validation repeated 100 times (1,000 evaluations per model).

| Algorithm | Mean Accuracy | Std |
|-----------|---------------|-----|
| Linear Classifier | 0.7040 | 0.0269 |
| Logistic Regression | 0.7653 | 0.0192 |
| KNN | 0.6966 | 0.0203 |
| Gaussian NB | 0.6786 | 0.0224 |
| Neural Network | 0.7114 | 0.0212 |

Logistic Regression performed best. The decision boundaries between the three
classes appear mostly linear, which favors this model.

Gaussian NB was the weakest because it assumes all features are independent,
which in this case is not true at all.

KNN struggled because with 36 features, distance calculations become less reliable.

The Neural Network did not fully converge, limiting its performance.

In summary, the gap between Logistic Regression and the others is larger than
their standard deviations, confirming that there is a real difference and not
random variation.

## Feature Selection
**Method:** Forward Selection using `SequentialFeatureSelector`.

Exhaustive search was not used because with m=36 features there are over
68 billion possible subsets and the PC would take too long to finish.
Forward Selection was chosen instead, which adds one feature at a
time and costs roughly O(m²) evaluations.

Note: `n_repeats` was reduced from 100 to 10 during feature selection
due to computational constraints.

| Algorithm | Features Selected | Mean Accuracy | Std |
|-----------|-------------------|---------------|-----|
| Linear Classifier | 18 | 0.7096 | 0.0345 |
| Logistic Regression | 18 | 0.7729 | 0.0188 |
| KNN | 18 | 0.7477 | 0.0186 |
| Gaussian NB | 18 | 0.7335 | 0.0215 |
| Neural Network | 18 | 0.7705 | 0.0195 |

## Discussion

| Algorithm | Part 2 | Part 3 | Change |
|-----------|--------|--------|--------|
| Linear Classifier | 0.7040 | 0.7096 | +0.0056 |
| Logistic Regression | 0.7653 | 0.7729 | +0.0076 |
| KNN | 0.6966 | 0.7477 | +0.0511 |
| Gaussian NB | 0.6786 | 0.7335 | +0.0549 |
| Neural Network | 0.7114 | 0.7705 | +0.0591 |

All five models improved after feature selection. Dropping from 36 to 18
features removed noise and helped every model focus on the most informative variables.

- KNN improved the most (+5.11%) because fewer features made distance
calculations more meaningful.
- Gaussian NB improved by +5.49% because removing correlated features
reduced violations of the independence assumption.
- The Neural Network improved by +5.91% with cleaner input data.
- Logistic Regression remained the best overall model in both parts.

**Limitations:**
- Forward Selection is greedy and may miss the truly optimal subset.
- The Neural Network never fully converged due to iteration limits.
- Class imbalance was not addressed and could affect minority class predictions.

## Reproduction
**Python version:** 3.10+

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn
```

**Run:**
```bash
python hw3.py
```

Seeds: random_state=42 for all models; random_state=17342 for all CV objects.ShareContentpdf# CSCI 3329 Homework #3


A college collected data on their students to predicpasted

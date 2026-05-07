import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('data.csv', sep=';')

df = df.dropna()

df = df.drop(columns=['id'], errors='ignore')

y = df['Target']
x = df.drop(columns=['Target'])

for col in x.select_dtypes(include='object').columns:
    x[col] = LabelEncoder().fit_transform(x[col])

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

x_scaled = StandardScaler().fit_transform(x)

feature_names = np.array(x.columns)

print("Shape:", x_scaled.shape, 'Classes', set(y))

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score

models = {
'Linear Classifier' : Perceptron(max_iter=1000, random_state=42),
'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
'KNN' : KNeighborsClassifier(n_neighbors=5),
'Gaussian NB' : GaussianNB(),
'Neural Network' : MLPClassifier(hidden_layer_sizes=(64,),
max_iter=2000, random_state=42),

}

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=17342)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, x_scaled, y, cv=rkf, scoring='accuracy', n_jobs=-1)
    results[name] = (scores.mean(), scores.std())
    print(f'{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}')



rkf_fs = RepeatedKFold(n_splits=10, n_repeats=10, random_state=17342)

models_fs = {
    'Linear Classifier' : Perceptron(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB'        : GaussianNB(),
    'Neural Network'     : MLPClassifier(hidden_layer_sizes=(64,),
                                         max_iter=2000, random_state=42),
}

from sklearn.feature_selection import SequentialFeatureSelector

results_fs = {}
for name, model in models_fs.items():
    print(f"Running: {name} ...")

    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select='auto',
        direction='forward',
        scoring='accuracy',
        cv=rkf_fs,
        n_jobs=-1,
    )
    sfs.fit(x_scaled, y)

    selected = feature_names[sfs.get_support()]
    x_selected = x_scaled[:, sfs.get_support()]

    scores = cross_val_score(model, x_selected, y, cv=rkf_fs,
                             scoring='accuracy', n_jobs=-1)

    results_fs[name] = {
        'features': list(selected),
        'mean': scores.mean(),
        'std': scores.std()
    }

    print(f'{name:20s}  mean={scores.mean():.4f}  std={scores.std():.4f}')
    print(f'  Features ({len(selected)}): {list(selected)}\n')

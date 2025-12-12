# =============================================================
# AI–ML LAB PROGRAMS (Single File - Auto Run All Programs)
# =============================================================

print("\n======================================")
print(" PROGRAM 1: FIND-S & CANDIDATE ELIMINATION")
print("======================================\n")

import pandas as pd

# Default dataset
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

print("Dataset:\n", data)

attributes = data.columns[:-1]
target = data.columns[-1]

# FIND-S Algorithm
def find_s_algorithm(data):
    specific_h = ['0'] * len(attributes)
    for i in range(len(data)):
        if data[target][i].lower() == 'yes':
            if specific_h[0] == '0':
                specific_h = data.iloc[i, :-1].tolist()
            else:
                for j in range(len(specific_h)):
                    if specific_h[j] != data.iloc[i, j]:
                        specific_h[j] = '?'
    return specific_h

specific_hypothesis = find_s_algorithm(data)
print("\nMost specific hypothesis (FIND-S):", specific_hypothesis)

# Candidate Elimination
def more_general(h1, h2):
    return all(x == '?' or x == y for x, y in zip(h1, h2))

def generalize_S(example, S):
    for h in S:
        for i in range(len(h)):
            if h[i] != example[i]:
                h[i] = '?'
    return S

def specialize_G(example, G, domains):
    new_G = []
    for h in G:
        for i in range(len(h)):
            if h[i] == '?':
                for value in domains[i]:
                    if value != example[i]:
                        new_h = h.copy()
                        new_h[i] = value
                        new_G.append(new_h)
    return new_G

def candidate_elimination(data):
    domains = [list(data[attr].unique()) for attr in attributes]
    S = [['0'] * len(attributes)]
    G = [['?'] * len(attributes)]

    for i, row in data.iterrows():
        example = row[:-1].tolist()
        label = row[-1].lower()

        if label == 'yes':
            G = [g for g in G if more_general(g, example)]
            if S[0] == ['0'] * len(attributes):
                S[0] = example
            else:
                S = generalize_S(example, S)

        else:
            G = specialize_G(example, G, domains)
            G = [g for g in G if any(more_general(g, s) for s in S)]

    return S, G

S_final, G_final = candidate_elimination(data)
print("\nFinal Specific Boundary (S):", S_final)
print("Final General Boundary (G):", G_final)


# =============================================================
# PROGRAM 2: BAGGING & BOOSTING
# =============================================================
print("\n======================================")
print(" PROGRAM 2: BAGGING & BOOSTING")
print("======================================\n")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bag.fit(X_train, y_train)

boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
boost.fit(X_train, y_train)

bag_acc = accuracy_score(y_test, bag.predict(X_test))
boost_acc = accuracy_score(y_test, boost.predict(X_test))

print(f"Bagging Accuracy: {bag_acc:.2f}")
print(f"Boosting Accuracy: {boost_acc:.2f}")


# =============================================================
# PROGRAM 3: BAYESIAN NETWORKS
# =============================================================
print("\n======================================")
print(" PROGRAM 3: BAYESIAN NETWORK")
print("======================================\n")

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data_bn = pd.DataFrame({
    'Rain': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'Traffic': ['High', 'High', 'Low', 'Low', 'Low', 'Low', 'High', 'Low', 'High', 'High'],
    'Late': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
})

model = DiscreteBayesianNetwork([('Rain', 'Traffic'), ('Traffic', 'Late')])
model.fit(data_bn, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)
prob = inference.query(variables=['Late'], evidence={'Rain': 'Yes'})

print(prob)


# =============================================================
# PROGRAM 4: SELF-ORGANIZING MAP (SOM)
# =============================================================
print("\n======================================")
print(" PROGRAM 4: SELF-ORGANIZING MAP (SOM)")
print("======================================\n")

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = load_iris().data
X = MinMaxScaler().fit_transform(X)

som = MiniSom(x=5, y=5, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

win_map = np.array([som.winner(x) for x in X])
print("Sample SOM neuron mappings:", win_map[:10])

print("\n==============================")
print(" ALL PROGRAMS EXECUTED")
print("==============================")

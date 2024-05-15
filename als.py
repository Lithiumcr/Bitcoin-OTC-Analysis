import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from implicit.als import AlternatingLeastSquares

data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])

# ground truth: positive links are those with a positive weight, negative links are those with a negative weight
positive_links = data[data['Weight'] > 0]
negative_links = data[data['Weight'] < 0]

# create the training matrix
# Note: Implicit library assumes the input is user-item interactions matrix,
#       here we treat the 'Source' as users and 'Target' as items for simplicity.
num_users = data['Source'].max() + 1
num_items = data['Target'].max() + 1
train_matrix = np.zeros((num_users, num_items))

# fill in the training matrix
for row in data.itertuples():
    train_matrix[row.Source, row.Target] = row.Weight

# change the train_matrix to csr_matrix format and transpose it for ALS expects item-user matrix
train_matrix = csr_matrix(train_matrix).T

from sklearn.model_selection import ParameterGrid

param_grid = {
    'factors': [10, 50, 100],
    'regularization': [0.01, 0.1, 1],
    'iterations': [10, 15, 20]
}

def predict_sign_als(model, user, item):
    user_vector = model.user_factors[user, :]
    item_vector = model.item_factors[item, :]
    prediction = user_vector.dot(item_vector)
    return 1 if prediction > 0 else -1

def generate_predictions(model, data):
    predictions = []
    true_labels = []
    for row in data.itertuples():
        true_labels.append(1 if row.Weight > 0 else -1)
        predicted_sign = predict_sign_als(model, row.Target, row.Source)
        predictions.append(predicted_sign)
    return true_labels, predictions

# grid search for the best parameters
best_accuracy = 0
best_precision = 0
best_recall = 0
best_params = None

for params in ParameterGrid(param_grid):
    als = AlternatingLeastSquares(**params)
    als.fit(train_matrix)
    true_labels, predictions = generate_predictions(als, data)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_params = params

print(f'Best Accuracy: {best_accuracy}')
print(f'Best Precision: {best_precision}')
print(f'Best Recall: {best_recall}')
print(f'Best Parameters: {best_params}')
print(f'Best Model: {als}')
# print(f'Best Model User Factors: {als.user_factors}')
# print(f'Best Model Item Factors: {als.item_factors}')
# print(f'Best Model User Factors Shape: {als.user_factors.shape}')
# print(f'Best Model Item Factors Shape: {als.item_factors.shape}')
print(f'Done!')
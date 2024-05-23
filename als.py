import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from implicit.als import AlternatingLeastSquares
import networkx as nx
import matplotlib.pyplot as plt


data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])

def signed_weighted_clustering_coefficient(G):
    clustering_coeffs = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            clustering_coeffs[node] = 0.0
            continue
        
        triangles = 0
        total_triplets = 0
        
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if G.has_edge(u, v):
                    total_triplets += 1
                    if (G.has_edge(u, node) and G.has_edge(node, v) and G.has_edge(u, v)) and (G[u][node]['weight'] * G[node][v]['weight'] * G[u][v]['weight']) > 0:
                        triangles += 1
        
        if total_triplets == 0:
            clustering_coeffs[node] = 0.0
        else:
            clustering_coeffs[node] = triangles / total_triplets
    
    return clustering_coeffs

def calculate_coefficients(G):
    common_neighbors = {}
    jaccard_coefficient = {}
    preferential_attachment = {}
    adamic_adar = {}
    resource_allocation = {}
    local_clustering = {}
    
    for idx, edge in enumerate(G.edges()):
        if idx % 1000 == 0:
            print(f"Processed {idx} edges for coefficients calculation.")
        u, v = edge
        
        predecessors_u = set(G.predecessors(u))
        predecessors_v = set(G.predecessors(v))
        
        common_neighbors[edge] = len(predecessors_u & predecessors_v)
        
        union_neighbors = len(predecessors_u | predecessors_v)
        jaccard_coefficient[edge] = common_neighbors[edge] / union_neighbors if union_neighbors != 0 else 0
        
        preferential_attachment[edge] = len(predecessors_u) * len(predecessors_v)
        
        adamic_adar[edge] = sum(1 / np.log(len(list(G.successors(x)))) for x in predecessors_u & predecessors_v if len(list(G.successors(x))) > 1)
        
        resource_allocation[edge] = sum(1 / len(list(G.successors(x))) for x in predecessors_u & predecessors_v if len(list(G.successors(x))) > 0)
        
        local_clustering[u] = nx.clustering(G, u, weight='weight')
        local_clustering[v] = nx.clustering(G, v, weight='weight')
    
    return common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering

# create a directed graph
G = nx.DiGraph()
for row in data.itertuples():
    G.add_edge(row.Source, row.Target, weight=row.Weight)
print("Graph construction completed.")

# calculate the directed signed weighted clustering coefficient and other coefficients
clustering_coeffs = signed_weighted_clustering_coefficient(G)
average_clustering_coeff = np.mean(list(clustering_coeffs.values()))
common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering = calculate_coefficients(G)

print(f'Average Weighted Signed Clustering Coefficient: {average_clustering_coeff}')

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
    'factors': [50],# [10, 50, 100],
    'regularization': [0.01], # [0.01, 0.1, 1],
    'iterations': [100] # [50, 100]
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
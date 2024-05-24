import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from implicit.als import AlternatingLeastSquares
import networkx as nx
import matplotlib.pyplot as plt
from powerlaw import Fit

# Load data
data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])

# Create a directed graph
G = nx.DiGraph()
for row in data.itertuples():
    G.add_edge(row.Source, row.Target, weight=row.Weight)

# Calculate in-degrees and out-degrees
in_degrees = {}
out_degrees = {}

for node in G.nodes():
    in_weight = sum(abs(G[u][node]['weight']) for u in G.predecessors(node))
    out_weight = sum(abs(G[node][v]['weight']) for v in G.successors(node))
    
    in_degrees[node] = in_weight
    out_degrees[node] = out_weight

# Split into positive and negative
in_degrees_pos = [deg for node, deg in in_degrees.items() if deg > 0]
in_degrees_neg = [deg for node, deg in in_degrees.items() if deg < 0]
out_degrees_pos = [deg for node, deg in out_degrees.items() if deg > 0]
out_degrees_neg = [deg for node, deg in out_degrees.items() if deg < 0]

# Fit power law and plot for in-degrees and out-degrees
def plot_powerlaw(data, title):
    if len(data) > 0:
        fit = Fit(data)
        print(f'{title} - Power law alpha: {fit.alpha}, sigma: {fit.sigma}')
        plt.figure()
        fit.plot_pdf(color='b')
        fit.power_law.plot_pdf(color='r', linestyle='--')
        plt.title(title)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.legend(['Empirical', 'Power law fit'])
        plt.savefig(f'{title}-powerlaw.png')
        # plt.show()
    else:
        print(f"No data for {title}")

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
    CNo_o, CNo_i, CNi_o, CNi_i = {}, {}, {}, {}
    JCo_o, JCo_i, JCi_o, JCi_i = {}, {}, {}, {}
    PAo_o, PAo_i, PAi_o, PAi_i = {}, {}, {}, {}
    RA, AA, CC = {}, {}, {}

    for idx, edge in enumerate(G.edges()):
        if idx % 1024 == 0:
            print(f"Processed {idx} edges for coefficients calculation.")
        u, v = edge
        
        out_neighbors_u = set(G.successors(u))
        in_neighbors_u = set(G.predecessors(u))
        out_neighbors_v = set(G.successors(v))
        in_neighbors_v = set(G.predecessors(v))
        
        # CN
        CNo_o[edge] = len(out_neighbors_u & out_neighbors_v)
        CNo_i[edge] = len(out_neighbors_u & in_neighbors_v)
        CNi_o[edge] = len(in_neighbors_u & out_neighbors_v)
        CNi_i[edge] = len(in_neighbors_u & in_neighbors_v)
        
        # JC
        union_out_o = len(out_neighbors_u | out_neighbors_v)
        union_out_i = len(out_neighbors_u | in_neighbors_v)
        union_in_o = len(in_neighbors_u | out_neighbors_v)
        union_in_i = len(in_neighbors_u | in_neighbors_v)
        
        JCo_o[edge] = CNo_o[edge] / union_out_o if union_out_o != 0 else 0
        JCo_i[edge] = CNo_i[edge] / union_out_i if union_out_i != 0 else 0
        JCi_o[edge] = CNi_o[edge] / union_in_o if union_in_o != 0 else 0
        JCi_i[edge] = CNi_i[edge] / union_in_i if union_in_i != 0 else 0
        
        # PA
        PAo_o[edge] = len(out_neighbors_u) * len(out_neighbors_v)
        PAo_i[edge] = len(out_neighbors_u) * len(in_neighbors_v)
        PAi_o[edge] = len(in_neighbors_u) * len(out_neighbors_v)
        PAi_i[edge] = len(in_neighbors_u) * len(in_neighbors_v)
        
        # RA, AA, CC
        common_neighbors = out_neighbors_u & out_neighbors_v
        RA[edge] = sum(1 / len(list(G.successors(x))) for x in common_neighbors if len(list(G.successors(x))) > 0)
        AA[edge] = sum(1 / np.log(len(list(G.successors(x)))) for x in common_neighbors if len(list(G.successors(x))) > 1)
        CC[u] = nx.clustering(G, u, weight='weight')
        CC[v] = nx.clustering(G, v, weight='weight')
    print(idx)
    print("Processed all edges for coefficients calculation.")
    return (CNo_o, CNo_i, CNi_o, CNi_i), (JCo_o, JCo_i, JCi_o, JCi_i), (PAo_o, PAo_i, PAi_o, PAi_i), RA, AA, CC

# Create a directed graph
G = nx.DiGraph()
for row in data.itertuples():
    G.add_edge(row.Source, row.Target, weight=row.Weight)
print("Graph construction completed.")

plot_powerlaw(in_degrees_pos, 'Positive In-Degree Distribution')
plot_powerlaw(in_degrees_neg, 'Negative In-Degree Distribution')
plot_powerlaw(out_degrees_pos, 'Positive Out-Degree Distribution')
plot_powerlaw(out_degrees_neg, 'Negative Out-Degree Distribution')
print("Power law fit completed.")

# Calculate coefficients
clustering_coeffs = signed_weighted_clustering_coefficient(G)
average_clustering_coeff = np.mean(list(clustering_coeffs.values()))
print(f'Average Weighted Signed Clustering Coefficient: {average_clustering_coeff}')
common_neighbors, jaccard_coefficient, preferential_attachment, RA, AA, CC = calculate_coefficients(G)
print("Coefficients calculation completed.")

# Symbol Prediction Task
positive_links = data[data['Weight'] > 0]
negative_links = data[data['Weight'] < 0]

# Create the training matrix
num_users = data['Source'].max() + 1
num_items = data['Target'].max() + 1
train_matrix = np.zeros((num_users, num_items))

for row in data.itertuples():
    train_matrix[row.Source, row.Target] = row.Weight

train_matrix = csr_matrix(train_matrix).T

param_grid = {
    'factors': [50],  # [10, 50, 100],
    'regularization': [0.01],  # [0.01, 0.1, 1],
    'iterations': [200]  # [50, 100]
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

best_accuracy = 0
best_precision = 0
best_recall = 0
best_params = None

for params in ParameterGrid(param_grid):
    als = AlternatingLeastSquares(**params)
    als.fit(train_matrix)
    _, train_predictions = generate_predictions(als, data)
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
print(f'Done with Symbol Prediction!')

# Weight Prediction Task
features = []
labels = []

for edge in G.edges():
    u, v = edge
    features.append([
        common_neighbors[0][edge], common_neighbors[1][edge], common_neighbors[2][edge], common_neighbors[3][edge],
        jaccard_coefficient[0][edge], jaccard_coefficient[1][edge], jaccard_coefficient[2][edge], jaccard_coefficient[3][edge],
        preferential_attachment[0][edge], preferential_attachment[1][edge], preferential_attachment[2][edge], preferential_attachment[3][edge],
        RA[edge], AA[edge], CC[u], CC[v]
    ])
    labels.append(abs(G[u][v]['weight']))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
# artificially fine-tune the predictions
# y_pred = np.where(y_pred == 0, 1, y_pred)
y_pred = np.where(y_pred > 10, 10, y_pred)

# to compare with the DL method using the same scale into [-1,1]
mse = mean_squared_error(y_test, y_pred)*0.01
print(f'Mean Squared Error: {mse}')

# Combine Symbol and Weight Predictions
true_labels, sign_predictions = generate_predictions(als, data)
predicted_weights = reg.predict(features)
predicted_signed_weights = [sign * weight for sign, weight in zip(sign_predictions, predicted_weights)]

# Evaluate the combined predictions
original_weights = data['Weight'].values
combined_mse = mean_squared_error(original_weights, predicted_signed_weights)*0.01
print(f'Combined Mean Squared Error: {combined_mse}')

# Frequency distribution histograms
plt.figure()
plt.hist([y_pred, y_test], bins=30, range=(0, 10), label=['Predicted (without sign)', 'True'])
plt.legend(loc='upper right')
plt.title('Weight Frequency Distribution (Without Sign)')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.savefig('weight_freq_without_sign.png')
plt.close()

plt.figure()
plt.hist([predicted_signed_weights, data['Weight'].values], bins=30, range=(-10, 10), label=['Predicted (with sign)', 'True'])
plt.legend(loc='upper right')
plt.title('Weight Frequency Distribution (With Sign)')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.savefig('weight_freq_with_sign.png')
plt.close()

# Scatter plot of true vs predicted values
plt.figure()
plt.scatter(data['Weight'].values, predicted_signed_weights, alpha=0.5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.savefig('scatter_plot.png')
plt.close()
from config.imports import *
from config.constants import *
from functions import *

# Caricamento dei grafi
reduction_methods = [
    "Point Cloud", 
    "PCA", 
    "Clustering" 
    #"VAE"
    ]
results = []

for method in reduction_methods:
    graphs_train, graphs_test = torch.load(f"graphs_{method}.pt")

    # Training della GNN
    gnn_model = ParticleGNN(input_dim=graphs_train[0].x.shape[1], hidden_dim=64, output_dim=2)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    train_gnn(gnn_model, prepare_graph_dataloader(graphs_train, batch_size), gnn_optimizer, epochs=5)
    gnn_accuracy = evaluate_gnn(gnn_model, prepare_graph_dataloader(graphs_test, batch_size))
    results.append({"Method": method, "GNN Accuracy": gnn_accuracy})

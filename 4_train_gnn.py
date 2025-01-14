from config.imports import *
from config.constants import *
from functions import *
from torch_geometric.loader import DataLoader  # Usa il DataLoader di torch_geometric

# Caricamento dei grafi
reduction_methods = [
    "Point Cloud", 
    "PCA", 
    "Clustering"
    #"VAE"
]
results = []

for method in reduction_methods:
    # Usa weights_only=True per prevenire l'avviso
    graphs_train, graphs_test = torch.load(f"graphs/graphs_{method}.pt", weights_only=True)

    # Training della GNN
    gnn_model = ParticleGNN(input_dim=graphs_train[0].x.shape[1], hidden_dim=64, output_dim=2)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

    # Assicurati che DataLoader sia di torch_geometric
    train_loader = DataLoader(graphs_train, batch_size=batch_size, shuffle=True, follow_batch=["x"])
    test_loader = DataLoader(graphs_test, batch_size=batch_size, shuffle=False, follow_batch=["x"])

    train_gnn(gnn_model, train_loader, gnn_optimizer, epochs=5)
    gnn_accuracy = evaluate_gnn(gnn_model, test_loader)
    results.append({"Method": method, "GNN Accuracy": gnn_accuracy})

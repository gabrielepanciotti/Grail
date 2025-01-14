from config.imports import *
from config.constants import *
from functions import *
from torch_geometric.loader import DataLoader  # Usa il DataLoader di torch_geometric

# Determina il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento dei grafi
reduction_methods = [
    "Point Cloud", 
    "PCA", 
    "Clustering"
    #"VAE"
]
results = []

for method in reduction_methods:
    # Caricamento dei grafi
    graphs_train, graphs_test = torch.load(f"graphs/graphs_{method}.pt")

    # Sposta i grafi sul dispositivo
    graphs_train = [graph.to(device) for graph in graphs_train]
    graphs_test = [graph.to(device) for graph in graphs_test]

    # Inizializzazione del modello
    gnn_model = ParticleGNN(input_dim=graphs_train[0].x.shape[1], hidden_dim=128, output_dim=2).to(device)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

    # Creazione dei DataLoader
    train_loader = DataLoader(graphs_train, batch_size=batch_size, shuffle=True, follow_batch=["x"])
    test_loader = DataLoader(graphs_test, batch_size=batch_size, shuffle=False, follow_batch=["x"])

    # Training del modello
    train_gnn(gnn_model, train_loader, test_loader, gnn_optimizer, epochs=15)

    # Valutazione del modello
    gnn_accuracy = evaluate_gnn(gnn_model, test_loader)
    results.append({"Method": method, "GNN Accuracy": gnn_accuracy})

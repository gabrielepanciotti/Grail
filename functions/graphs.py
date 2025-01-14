from config.imports import *
from config.constants import *

def visualize_graph(graph, title="Graph Visualization"):
    """
    Visualizza un grafo usando NetworkX e Matplotlib.

    Args:
        graph (torch_geometric.data.Data): Grafo PyTorch Geometric da visualizzare.
        title (str): Titolo del grafico.
    """
    # Converte il grafo PyTorch Geometric in un grafo NetworkX
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    
    # Estrae le feature dei nodi per colorare i nodi (se disponibili)
    if graph.x is not None and graph.x.size(1) > 0:
        # Sposta le feature dei nodi su CPU
        node_colors = graph.x[:, 0].cpu().numpy()  # Prima feature come colore
    else:
        node_colors = None
    
    # Crea una figura e un asse
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Disegna il grafo
    pos = nx.spring_layout(nx_graph)  # Layout del grafo
    nodes = nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_size=50, cmap=cm.viridis, 
        node_color=node_colors, alpha=0.8
    )
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)
    
    # Aggiungi il colorbar solo se i colori dei nodi sono definiti
    if node_colors is not None:
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Node Feature (First Dimension)")
    
    # Titolo e layout
    ax.set_title(title)
    plt.axis("off")
    plt.show()


def visualize_all_graphs(graphs):
    """
    Visualizza tutti i grafi in una lista utilizzando la funzione visualize_graph.

    Args:
        graphs (list of torch_geometric.data.Data): Lista di grafi PyTorch Geometric.
    """
    for i, graph in enumerate(graphs):
        print(f"Visualizzazione del grafo {i + 1}/{len(graphs)}:")
        visualize_graph(graph, title=f"Grafo {i + 1}")


# Funzione per processare tutte le point cloud e generare i grafi
def process_point_clouds_to_graphs(point_clouds, k=10):
    """
    Processa una lista di point cloud e crea una lista di grafi.

    Args:
        point_clouds (list of np.ndarray): Lista di point cloud.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list of torch_geometric.data.Data: Lista di grafi per PyTorch Geometric.
    """
    graphs = []
    for point_cloud in point_clouds:
        if point_cloud.shape[0] < 2:
            print(f"Skipping point cloud with insufficient points: {point_cloud.shape[0]}")
            continue
        graph = create_graph_from_point_cloud(point_cloud, k=k)
        graphs.append(graph)
    return graphs

def process_point_clouds_to_graphs_with_labels(point_clouds, targets, k=10):
    """
    Processa una lista di point cloud e crea una lista di grafi, aggiungendo le etichette.

    Args:
        point_clouds (list of np.ndarray): Lista di point cloud.
        targets (list of int): Lista delle etichette per ogni point cloud.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list of torch_geometric.data.Data: Lista di grafi con etichette.
    """
    graphs = []
    for point_cloud, target in zip(point_clouds, targets):
        if point_cloud.shape[0] < 2:
            #print(f"Skipping point cloud with insufficient points: {point_cloud.shape[0]}")
            continue
        graph = create_graph_from_point_cloud(point_cloud, k=k)
        graph.y = torch.tensor([target], dtype=torch.long)  # Aggiunge l'etichetta al grafo
        graphs.append(graph)
    return graphs

class ParticleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParticleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN Layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pooling globale
        x = global_mean_pool(x, batch)  # Aggrega le rappresentazioni dei nodi a livello di grafo
        
        # Fully connected layer per la classificazione
        out = self.fc(x)
        return out
    
def train_gnn(model, dataloader, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            out = model(data)  # Output a livello di grafo
            loss = F.cross_entropy(out, data.y)  # Confronta l'output del grafo con le etichette
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_gnn(model, dataloader):
    """
    Valuta le prestazioni della GNN su un dataloader di validazione o test.

    Args:
        model (torch.nn.Module): Modello GNN addestrato.
        dataloader (torch_geometric.loader.DataLoader): DataLoader PyTorch Geometric con i grafi di validazione o test.

    Returns:
        None: Stampa i risultati della valutazione.
    """
    model.eval()  # Imposta il modello in modalità di valutazione
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')  # Sposta i dati su GPU o CPU
            out = model(data)  # Predizioni del modello
            predictions = out.argmax(dim=1)  # Classe predetta (indice della massima probabilità)
            
            # Aggrega predizioni e target reali
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
    
    # Creazione del DataFrame
    df = pd.DataFrame({
        'Prediction': all_predictions,
        'Target': all_targets
    })

    # Calcola l'accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Report dettagliato (precision, recall, F1-score)
    print("Classification Report:")
    print(classification_report(all_targets, all_predictions))
    return df

def prepare_graph_dataloader(graphs, batch_size=32):
    """
    Prepara un DataLoader per grafi.
    """
    dataset = DataLoader(graphs, batch_size=batch_size, shuffle=True, follow_batch=["x"])
    
    return dataset

def save_gnn_model(model, file_path):
    """
    Salva il modello GNN in un file.

    Args:
        model (torch.nn.Module): Modello GNN addestrato.
        file_path (str): Percorso del file per salvare il modello.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Modello salvato in: {file_path}")

def create_graph_from_point_cloud(point_cloud, k=10):
    """
    Crea un grafo a partire da una point cloud utilizzando il k-nearest neighbors.

    Args:
        point_cloud (np.ndarray): Point cloud con forma (N, 4), dove N è il numero di punti e le colonne rappresentano (x, y, z, energy).
        k (int): Numero massimo di vicini per il grafo KNN.

    Returns:
        torch_geometric.data.Data: Oggetto grafo per PyTorch Geometric.
    """
    # Usa le coordinate spaziali (x, y, z) per costruire il grafo
    spatial_coords = point_cloud[:, :3]
    
    # Adatta k al numero di nodi nella point cloud
    #k = min(k, spatial_coords.shape[0] - 1)  # k deve essere < numero di nodi
    
    if k <= 0:
        raise ValueError(f"Point cloud with insufficient points: {spatial_coords.shape[0]}")
    
    edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
    edge_index = np.vstack((edge_index.row, edge_index.col))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Usa le energie come feature dei nodi
    node_features = torch.tensor(point_cloud[:, 3:], dtype=torch.float)

    # Crea il grafo
    graph = Data(x=node_features, edge_index=edge_index)
    return graph

def prepare_graph_dataloader(graphs, batch_size=32):
    """
    Prepara un DataLoader per grafi.
    """
    dataset = DataLoader(graphs, batch_size=batch_size, shuffle=True, follow_batch=["x"])
    
    return dataset

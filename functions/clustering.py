from config.imports import *
from config.constants import *

def reduce_with_clustering(data, n_clusters=50, kmeans_model=None):
    """
    Riduce la dimensionalità dei dati raggruppandoli in cluster usando K-Means.
    
    Args:
        data (np.ndarray): Dataset originale con forma (n_samples, n_features).
        n_clusters (int): Numero di cluster da formare.

    Returns:
        reduced_data (np.ndarray): Dataset ridotto, rappresentato dai centroidi dei cluster.
        inertia (float): Inerzia dei cluster (somma delle distanze al quadrato).
        reduction_time (float): Tempo impiegato per eseguire il clustering.
        compression_ratio (float): Rapporto di compressione tra dimensione originale e ridotta.
        kmeans_model (KMeans): Modello K-Means utilizzato o addestrato.
    """
    # Controlla che il numero di cluster sia valido
    if n_clusters > data.shape[0]:
        raise ValueError(f"n_clusters ({n_clusters}) non può essere maggiore del numero di campioni ({data.shape[0]}).")
    
    # Avvia il timer
    start_time = time.time()
    
    # Esegui K-Means
    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_model.fit(data)
    reduced_data = kmeans_model.transform(data)  # Centroidi dei cluster
    inertia = kmeans_model.inertia_  # Somma delle distanze al quadrato
    
    # Calcola il tempo impiegato
    reduction_time = time.time() - start_time

    # Calcola il rapporto di compressione
    original_size = data.size  # Dimensione originale (numero totale di elementi)
    reduced_size = reduced_data.size  # Dimensione ridotta (numero totale di elementi)
    compression_ratio = reduced_size / original_size  # Rapporto di compressione
    
    print(f"Clustering completato: {n_clusters} cluster formati.")
    print(f"Inerzia dei cluster: {inertia:.4f}")
    print(f"Tempo impiegato: {reduction_time:.4f} secondi")
    print(f"Rapporto di compressione: {compression_ratio:.4f}")

    return reduced_data, inertia, reduction_time, compression_ratio, kmeans_model

def prepare_cluster_data(reduced_data, labels, batch_size=32):
    """
    Prepara i dati Clustering per il DataLoader.
    """
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert_clustering_to_graph(cluster_data, labels, k=5):
    """
    Converte i dati ridotti con clustering in grafi per l'utilizzo con GNN.

    Args:
        cluster_data (np.ndarray): Dati ridotti tramite clustering.
        labels (list or np.ndarray): Etichette corrispondenti ai grafi.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list: Lista di grafi PyTorch Geometric.
    """
    graphs = []
    for i, row in enumerate(cluster_data):
        spatial_coords = np.expand_dims(row, axis=1)  # Usa le feature come proxy per le coordinate spaziali
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_features = torch.tensor(row, dtype=torch.float32).unsqueeze(1)
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long))
        graphs.append(graph)
    return graphs
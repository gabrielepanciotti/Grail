from config.imports import *
from config.constants import *

def reduce_with_pca(data, n_components=16, pca_model=None):
    """
    Riduce la dimensionalità dei dati usando Principal Component Analysis (PCA).
    
    Args:
        data (np.ndarray): Dataset originale con forma (n_samples, n_features).
        n_components (int): Numero di componenti principali da mantenere.

    Returns:
        reduced_data (np.ndarray): Dataset ridotto con forma (n_samples, n_components).
        explained_variance (float): Varianza spiegata cumulativa dai componenti principali.
        reduction_time (float): Tempo impiegato per eseguire la PCA.
        compression_ratio (float): Rapporto di compressione tra dimensione originale e ridotta.
        pca_model (PCA): Modello PCA utilizzato o addestrato.
    """
    # Controlla che il numero di componenti sia valido
    if n_components > data.shape[1]:
        raise ValueError(f"n_components ({n_components}) non può essere maggiore del numero di feature ({data.shape[1]}).")
    
    # Avvia il timer
    start_time = time.time()
    explained_variance = 0
    # Esegui la PCA
    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        reduced_data = pca_model.fit_transform(data)
        explained_variance = np.sum(pca_model.explained_variance_ratio_)  # Percentuale di varianza spiegata
        print(f"PCA completata: {n_components} componenti principali selezionati.")
        print(f"Varianza spiegata cumulativa: {explained_variance:.4f}")
    else:
        reduced_data = pca_model.transform(data)

    # Calcola il tempo impiegato
    reduction_time = time.time() - start_time
    # Calcola il rapporto di compressione
    original_size = data.size  # Dimensione originale (numero totale di elementi)
    reduced_size = reduced_data.size  # Dimensione ridotta (numero totale di elementi)
    compression_ratio = reduced_size / original_size  # Rapporto di compressione
    
    print(f"Tempo impiegato: {reduction_time:.4f} secondi")
    print(f"Rapporto di compressione: {compression_ratio:.4f}")

    return reduced_data, explained_variance, reduction_time, compression_ratio, pca_model

def prepare_pca_data(reduced_data, labels, batch_size=32):
    """
    Prepara i dati PCA per il DataLoader.
    """
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert_pca_to_graph(pca_data, labels, k=20):
    """
    Converte i dati ridotti con PCA in grafi per l'utilizzo con GNN.

    Args:
        pca_data (np.ndarray): Dati ridotti tramite PCA.
        labels (list or np.ndarray): Etichette corrispondenti ai grafi.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list: Lista di grafi PyTorch Geometric.
    """
    graphs = []
    for i, row in enumerate(pca_data):
        spatial_coords = np.expand_dims(row, axis=1)  # PCA non ha coordinate spaziali, usa le feature come proxy
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_features = torch.tensor(row, dtype=torch.float32).unsqueeze(1)  # Feature come nodi
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long))
        graphs.append(graph)
    return graphs
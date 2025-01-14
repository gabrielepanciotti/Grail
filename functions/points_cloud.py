from config.imports import *
from config.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_voxel_to_position(voxel_id):
    """
    Mappa un voxel numerico alla sua posizione spaziale (layer, raggio, angolo).
    """
    cumulative_voxels = np.cumsum([0] + LAYER_STRUCTURE)
    for layer, (start, end) in enumerate(zip(cumulative_voxels[:-1], cumulative_voxels[1:])):
        if start <= voxel_id < end:
            # Identifica il voxel all'interno del layer
            layer_voxel_id = voxel_id - start
            Nr, Nalpha = divmod(LAYER_STRUCTURE[layer], 10) if LAYER_STRUCTURE[layer] >= 10 else (1, LAYER_STRUCTURE[layer])
            r = layer_voxel_id // Nalpha
            alpha = layer_voxel_id % Nalpha
            return layer, r, alpha
    raise ValueError(f"Voxel ID {voxel_id} non valido.")

def transform_to_point_cloud(data, labels):
    """
    Trasforma i dati del dataset in una point cloud per ogni shower.
    """
    point_clouds = []
    selected_labels = []

    for index, row in data.iterrows():
        points = []
        for voxel_id, energy in enumerate(row):
            if energy > 0:  # Considera solo voxel con energia non nulla
                layer, r, alpha = map_voxel_to_position(voxel_id)
                points.append([layer, r, alpha, energy])
        if points:  # Aggiungi solo point cloud non vuote
            point_clouds.append(np.array(points))
            selected_labels.append(labels[index])
    return point_clouds, selected_labels

def normalize_point_cloud(point_clouds, scaler=None):
    """
    Normalizza ogni point cloud individualmente.
    """
    normalized_clouds = []
    scaler = StandardScaler()
    for cloud in point_clouds:
        if cloud.shape[0] > 0:  # Gestisce il caso di point cloud non vuote
            normalized_cloud = scaler.fit_transform(cloud)
            normalized_clouds.append(normalized_cloud)
    return normalized_clouds, scaler

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualizza una point cloud in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 0], c=point_cloud[:, 3], cmap='viridis')
    plt.colorbar(sc, label='Energy')
    ax.set_xlabel('R')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Layer')
    plt.title(title)
    plt.show()

    # Misura il tempo e la compressione per le nuvole di punti
def measure_point_cloud_reduction(dataloader):
    """
    Misura il tempo e la quantità di riduzione per le nuvole di punti.

    Args:
        dataloader (DataLoader): DataLoader contenente i dati grezzi del calorimetro.

    Returns:
        tuple: (tempo, rapporto di compressione)
    """
    start_time = time.time()
    original_size = 0
    reduced_size = 0

    for batch in dataloader:
        data = batch[0].numpy()
        for event in data:
            point_cloud = [(voxel_id, energy) for voxel_id, energy in enumerate(event) if energy > 0]
            original_size += event.size
            reduced_size += len(point_cloud) * 2  # 2 valori per punto: voxel_id e energia

    end_time = time.time()
    reduction_ratio = reduced_size / original_size
    return end_time - start_time, reduction_ratio

def reduce_with_point_cloud(data, labels, threshold=0.1):
    """
    Riduce i dati in nuvole di punti eliminando voxel con energia inferiore a una soglia.
    Misura anche il tempo e la compressione della riduzione.

    Args:
        data (pd.DataFrame): Dataset originale con voxel come colonne.
        threshold (float): Soglia di energia per includere un voxel nella nuvola di punti.

    Returns:
        point_clouds (list of np.ndarray): Lista di nuvole di punti ridotte.
        compression_ratio (float): Rapporto di compressione tra dimensione originale e ridotta.
        reduction_time (float): Tempo impiegato per la riduzione.
    """
    start_time = time.time()
    
    # Dimensione originale
    original_size = data.values.size
    
    # Trasformazione in nuvole di punti
    point_clouds, labels = transform_to_point_cloud(data, labels)

    # Rimuovi nuvole di punti vuote
    filtered_point_clouds = []
    filtered_labels = []

    for cloud, label in zip(point_clouds, labels):  # labels è l'elenco delle etichette
        if cloud.size > 0:  # Controlla che la nuvola di punti non sia vuota
            filtered_point_clouds.append(cloud)
            filtered_labels.append(label)

    # Aggiorna le variabili con i dati filtrati
    point_clouds = filtered_point_clouds
    labels = filtered_labels
    
    # Filtraggio basato sulla soglia
    reduced_point_clouds = []
    reduced_size = 0
    for cloud in point_clouds:
        if cloud.shape[0] > 0:
            filtered_cloud = cloud[cloud[:, 3] > threshold]  # Considera solo punti con energia > threshold
            reduced_point_clouds.append(filtered_cloud)
            reduced_size += filtered_cloud.size
    
    # Calcolo della dimensione ridotta
    reduced_size = sum(cloud[:, 3].size for cloud in point_clouds)  # Conta solo i valori energetici validi

    # Calcola il tempo e il rapporto di compressione
    reduction_time = time.time() - start_time
    compression_ratio = reduced_size / original_size

    print(f"Point Cloud - Tempo: {reduction_time:.4f}s, Compressione: {compression_ratio:.4f}")
    return point_clouds, labels, compression_ratio, reduction_time

def prepare_point_cloud_data(reduced_data, labels, batch_size=32, max_length=None):
    """
    Prepara i dati Point Cloud per il DataLoader.
    """
    # Determina la lunghezza massima
    if max_length is None:
        max_length = max(len(cloud.flatten()) for cloud in reduced_data)

    # Normalizza la dimensione delle point cloud
    normalized_data = []
    for cloud in reduced_data:
        flattened_cloud = cloud.flatten()
        if len(flattened_cloud) < max_length:
            # Padding con zeri
            padded_cloud = np.pad(flattened_cloud, (0, max_length - len(flattened_cloud)))
        else:
            # Troncamento se necessario
            padded_cloud = flattened_cloud[:max_length]
        normalized_data.append(padded_cloud)

    # Converti la lista in un array NumPy e poi in tensori PyTorch
    normalized_data = np.array(normalized_data, dtype=np.float32)  # Conversione in array NumPy
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    # Crea il DataLoader
    dataset = TensorDataset(tensor_data, tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert_point_cloud_to_graph(point_clouds, labels, k=20):
    """
    Converte le nuvole di punti in grafi per l'utilizzo con GNN.

    Args:
        point_clouds (list): Lista di point cloud (array numpy).
        labels (list): Lista di etichette corrispondenti ai grafi.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list: Lista di grafi PyTorch Geometric.
    """
    graphs = []
    for i, cloud in enumerate(point_clouds):
        spatial_coords = cloud[:, :3]  # Coordinate spaziali (layer, r, alpha)
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        node_features = torch.tensor(cloud[:, 3:], dtype=torch.float32).to(device)  # Energia come feature
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long)).to(device)
        graphs.append(graph)
    return graphs

def normalize_data(reduced_data):
    """
    Garantisce che tutti i record in reduced_data abbiano la stessa lunghezza.
    """
    max_length = max(len(item.flatten()) for item in reduced_data)
    normalized_data = []
    for item in reduced_data:
        flattened_item = item.flatten()
        if len(flattened_item) < max_length:
            # Applica padding
            padded_item = np.pad(flattened_item, (0, max_length - len(flattened_item)), mode='constant')
        else:
            # Troncamento se necessario
            padded_item = flattened_item[:max_length]
        normalized_data.append(padded_item)

    # Usa np.vstack per unire array uniformi
    point_clouds = np.vstack(normalized_data)
    # Trasferisci su device
    point_clouds = [torch.tensor(cloud, dtype=torch.float32).to(device) for cloud in point_clouds]
    return point_clouds

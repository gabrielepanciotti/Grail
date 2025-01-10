from config.imports import *
from config.constants import *
from functions.points_cloud import *
from functions.vae import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    """
    Carica i dati del dataset dal file specificato, escludendo 'incident_energy' e 'target'.
    """
    data = pd.read_csv(file_path)
    # Escludi incident_energy e target
    voxel_data = data.iloc[:, :-2]  # Tutte le colonne eccetto le ultime due
    # Ottieni le etichette (target)
    targets = data["target"].tolist()
    return voxel_data, targets

# Preparazione dei dati
def prepare_data(file_path, batch_size=64):
    """
    Prepara i dati da un file CSV in un DataLoader PyTorch.

    Args:
        file_path (str): Percorso al file CSV contenente i dati.
        batch_size (int): Dimensione del batch.

    Returns:
        DataLoader: Dataloader PyTorch per i dati normalizzati.
    """
    # Legge il file CSV con intestazioni
    raw_data = pd.read_csv(file_path)

    # Rimuove eventuali colonne non numeriche (es. intestazioni come "voxel_0", "incident_energy", "target")
    numeric_data = raw_data.drop(columns=["incident_energy", "target"], errors="ignore")

    
    # Converte i dati in un array NumPy
    raw_array = numeric_data.to_numpy()
    
    # Converte in tensore PyTorch
    raw_tensor = torch.tensor(raw_array, dtype=torch.float32).to(device)
    
    # Crea il dataset e il dataloader
    dataset = TensorDataset(raw_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Valutazione della compressione
def evaluate_compression(model, dataloader, latent_dim):
    original_size = 0
    compressed_size = 0
    for data in dataloader:
        batch = data[0]
        _, mu, _ = model(batch)
        original_size += batch.numel()
        compressed_size += mu.numel() * latent_dim
    compression_ratio = compressed_size / original_size
    print(f"Compression Ratio: {compression_ratio:.4f}")
    return compression_ratio

# Funzione per misurare tempo e compressione
def measure_reduction(method_name, reduction_func, data, **kwargs):
    start_time = time.time()
    reduced_data = reduction_func(data, **kwargs)
    end_time = time.time()

    original_size = data.size
    reduced_size = reduced_data.size
    compression_ratio = reduced_size / original_size

    print(f"{method_name} - Tempo: {end_time - start_time:.4f}s, Compressione: {compression_ratio:.4f}")
    return reduced_data, compression_ratio, end_time - start_time

# Funzione principale per il confronto
def main_comparison(dataloader, trained_model):
    # Misura per le nuvole di punti
    point_cloud_time, point_cloud_ratio = measure_point_cloud_reduction(dataloader)

    # Misura per il VAE
    vae_time, vae_ratio = measure_vae_reduction(trained_model, dataloader)

    # Confronto dei risultati
    print("Point Cloud:")
    print(f" - Tempo di riduzione: {point_cloud_time:.4f} secondi")
    print(f" - Rapporto di compressione: {point_cloud_ratio:.4f}")

    print("VAE:")
    print(f" - Tempo di riduzione: {vae_time:.4f} secondi")
    print(f" - Rapporto di compressione: {vae_ratio:.4f}")
from config.imports import *
from config.constants import *

# Definizione del VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)  # Per la media
        self.logvar = nn.Linear(128, latent_dim)  # Per il log della varianza

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Poiché i dati sono valori normalizzati
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Funzione di perdita (VAE)
def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence / original.size(0)

# Addestramento del VAE
def train_vae(model, dataloader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            batch = data[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    return model

def reduce_with_vae(model, dataloader, latent_dim, original_data_size, is_training=False):
    """
    Riduce i dati utilizzando un VAE addestrato, misura le prestazioni,
    e uniforma la dimensione dei dati ridotti.

    Args:
        model (torch.nn.Module): Modello VAE già addestrato.
        dataloader (DataLoader): DataLoader contenente i dati da ridurre.
        latent_dim (int): Dimensione dello spazio latente.
        original_data_size (int): Dimensione originale del dataset per calcolare il compression ratio.
        is_training (bool): Indica se siamo in fase di training (influenza il contesto).

    Returns:
        reduced_data (np.ndarray): Dataset ridotto con padding o troncamento per uniformare le dimensioni.
        compression_ratio (float): Rapporto di compressione tra dimensione originale e ridotta.
        reduction_time (float): Tempo impiegato per la riduzione.
    """
    start_time = time.time()

    reduced_data = []
    reduced_size = 0

    model.eval()  # Modalità di valutazione
    with torch.no_grad() if not is_training else nullcontext():
        for batch in dataloader:
            data = batch[0]
            mu, _ = model.encode(data)  # Solo la media nello spazio latente
            reduced_data.append(mu.cpu().numpy())
            reduced_size += mu.numel()

    # Determina la lunghezza massima per uniformare le dimensioni
    max_length = max(len(item.flatten()) for item in reduced_data)

    # Uniforma la dimensione dei dati ridotti
    reduced_data = np.array([
        np.pad(item.flatten(), (0, max_length - len(item.flatten())), mode='constant')
        if len(item.flatten()) < max_length
        else item.flatten()[:max_length]
        for item in reduced_data
    ])

    # Calcola il tempo di riduzione e il rapporto di compressione
    reduction_time = time.time() - start_time
    compression_ratio = reduced_size / original_data_size

    print(f"VAE - Tempo: {reduction_time:.4f}s, Compressione: {compression_ratio:.4f}")
    return reduced_data, compression_ratio, reduction_time

def prepare_vae_data(reduced_data, labels, batch_size=32):
    """
    Prepara i dati VAE per il DataLoader.
    """
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert_vae_to_graph(vae_data, labels, k=5):
    """
    Converte i dati ridotti con VAE in grafi per l'utilizzo con GNN.

    Args:
        vae_data (np.ndarray): Dati ridotti tramite VAE.
        labels (list or np.ndarray): Etichette corrispondenti ai grafi.
        k (int): Numero di vicini per il grafo KNN.

    Returns:
        list: Lista di grafi PyTorch Geometric.
    """
    graphs = []
    for i, row in enumerate(vae_data):
        spatial_coords = np.expand_dims(row, axis=1)  # Usa le feature come proxy per le coordinate spaziali
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_features = torch.tensor(row, dtype=torch.float32).unsqueeze(1)
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long))
        graphs.append(graph)
    return graphs
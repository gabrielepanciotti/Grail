from config.imports import *
from config.constants import *

# Definizione del VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, clamp_logvar=(-4, 4)):
        super(VariationalAutoencoder, self).__init__()
        
        # --- Check dimensione di input ---
        if input_dim <= 0:
            raise ValueError(f"input_dim non valido: {input_dim}. Assicurati che i dati abbiano feature > 0.")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clamp_logvar = clamp_logvar
        
        # --- Encoder ---
        # Riduciamo un po' la complessità (da 256 -> 128, da 128 -> 64) per stabilizzare
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)      # Per la media
        self.logvar = nn.Linear(64, latent_dim)  # Per il log(varianza)

        # --- Decoder ---
        # Se i tuoi dati non sono in [0,1], rimuovi la Sigmoid o sostituiscila
        # con nn.Identity() o con un'altra attivazione
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        # Clamping per evitare che logvar vada a +/- infinito
        min_logvar, max_logvar = self.clamp_logvar
        logvar = torch.clamp(logvar, min_logvar, max_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# Funzione di perdita (VAE)
def vae_loss(reconstructed, original, mu, logvar):
    # Ricostruzione con MSE
    recon_loss = nn.MSELoss()(reconstructed, original)
    
    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence = kl_divergence / original.size(0)  # Divisione per batch_size
    
    loss = recon_loss + kl_divergence
    return loss

# Addestramento del VAE con LR più basso + controlli
def train_vae(model, dataloader, epochs=50, lr=1e-5):
    """
    Addestra il VAE:
    - LR molto basso (1e-5) per prevenire possibili esplosioni di loss
    - Log aggiuntivi per debug
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for data in dataloader:
            batch = data[0]  # shape: (batch_size, input_dim)
            
            # Controllo: batch_size e input_dim corrispondono a model.input_dim?
            # Se batch.shape[-1] != model.input_dim, c'è un mismatch nei dati
            if batch.shape[-1] != model.input_dim:
                raise ValueError(
                    f"Mismatch dimensioni: la rete aspetta input_dim={model.input_dim}, ma batch ha shape {batch.shape}"
                )
            
            optimizer.zero_grad()
            
            reconstructed, mu, logvar = model(batch)
            
            # Controllo debug su eventuali NaN
            if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(reconstructed).any():
                print(f"ATTENZIONE: NaN in mu/logvar/reconstructed, epoch: {epoch+1}")
            
            loss = vae_loss(reconstructed, batch, mu, logvar)
            if torch.isnan(loss):
                print(f"ATTENZIONE: Loss è NaN all'epoch {epoch+1}")
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    return model


def reduce_with_vae(model, dataloader, latent_dim, original_data_size, is_training=False):
    """
    Riduce i dati utilizzando un VAE addestrato, misura le prestazioni,
    e restituisce un array (num_campioni, latent_dim).
    """
    start_time = time.time()

    model.eval()
    all_latent = []

    # Disabilita il calcolo del gradiente se non siamo in training
    with torch.no_grad() if not is_training else nullcontext():
        for data in dataloader:
            batch = data[0]
            mu, _ = model.encode(batch)
            all_latent.append(mu.cpu().numpy())

    # Concateniamo i batch lungo la dimensione 0
    reduced_data = np.concatenate(all_latent, axis=0)

    total_processed = reduced_data.shape[0]
    if total_processed != len(dataloader.dataset):
        raise ValueError(
            f"Mismatch tra dati processati ({total_processed}) e dataset ({len(dataloader.dataset)})."
        )

    reduction_time = time.time() - start_time
    reduced_size = total_processed * latent_dim
    compression_ratio = reduced_size / original_data_size

    print(f"[reduce_with_vae] Tempo: {reduction_time:.4f}s, Compressione: {compression_ratio:.4f}")
    return reduced_data, compression_ratio, reduction_time


def prepare_vae_data(reduced_data, labels, batch_size=32):
    """
    Prepara i dati VAE per il DataLoader.
    """
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(tensor_data, tensor_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Stampa di debug per confermare dimensioni
    print(f"[prepare_vae_data] Creato DataLoader con {len(dataset)} campioni, batch_size={batch_size}")
    return loader


def convert_vae_to_graph(vae_data, labels, k=5):
    """
    Converte i dati ridotti con VAE in grafi per l'utilizzo con GNN.
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

from config.imports import *
from config.constants import *

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4, clamp_logvar=(-4, 4), use_sigmoid=True):
        """
        Args:
            input_dim (int): Numero di feature di input. (es. se hai 1000 colonne dopo lo scaling, input_dim=1000)
            latent_dim (int): Dimensione dello spazio latente (default: 4).
            clamp_logvar (tuple): Limiti per il clamping di logvar (evita esplosione / underflow).
            use_sigmoid (bool): Se True, usa Sigmoid in uscita; se i dati non sono in [0,1], metti False.
        """
        super(VariationalAutoencoder, self).__init__()
        
        if input_dim <= 0:
            raise ValueError(
                f"input_dim non valido ({input_dim}). Assicurati che i tuoi dati abbiano feature > 0."
            )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clamp_logvar = clamp_logvar
        
        # ------------------- Encoder -------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Layer per media e log-varianza
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        # ------------------- Decoder -------------------
        # Se i dati sono in [0,1], Sigmoid() è ragionevole.
        # Altrimenti, sostituire con nn.Identity() (o altra attivazione) per dati non normalizzati.
        decoder_layers = [
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        ]
        if use_sigmoid:
            decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Esegue l'encoder e applica il clamp su logvar."""
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        min_logvar, max_logvar = self.clamp_logvar
        logvar = torch.clamp(logvar, min_logvar, max_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodifica dal vettore latente allo spazio originale."""
        return self.decoder(z)

    def forward(self, x):
        """
        Esegue l'intero passaggio: encode -> reparameterize -> decode.
        Ritorna (ricostruito, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """
    Calcola Recon Loss (MSE) + beta * KL Divergence.
    Se beta < 1, attenua la KL (stile beta-VAE).
    """
    # Recon (media dei quadrati degli errori)
    recon_loss = nn.MSELoss(reduction='mean')(reconstructed, original)
    
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / original.size(0)  # normalizza per batch_size
    
    # Combiniamo
    loss = recon_loss + beta * kl_div
    return loss, recon_loss, kl_div

def train_vae(model, dataloader, epochs=30, lr=1e-5, beta=0.001):
    """
    Addestra il VAE per un certo numero di epoche.
    - beta (float): fattore di scala per la KL Divergence (default: 0.001).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            batch = data[0]  # shape: (batch_size, input_dim)
            
            # Controllo dimensioni
            if batch.shape[-1] != model.input_dim:
                raise ValueError(
                    f"Mismatch dimensioni: input_dim={model.input_dim}, "
                    f"ma batch.shape={batch.shape}"
                )
            
            optimizer.zero_grad()
            
            # Forward
            reconstructed, mu, logvar = model(batch)
            
            # Calcola la loss con separazione di Recon e KL
            loss, recon_loss, kl_div = vae_loss(reconstructed, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            
            # Aggiorna contatori
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_div.item()
            num_batches += 1
            
            # Esempio di debug sul primo batch di ogni epoca
            if (epoch == 0) and (batch_idx == 0):
                print(f"[DEBUG epoch {epoch+1}, batch {batch_idx+1}] Reconstructed min={reconstructed.min().item():.4f}, "
                      f"max={reconstructed.max().item():.4f}, mean={reconstructed.mean().item():.4f}")
        
        # Medie su tutti i batch
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
    
    return model

def reduce_with_vae(model, dataloader, latent_dim, original_data_size, is_training=False):
    """
    Riduce i dati utilizzando un VAE addestrato, misura le prestazioni,
    e restituisce un array (num_campioni, latent_dim).
    """
    start_time = time.time()
    model.eval()
    all_latent = []

    with torch.no_grad() if not is_training else nullcontext():
        for data in dataloader:
            batch = data[0]
            # Otteniamo mu (mean) come embedding
            mu, _ = model.encode(batch)
            all_latent.append(mu.cpu().numpy())

    # Concateniamo i batch
    reduced_data = np.concatenate(all_latent, axis=0)

    total_processed = reduced_data.shape[0]
    if total_processed != len(dataloader.dataset):
        raise ValueError(
            f"Mismatch dati processati ({total_processed}) vs dataset ({len(dataloader.dataset)})."
        )

    # Tempo e compressione
    reduction_time = time.time() - start_time
    reduced_size = total_processed * latent_dim
    compression_ratio = reduced_size / original_data_size

    print(f"[reduce_with_vae] Tempo: {reduction_time:.4f}s, Compressione: {compression_ratio:.4f}")
    return reduced_data, compression_ratio, reduction_time

def prepare_vae_data(reduced_data, labels, batch_size=32):
    """
    Converte i dati in tensori PyTorch (float32), li impacchetta
    in un DataLoader e li restituisce.
    """
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(tensor_data, tensor_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"[prepare_vae_data] Creato DataLoader: {len(dataset)} campioni, batch_size={batch_size}")
    return loader

def convert_vae_to_graph(vae_data, labels, k=5):
    """
    Converte i dati ridotti (VAE) in grafi (PyTorch Geometric).
    Utile se devi usarli in GNN.
    """
    graphs = []
    for i, row in enumerate(vae_data):
        spatial_coords = np.expand_dims(row, axis=1)  # Proxy per coordinate
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_features = torch.tensor(row, dtype=torch.float32).unsqueeze(1)
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long))
        graphs.append(graph)
    return graphs

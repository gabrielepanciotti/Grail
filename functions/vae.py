from config.imports import *
from config.constants import *

# Definizione del VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4, clamp_logvar=(-4, 4), use_sigmoid=True):
        """
        Args:
            input_dim (int): Numero di feature di input.
            latent_dim (int): Dimensione dello spazio latente (default: 4).
            clamp_logvar (tuple): Limiti per il clamping di logvar.
            use_sigmoid (bool): Se True, usa Sigmoid in uscita. Se i dati non sono in [0,1], metti False.
        """
        super(VariationalAutoencoder, self).__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim non valido: {input_dim}. Assicurati che i dati abbiano feature > 0.")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clamp_logvar = clamp_logvar
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        # Decoder
        layers_dec = [
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        ]
        # Se vogliamo rimanere in [0,1], aggiungiamo Sigmoid
        if use_sigmoid:
            layers_dec.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers_dec)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
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


def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence = kl_divergence / original.size(0)
    return recon_loss + kl_divergence


def train_vae(model, dataloader, epochs=30, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for data in dataloader:
            batch = data[0]  # shape: (batch_size, input_dim)
            if batch.shape[-1] != model.input_dim:
                raise ValueError(f"Mismatch dimensioni: input_dim={model.input_dim}, batch.shape={batch.shape}")
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            
            # Log min/max del reconstructed (solo per debug, disattiva se spam troppo)
            if epoch % 10 == 0 and num_batches == 0:
                print(f"[DEBUG epoch {epoch+1}] reconstructed range: min={reconstructed.min().item():.4f}, "
                      f"max={reconstructed.max().item():.4f}")

            loss = vae_loss(reconstructed, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    return model


def reduce_with_vae(model, dataloader, latent_dim, original_data_size, is_training=False):
    start_time = time.time()
    model.eval()
    all_latent = []

    with torch.no_grad() if not is_training else nullcontext():
        for data in dataloader:
            batch = data[0]
            mu, _ = model.encode(batch)
            all_latent.append(mu.cpu().numpy())

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
    tensor_data = torch.tensor(reduced_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(tensor_data, tensor_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def convert_vae_to_graph(vae_data, labels, k=5):
    graphs = []
    for i, row in enumerate(vae_data):
        spatial_coords = np.expand_dims(row, axis=1)  # feature come coordinate
        edge_index = kneighbors_graph(spatial_coords, n_neighbors=k, mode='connectivity', include_self=False).tocoo()
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = torch.tensor(row, dtype=torch.float32).unsqueeze(1)
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([labels[i]], dtype=torch.long))
        graphs.append(graph)
    return graphs

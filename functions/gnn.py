from config.imports import *
from config.constants import *
from functions import *

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

def save_gnn_model(model, file_path):
    """
    Salva il modello GNN in un file.

    Args:
        model (torch.nn.Module): Modello GNN addestrato.
        file_path (str): Percorso del file per salvare il modello.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Modello salvato in: {file_path}")

# Funzione per valutare il modello GNN dato un valore di k
def evaluate_gnn_for_k(k, point_clouds, labels, model, optimizer, epochs=3, batch_size=32):
    """
    Valuta la GNN per un dato valore di k.

    Args:
        k (int): Numero di vicini per il grafo.
        point_clouds (list): Lista di point cloud.
        labels (list): Lista di etichette corrispondenti.
        model (torch.nn.Module): Modello GNN.
        optimizer (torch.optim.Optimizer): Ottimizzatore per il modello.
        epochs (int): Numero di epoche per l'addestramento.
        batch_size (int): Dimensione del batch per il DataLoader.

    Returns:
        float: Accuracy media sul dataset di validazione.
    """
    # Crea grafi per il valore specifico di k
    graphs = process_point_clouds_to_graphs_with_labels(point_clouds, labels, k=k)

    # Dividi in training e validation
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    # Crea DataLoader
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    # Addestramento del modello
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

    # Valutazione sul validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total

# Ottimizzazione di k
def find_best_k(k_values, point_clouds, labels, input_dim, hidden_dim, output_dim, lr=0.01, epochs=3):
    """
    Trova il miglior valore di k per la GNN.

    Args:
        k_values (list): Lista di valori di k da testare.
        point_clouds (list): Lista di point cloud.
        labels (list): Lista di etichette corrispondenti.
        input_dim (int): Dimensione delle feature dei nodi.
        hidden_dim (int): Dimensione dello strato nascosto della GNN.
        output_dim (int): Numero di classi di output.
        lr (float): Learning rate per l'ottimizzatore.
        epochs (int): Numero di epoche per ogni valore di k.

    Returns:
        int: Miglior valore di k.
    """
    best_k = None
    best_accuracy = 0
    results = {}

    for k in k_values:
        print(f"Valutazione per k={k}...")

        # Inizializza il modello e l'ottimizzatore
        model = ParticleGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Valuta la GNN per il valore corrente di k
        accuracy = evaluate_gnn_for_k(k, point_clouds, labels, model, optimizer, epochs=epochs)
        results[k] = accuracy

        print(f"Accuracy per k={k}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"Miglior k: {best_k} con accuracy: {best_accuracy:.4f}")
    return best_k, results
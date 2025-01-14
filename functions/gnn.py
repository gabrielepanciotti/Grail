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
        x, edge_index, batch = data.x.to(data.x.device), data.edge_index.to(data.edge_index.device), data.batch.to(data.batch.device)
        
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
            data = data.to(next(model.parameters()).device)  # Sposta i dati sullo stesso dispositivo del modello
            optimizer.zero_grad()
            out = model(data)  # Output a livello di grafo
            loss = F.cross_entropy(out, data.y)  # Confronta l'output del grafo con le etichette
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_gnn(model, dataloader):
    model.eval()  # Imposta il modello in modalità di valutazione
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(next(model.parameters()).device)  # Sposta i dati sullo stesso dispositivo del modello
            out = model(data)  # Predizioni del modello
            pred = out.argmax(dim=1)  # Classe predetta
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
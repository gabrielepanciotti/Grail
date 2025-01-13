from config.imports import *
from config.constants import *
from functions import *

class ParticleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ParticleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_dim * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Aggiunge una dimensione per il canale
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_cnn(model, train_dataloader, test_dataloader, num_epochs=30, lr=0.001, device='cuda'):
    """
    Addestra il modello CNN sui dati ridotti.

    Args:
        model (nn.Module): Modello CNN da addestrare.
        train_dataloader (DataLoader): Dati di addestramento.
        test_dataloader (DataLoader): Dati di test.
        num_epochs (int): Numero di epoche per l'addestramento.
        lr (float): Tasso di apprendimento.
        device (str): Dispositivo da utilizzare ("cuda" o "cpu").

    Returns:
        model (nn.Module): Modello addestrato.
        history (list): Storico delle perdite di addestramento.
    """
    model.to(device)  # Sposta il modello sul dispositivo
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Sposta gli input sul dispositivo
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Valutazione sul set di test
        evaluate_cnn(model, test_dataloader, device=device)

    return model, history

def evaluate_cnn(model, test_dataloader, device='cuda'):
    """
    Valuta la CNN sul set di test.

    Args:
        model (nn.Module): Modello CNN addestrato.
        test_dataloader (DataLoader): Dati di test.
        device (str): Dispositivo da utilizzare ("cuda" o "cpu").
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Sposta gli input sul dispositivo
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())  # Sposta su CPU per l'elaborazione
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
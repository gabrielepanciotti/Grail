from config.imports import *
from config.constants import *
from functions import * 


# Carica i dati
data, targets = load_data(file_path)
print("Dati caricati")
# Trasforma i dati in point cloud
point_clouds = transform_to_point_cloud(data)
print("point cloud create")
# Normalizza le point cloud
normalized_clouds, scaler = normalize_point_cloud(point_clouds)
print("point cloud normalizzate")
# Esempio di utilizzo
k_values = [20,25,30,35]
input_dim = 1  # Numero di feature per nodo
hidden_dim = 64
output_dim = 2  # Numero di classi (fotoni, pioni)

best_k, results = find_best_k(k_values, normalized_clouds, targets, input_dim, hidden_dim, output_dim)

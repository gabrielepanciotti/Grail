from config.imports import *
from config.constants import *
from functions import *

# Dizionario che mappa i nomi dei metodi di riduzione alla funzione di conversione in grafi
graph_converters = {
    "Point Cloud": convert_point_cloud_to_graph,
    "PCA": convert_pca_to_graph,
    "Clustering": convert_clustering_to_graph,
    #"VAE": convert_vae_to_graph,
}

# 1. Loop su ogni metodo di riduzione
for method, converter_func in graph_converters.items():
    print(f"\nCaricamento dati ridotti per metodo: {method}")

    # Caricamento dei dati
    # Carica i dati ridotti
    train_data_file = f"data_reduced/reduced_train_{method}.npz"
    test_data_file = f"data_reduced/reduced_test_{method}.npz"

    # Caricamento dei dati
    train_data = np.load(train_data_file, allow_pickle=True)  # Abilita allow_pickle
    test_data = np.load(test_data_file, allow_pickle=True)    # Abilita allow_pickle

    reduced_train = train_data["data"]
    train_labels = train_data["labels"]
    reduced_test = test_data["data"]
    test_labels = test_data["labels"]
    # 2. Creazione dei grafi
    graphs_train = converter_func(reduced_train, train_labels)
    graphs_test = converter_func(reduced_test, test_labels)

    # 3. Salva i grafi
    torch.save((graphs_train, graphs_test), f"graphs_{method}.pt")
    print(f"Grafi salvati in: graphs_{method}.pt")

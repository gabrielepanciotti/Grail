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
    data = np.load(f"reduced_{method}.npz", allow_pickle=True)
    reduced_train = data["train"]
    reduced_label_train = data["train_labels"]
    reduced_test = data["test"]
    reduced_label_test = data["test_labels"]

    # 2. Creazione dei grafi
    graphs_train = converter_func(reduced_train, reduced_label_train)
    graphs_test = converter_func(reduced_test, reduced_label_test)

    # 3. Salva i grafi
    torch.save((graphs_train, graphs_test), f"graphs_{method}.pt")
    print(f"Grafi salvati in: graphs_{method}.pt")

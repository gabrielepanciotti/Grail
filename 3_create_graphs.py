from config.imports import *
from config.constants import *
from functions import *

# Caricamento dei dati ridotti
reduction_methods = ["Point Cloud", "PCA", "Clustering", "VAE"]

for method in reduction_methods:
    data = np.load(f"reduced_{method}.npz", allow_pickle=True)
    reduced_train = data["train"]
    reduced_label_train = data["train_labels"]
    reduced_test = data["test"]
    reduced_label_test = data["test_labels"]

    # Creazione dei grafi
    if method == "Point Cloud":
        graphs_train = convert_point_cloud_to_graph(reduced_train, reduced_label_train)
        graphs_test = convert_point_cloud_to_graph(reduced_test, reduced_label_test)
    elif method == "PCA":
        graphs_train = convert_pca_to_graph(reduced_train, reduced_label_train)
        graphs_test = convert_pca_to_graph(reduced_test, reduced_label_test)
    elif method == "Clustering":
        graphs_train = convert_clustering_to_graph(reduced_train, reduced_label_train)
        graphs_test = convert_clustering_to_graph(reduced_test, reduced_label_test)
    else:  # VAE
        graphs_train = convert_vae_to_graph(reduced_train, reduced_label_train)
        graphs_test = convert_vae_to_graph(reduced_test, reduced_label_test)

    # Salva i grafi
    torch.save((graphs_train, graphs_test), f"graphs_{method}.pt")

# Import delle librerie necessarie
from config.imports import *
from config.constants import *
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# 1. Caricamento dei dati
print("Caricamento dati...")
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
train_labels = train_data["target"]
test_labels = test_data["target"]

# Rimuovi colonne non numeriche
train_data = train_data.drop(columns=["incident_energy", "target"])
test_data = test_data.drop(columns=["incident_energy", "target"])

# 2. Applicazione delle tecniche di riduzione
reduction_results = []

reduction_methods = {
    "PCA": reduce_with_pca,
    "Clustering": reduce_with_clustering,
    "VAE": reduce_with_vae,
    "Point Cloud": reduce_with_point_cloud,
}

for method, reducer in reduction_methods.items():
    print(f"\nMetodo di riduzione: {method}")

    # Applicazione della tecnica di riduzione
    if method == "Point Cloud":
        reduced_train, reduced_label_train, compression_ratio_train, reduction_time_train = reducer(train_data, train_labels)
        reduced_test, reduced_label_test, compression_ratio_test, reduction_time_test = reducer(test_data, test_labels)
    elif method == "VAE":
        dataloader_train = prepare_data(train_file, batch_size=batch_size)
        vae_model = VariationalAutoencoder(input_dim=train_data.shape[1], latent_dim=latent_dim).to(device)
        vae_model = train_vae(vae_model, dataloader_train)

        reduced_train, compression_ratio_train, reduction_time_train = reduce_with_vae(
            vae_model, dataloader_train, latent_dim, original_data_size=train_data.values.size
        )
        dataloader_test = prepare_data(test_file, batch_size=batch_size)
        reduced_test, compression_ratio_test, reduction_time_test = reduce_with_vae(
            vae_model, dataloader_test, latent_dim, original_data_size=test_data.values.size
        )
        reduced_label_train, reduced_label_test = train_labels, test_labels
    elif method == "PCA":
        reduced_train, explained_variance, reduction_time_train, compression_ratio_train, pca_model = reducer(train_data.values, n_components=pca_components)
        reduced_test, _, reduction_time_test, compression_ratio_test, _ = reducer(test_data.values, pca_model=pca_model)
        reduced_label_train, reduced_label_test = train_labels, test_labels
    else:  # Clustering
        reduced_train, inertia, reduction_time_train, compression_ratio_train, kmeans_model = reducer(train_data.values, n_clusters=n_clusters)
        reduced_test, _, reduction_time_test, compression_ratio_test, _ = reducer(test_data.values, kmeans_model=kmeans_model)
        reduced_label_train, reduced_label_test = train_labels, test_labels

    # Converti eventuali tensori in NumPy
    if isinstance(reduced_train, torch.Tensor):
        reduced_train = reduced_train.cpu().numpy()
    if isinstance(reduced_test, torch.Tensor):
        reduced_test = reduced_test.cpu().numpy()
    if isinstance(reduced_label_train, torch.Tensor):
        reduced_label_train = reduced_label_train.cpu().numpy()
    if isinstance(reduced_label_test, torch.Tensor):
        reduced_label_test = reduced_label_test.cpu().numpy()

    # Controllo e conversione in NumPy per point clouds
    if isinstance(reduced_train, list):  # Nel caso delle point cloud è una lista di tensori
        reduced_train = [cloud.cpu().numpy() if isinstance(cloud, torch.Tensor) else cloud for cloud in reduced_train]
        reduced_train = np.array(reduced_train, dtype=object)  # Usa un array di oggetti per liste di diversa lunghezza

    if isinstance(reduced_test, list):  # Stesso controllo per il test set
        reduced_test = [cloud.cpu().numpy() if isinstance(cloud, torch.Tensor) else cloud for cloud in reduced_test]
        reduced_test = np.array(reduced_test, dtype=object)

    # Salva i risultati della riduzione in file separati
    np.savez_compressed(f"reduced_train_{method}.npz", data=reduced_train, labels=reduced_label_train)
    np.savez_compressed(f"reduced_test_{method}.npz", data=reduced_test, labels=reduced_label_test)

    # Registra i risultati per il confronto
    reduction_results.append({
        "Method": method,
        "Compression Ratio (Train)": compression_ratio_train,
        "Compression Ratio (Test)": compression_ratio_test,
        "Reduction Time (Train) (s)": reduction_time_train,
        "Reduction Time (Test) (s)": reduction_time_test,
    })

# 3. Confronto dei risultati
reduction_results_df = pd.DataFrame(reduction_results)
print("\nRisultati Riduzione:\n", reduction_results_df)

# Salva il confronto in un file CSV
reduction_results_df.to_csv("results/reduction_results.csv", index=False)
from config.imports import *
from config.constants import *
from functions import * 

# 1. Caricamento dei dati
print("Caricamento dati...")
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
train_labels = train_data["target"]
test_labels = test_data["target"]

# Rimuovi colonne non numeriche
train_data = train_data.drop(columns=["incident_energy", "target"])
test_data = test_data.drop(columns=["incident_energy", "target"])

# Risultati per la tabella riassuntiva
results = []

# 2. Riduzione e Predizione
reduction_methods = {
    "Point Cloud": reduce_with_point_cloud,
    "Clustering": reduce_with_clustering,
    "PCA": reduce_with_pca,
    "VAE": reduce_with_vae,
}

trained_models = {}

for method, reducer in reduction_methods.items():
    print(f"\nMetodo di riduzione: {method}")

    # Riduzione
    reduction_start = time.time()
    if method == "Point Cloud":
        reduced_train, compression_ratio, reduction_time = reducer(train_data, train_labels)
        reduced_test, _, _ = reducer(test_data, test_labels)
    elif method == "VAE":
        dataloader_train = prepare_data(train_file, batch_size=batch_size)
        vae_model = VariationalAutoencoder(input_dim=train_data.shape[1], latent_dim=latent_dim, )
        vae_model = train_vae(vae_model, dataloader_train, epochs=30)
        reduced_train, _, reduction_time = reducer(vae_model, dataloader_train, latent_dim)
        dataloader_test = prepare_data(test_file, batch_size=batch_size)
        reduced_test, _, _ = reducer(vae_model, dataloader_test, latent_dim)
    elif method == "PCA":
        reduced_train, _, reduction_time, pca_model = reducer(train_data.values, n_components=pca_components)
        trained_models[method] = pca_model  # Salva il modello
        reduced_test, _, reduction_time, pca_model = reducer(test_data.values, pca_model=pca_model)
    else:  # Clustering
        reduced_train, _, reduction_time, kmeans_model = reducer(train_data.values, n_clusters=n_clusters)
        trained_models[method] = kmeans_model  # Salva il modello
        reduced_test, _, _, _ = reducer(test_data.values, kmeans_model=kmeans_model)
    
    reduction_end = time.time()
    reduction_time = reduction_end - reduction_start

    print(f"Numero di point cloud: {len(reduced_train)}")
    print(f"Numero di etichette: {len(train_labels)}")

    # Predizione con CNN
    print("Predizione con CNN...")
    if method == "Point Cloud":
        max_length = max(len(cloud.flatten()) for cloud in reduced_train)
        train_loader = prepare_point_cloud_data(reduced_train, train_labels, batch_size=batch_size, max_length=max_length)
        test_loader = prepare_point_cloud_data(reduced_test, test_labels, batch_size=batch_size, max_length=max_length)
    elif method == "VAE":
        train_loader = prepare_vae_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_vae_data(reduced_test, test_labels, batch_size=batch_size)
    elif method == "PCA":
        train_loader = prepare_pca_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_pca_data(reduced_test, test_labels, batch_size=batch_size)
    else:  # Clustering
        train_loader = prepare_cluster_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_cluster_data(reduced_test, test_labels, batch_size=batch_size)

    cnn_model = ParticleCNN(input_dim=reduced_train.shape[1], num_classes=2)
    cnn_start = time.time()
    cnn_model, _ = train_cnn(cnn_model, train_loader, test_loader, num_epochs=5)
    cnn_accuracy = evaluate_cnn(cnn_model, test_loader)
    cnn_end = time.time()
    cnn_time = cnn_end - cnn_start

    # Predizione con GNN
    print("Predizione con GNN...")
    if method == "Point Cloud":
        graphs_train = convert_point_cloud_to_graph(reduced_train, train_labels)
        graphs_test = convert_point_cloud_to_graph(reduced_test, test_labels)
    elif method == "VAE":
        graphs_train = convert_vae_to_graph(reduced_train, train_labels)
        graphs_test = convert_vae_to_graph(reduced_test, test_labels)
    elif method == "PCA":
        graphs_train = convert_pca_to_graph(reduced_train, train_labels)
        graphs_test = convert_pca_to_graph(reduced_test, test_labels)
    else:  # Clustering
        graphs_train = convert_clustering_to_graph(reduced_train, train_labels)
        graphs_test = convert_clustering_to_graph(reduced_test, test_labels)

    gnn_model = ParticleGNN(input_dim=graphs_train[0].x.shape[1], hidden_dim=64, output_dim=2)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    gnn_start = time.time()
    train_gnn(gnn_model, prepare_graph_dataloader(graphs_train, batch_size), gnn_optimizer, epochs=5)
    gnn_accuracy = evaluate_gnn(gnn_model, prepare_graph_dataloader(graphs_test, batch_size))
    gnn_end = time.time()
    gnn_time = gnn_end - gnn_start

    # Salva i risultati
    results.append({
        "Method": method,
        "Reduction Time (s)": reduction_time,
        "Compression Ratio": compression_ratio,
        "CNN Accuracy": cnn_accuracy,
        "CNN Time (s)": cnn_time,
        "GNN Accuracy": gnn_accuracy,
        "GNN Time (s)": gnn_time,
    })

# 3. Tabella Riassuntiva
results_df = pd.DataFrame(results)
print("\nRisultati:\n", results_df)

# Salva i risultati in un file CSV
results_df.to_csv("dimensionality_reduction_comparison.csv", index=False)
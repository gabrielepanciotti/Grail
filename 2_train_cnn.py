from config.imports import *
from config.constants import *
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Risultati per la tabella riassuntiva
cnn_results = []

# Dizionario dei metodi di riduzione
reduction_methods = ["Point Cloud", "PCA", "Clustering", "VAE"]

# 1. Loop su ogni metodo di riduzione
for method in reduction_methods:
    print(f"\nCaricamento dati ridotti per metodo: {method}")

    # Carica i dati ridotti
    train_data_file = f"reduced_train_{method}.npz"
    test_data_file = f"reduced_test_{method}.npz"

    # Caricamento dei dati
    train_data = np.load(train_data_file, allow_pickle=True)  # Abilita allow_pickle
    test_data = np.load(test_data_file, allow_pickle=True)    # Abilita allow_pickle

    reduced_train = train_data["data"]
    train_labels = train_data["labels"]
    reduced_test = test_data["data"]
    test_labels = test_data["labels"]

    print(f"Dati di addestramento: {reduced_train.shape}, Etichette: {train_labels.shape}")
    print(f"Dati di test: {reduced_test.shape}, Etichette: {test_labels.shape}")

    # Prepara i DataLoader per CNN
    if method == "Point Cloud":
        max_length = max(len(cloud.flatten()) for cloud in reduced_train)
        train_loader = prepare_point_cloud_data(
            reduced_train, train_labels, batch_size=batch_size, max_length=max_length
        )
        test_loader = prepare_point_cloud_data(
            reduced_test, test_labels, batch_size=batch_size, max_length=max_length
        )
        input_dim = max_length
    elif method == "VAE":
        train_loader = prepare_vae_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_vae_data(reduced_test, test_labels, batch_size=batch_size)
        input_dim = reduced_train.shape[1]
    elif method == "PCA":
        train_loader = prepare_pca_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_pca_data(reduced_test, test_labels, batch_size=batch_size)
        input_dim = reduced_train.shape[1]
    else:  # Clustering
        train_loader = prepare_cluster_data(reduced_train, train_labels, batch_size=batch_size)
        test_loader = prepare_cluster_data(reduced_test, test_labels, batch_size=batch_size)
        input_dim = reduced_train.shape[1]

    # Addestramento della CNN
    print(f"Addestramento della CNN per metodo: {method}")
    cnn_model = ParticleCNN(input_dim=input_dim, num_classes=2).to(device)
    cnn_start = time.time()
    cnn_model, cnn_train_history = train_cnn(cnn_model, train_loader, test_loader, device=device)
    cnn_end = time.time()

    # Valutazione della CNN
    cnn_accuracy = evaluate_cnn(cnn_model, test_loader)
    cnn_time = cnn_end - cnn_start

    # Salva il modello addestrato
    cnn_model_file = f"cnn_model_{method}.pth"
    torch.save(cnn_model.state_dict(), cnn_model_file)
    print(f"Modello CNN salvato in: {cnn_model_file}")

    # Salva i risultati
    cnn_results.append({
        "Method": method,
        "CNN Accuracy": cnn_accuracy,
        "CNN Time (s)": cnn_time,
    })

# 2. Tabella Riassuntiva
cnn_results_df = pd.DataFrame(cnn_results)
print("\nRisultati Addestramento CNN:\n", cnn_results_df)

# Salva i risultati in un file CSV
cnn_results_df.to_csv("cnn_results.csv", index=False)

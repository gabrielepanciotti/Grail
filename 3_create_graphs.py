from config.imports import *
from config.constants import *
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if method == 'Point Cloud':
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        train_labels = train_data["target"]
        test_labels = test_data["target"]

        # Rimuovi colonne non numeriche
        train_data = train_data.drop(columns=["incident_energy", "target"])
        test_data = test_data.drop(columns=["incident_energy", "target"])

        # Salviamo le colonne per ricostruire il DataFrame dopo lo scaling
        train_columns = train_data.columns
        test_columns = test_data.columns

        # 3. Applico il MinMaxScaler per portare i dati in [0,1]
        scaler = MinMaxScaler()

        # "Fitta" lo scaler sui dati di train e trasforma
        train_data = scaler.fit_transform(train_data)

        # Trasforma i dati di test con lo stesso scaler (senza rifittare!)
        test_data = scaler.transform(test_data)

        # Ricostruisco il DataFrame in modo da poter utilizzare .values
        train_data = pd.DataFrame(train_data, columns=train_columns)
        test_data = pd.DataFrame(test_data, columns=test_columns)
        reduced_train, train_labels, _, _ = reduce_with_point_cloud(train_data, train_labels)
        reduced_test, test_labels, _, _ = reduce_with_point_cloud(test_data, test_labels)
    else:
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
    torch.save((graphs_train, graphs_test), f"graphs/graphs_{method}.pt")
    print(f"Grafi salvati in: graphs_{method}.pt")

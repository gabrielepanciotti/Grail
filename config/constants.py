# Percorso al file del dataset
file_path = 'dataset/dataset_1_merged_withTarget.csv'
#file_path = 'dataset/reduced_dataset.csv'
# Percorso al file del dataset
train_file = 'dataset/dataset_1_merged_withTarget.csv'
#train_file = 'dataset/reduced_dataset.csv'
test_file = 'dataset/dataset_test_merged_withTarget.csv'
#test_file = 'dataset/reduced_dataset_test.csv'
batch_size = 32
latent_dim = 16
pca_components = 50  # Numero di componenti PCA
n_clusters = 50  # Numero di cluster per il clustering

# Definizione della struttura dei layer per i voxel
LAYER_STRUCTURE = [8 * 1, 16 * 10, 19 * 10, 5 * 1, 5 * 1, 15 * 10, 16 * 10, 10 * 1]  # Esempio aggiornato

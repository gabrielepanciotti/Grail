from config.imports import *
from config.constants import *
from functions import * 

data = pd.read_csv("dataset_test_merged_withTarget.csv")
# Esempio di utilizzo
reduction_ratio = 0.1  # Mantieni solo il 10% dei dati
reduced_dataset = shuffle_and_reduce_dataset(data, reduction_ratio=reduction_ratio)

print("Dataset ridotto:")
print(reduced_dataset.head())

# Salva il dataset ridotto su un file CSV, se necessario
reduced_dataset.to_csv("reduced_dataset_test.csv", index=False)

from config.imports import *
from config.constants import *

def merge_and_normalize_datasets(directory_path, output_csv):
    """
    Unisce e normalizza i dataset di test (showers e incident energies), aggiunge una colonna target,
    e unisce fotoni e pioni in un unico dataset con struttura voxel unificata.

    Args:
        directory_path (str): Percorso alla directory contenente i dataset di test.
        output_csv (str): Nome del file CSV di output.
    """
    # Definizione dei file per fotoni e pioni (test)
    datasets = {
        "photon": {
            "showers": "dataset_1_photons_2_showers.csv",
            "energies": "dataset_1_photons_2_incident_energies.csv",
        },
        "pion": {
            "showers": "dataset_1_pions_2_showers.csv",
            "energies": "dataset_1_pions_2_incident_energies.csv",
        },
    }

    max_voxels = 533  # Numero massimo di voxel (struttura dei pioni)
    merged_dataframes = []

    for particle, files in datasets.items():
        # Carica i file
        showers_path = os.path.join(directory_path, files["showers"])
        energies_path = os.path.join(directory_path, files["energies"])
        
        showers_df = pd.read_csv(showers_path)
        energies_df = pd.read_csv(energies_path, header=None, names=["incident_energy"])
        
        # Rimuovi il primo record di energies, se necessario
        if energies_df.iloc[0]["incident_energy"] == 0:
            energies_df = energies_df.iloc[1:].reset_index(drop=True)

        # Resetta gli indici
        showers_df = showers_df.reset_index(drop=True)
        energies_df = energies_df.reset_index(drop=True)

        # Verifica corrispondenza dei record
        if len(showers_df) != len(energies_df):
            raise ValueError(f"I dataset {files['showers']} e {files['energies']} non corrispondono nei record.")

        # Rinominare tutte le colonne voxel in modo coerente
        showers_df.columns = [f"voxel_{col}" if col.isdigit() else col for col in showers_df.columns]

        # Unisci showers ed energies
        merged_df = pd.concat([showers_df, energies_df], axis=1)

        # Aggiungi colonna target
        merged_df["target"] = 0 if particle == "photon" else 1

        # Aggiungi padding per normalizzare la struttura dei voxel
        for i in range(len(showers_df.columns), max_voxels):
            merged_df[f"voxel_{i}"] = 0

        merged_dataframes.append(merged_df)

    # Concatena i dataset di test di fotoni e pioni
    final_dataset = pd.concat(merged_dataframes, ignore_index=True)

    # Riordina colonne per mettere incident_energy e target alla fine
    final_columns_order = [col for col in final_dataset.columns if col not in ["incident_energy", "target"]]
    final_columns_order += ["incident_energy", "target"]
    final_dataset = final_dataset[final_columns_order]

    # Salva il dataset normalizzato e unito
    final_dataset.to_csv(output_csv, index=False)
    print(f"Dataset di test normalizzato e unito salvato in {output_csv}")
    return final_dataset

def shuffle_and_reduce_dataset(dataset, reduction_ratio=0.1, random_state=42):
    """
    Mescola il dataset e riduce il numero di record mantenendo un mix equilibrato di fotoni e pioni.

    Args:
        dataset (pd.DataFrame): Dataset unificato con colonne "target".
        reduction_ratio (float): Percentuale di dati da mantenere (es. 0.1 per il 10%).
        random_state (int): Seed per la riproducibilità.

    Returns:
        pd.DataFrame: Dataset ridotto e mescolato.
    """
    # Imposta il seed per la riproducibilità
    np.random.seed(random_state)

    # Mescola i dati
    shuffled_dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Riduci il dataset
    total_samples = int(len(shuffled_dataset) * reduction_ratio)
    reduced_dataset = shuffled_dataset.iloc[:total_samples]

    return reduced_dataset
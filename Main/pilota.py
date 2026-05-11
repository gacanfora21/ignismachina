# Intelligenza Artificiale (Behavioral Cloning) tramite K-NN Regressor
import time
import glob
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import snakeoil as snakeoil3

# Configurazioni
DATASET_FILE = "dataset.csv"
LAPS_FOLDER  = "Laps"          # Cartella contenente i file lap_*.csv
K_NEIGHBORS  = 10              # Numero di campioni vicini da consultare
MAX_STEPS    = 200_000

def merge_laps(laps_folder: str, output_path: str) -> None:
    """
    Unisce tutti i file lap_*.csv presenti in `laps_folder` in un unico
    dataset CSV, eliminando le intestazioni ridondanti dei singoli giri.
    Il file di output viene (ri)creato ogni volta che viene chiamata questa funzione.
    """
    pattern = os.path.join(laps_folder, "lap_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Nessun file lap_*.csv trovato in '{laps_folder}'. "
              "Assicurati di aver registrato almeno un giro.")
        return

    print(f"Trovati {len(files)} giri da unire:")
    for f in files:
        print(f"        - {f}")

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Dataset salvato in '{output_path}' ({len(merged)} campioni totali).\n")

class PilotaKNN:
    def __init__(self, dataset_path):
        self.model = KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights='distance')
        self.scaler = StandardScaler()
        # Impostiamo la PCA per mantenere il 97% della varianza (informazione utile)
        self.pca = PCA(n_components=0.97) 
        
        self.features_names = ['speedX', 'trackPos', 'angle'] + [f'track_{i}' for i in range(19)]
        self.targets_names = ['steer', 'accel', 'brake', 'gear']
        
        self._addestra_modello(dataset_path)

    def _addestra_modello(self, dataset_path):
        print("\n[1] Caricamento del dataset...")
        try:
            df = pd.read_csv(dataset_path, comment='#')
        except FileNotFoundError:
            print(f"ERRORE: File {dataset_path} non trovato. Registra prima i dati!")
            exit()

        print(f"Trovati {len(df)} campioni di guida.")

        X = df[self.features_names].values
        y = df[self.targets_names].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("[2] Standardizzazione, PCA e addestramento del modello K-NN in corso...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Applichiamo la PCA sui dati già scalati
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"-Dimensioni originali dei sensori : {X_train.shape[1]}")
        print(f"-Dimensioni ridotte (dopo la PCA): {X_train_pca.shape[1]}")

        self.model.fit(X_train_pca, y_train)

        print("[3] Valutazione del modello (Mean Squared Error)...")
        y_pred = self.model.predict(X_test_pca)
        
        mse_steer = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mse_accel = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        print(f" - Errore Sterzo: {mse_steer:.4f} (Più vicino a 0 è meglio)")
        print(f" - Errore Accel : {mse_accel:.4f} (Più vicino a 0 è meglio)")
        print("\nMODELLO PRONTO! IN ATTESA DI TORCS...\n")

    def predici_azioni(self, sensors: dict) -> dict:
        """Prende i sensori in tempo reale da TORCS e restituisce le azioni predette dal KNN"""
        current_state = [
            sensors.get('speedX', 0.0),
            sensors.get('trackPos', 0.0),
            sensors.get('angle', 0.0)
        ] + list(sensors.get('track', [200.0] * 19))

        current_state_scaled = self.scaler.transform([current_state])
        
        # Trasformiamo i dati con la PCA prima di passarli al modello
        current_state_pca = self.pca.transform(current_state_scaled)

        pred = self.model.predict(current_state_pca)[0]

        return {
            'steer': max(-1.0, min(1.0, pred[0])),
            'accel': max(0.0, min(1.0, pred[1])),
            'brake': max(0.0, min(1.0, pred[2])),
            'gear' : int(round(pred[3]))
        }

# Esecuzione Guida Autonoma 
def main():
    merge_laps(LAPS_FOLDER, DATASET_FILE)

    ai_driver = PilotaKNN(DATASET_FILE)
    
    client = snakeoil3.Client(p=3001, vision=False)
    
    try:
        while True:
            client.get_servers_input()
            t0 = time.time()
            print("Guida Autonoma KNN")
            
            for step in range(MAX_STEPS):
                sensors = client.S.d
                
                if abs(sensors.get('trackPos', 0.0)) > 1.4:
                    print("L'AI è uscita di pista! Riavvio sessione...")
                    client.R.d['meta'] = 1
                    client.respond_to_server()
                    break

                actions = ai_driver.predici_azioni(sensors)

                client.R.d['steer']  = actions['steer']
                client.R.d['accel']  = actions['accel']
                client.R.d['brake']  = actions['brake']
                client.R.d['gear']   = actions['gear']
                client.R.d['clutch'] = 0.0
                client.R.d['meta']   = 0

                client.respond_to_server()
                client.get_servers_input()
                
            time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\nChiusura pilota automatico.")
    except Exception as e:
        print(f"\nDisconnesso da TORCS: {e}")

if __name__ == "__main__":
    main()
# Intelligenza Artificiale (Behavioral Cloning) tramite Rete Neurale (MLP)
import time
import glob
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import snakeoil as snakeoil3

# ── Configurazioni ─────────────────────────────────────────────────────────────
DATASET_FILE = "dataset.csv"
LAPS_FOLDER  = "Laps"

# ── Architettura della rete neurale ───────────────────────────────────────────
MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_MAX_ITER      = 500
MLP_LEARNING_RATE = 0.001

MAX_STEPS = 200_000

# ── Smoothing delle predizioni ─────────────────────────────────────────────────
# Quanto velocemente i comandi seguono la predizione grezza della rete.
# Valori più bassi = transizioni più morbide (0.0 = bloccato, 1.0 = istantaneo).
STEER_ALPHA = 0.4   # Lo sterzo deve reagire abbastanza velocemente in curva
ACCEL_ALPHA = 0.3   # Acceleratore più morbido evita pattinamento
BRAKE_ALPHA = 0.5   # Il freno deve essere reattivo

# ── Safety Layer ───────────────────────────────────────────────────────────────
# Quando il modello va in confusione, una correzione geometrica prende il sopravvento.
# Aumenta SAFETY_WEIGHT se l'AI continua a uscire; abbassalo se sovrasterza.
SAFETY_WEIGHT   = 0.45  # Peso del correttore geometrico (0=solo rete, 1=solo correttore)
TRACKPOS_GAIN   = 0.5   # Quanto trackPos contribuisce alla correzione sterzo
ANGLE_GAIN      = 0.8   # Quanto l'angolo di deriva contribuisce alla correzione sterzo
OFF_TRACK_RESET = 1.6   # |trackPos| oltre cui forza il reset


def merge_laps(laps_folder: str, output_path: str) -> None:
    """
    Unisce tutti i file lap_*.csv presenti in `laps_folder` in un unico
    dataset CSV, eliminando le intestazioni ridondanti dei singoli giri.
    Il file di output viene (ri)creato ogni volta che viene chiamata questa funzione.
    """
    pattern = os.path.join(laps_folder, "lap_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[MERGE] Nessun file lap_*.csv trovato in '{laps_folder}'. "
              "Assicurati di aver registrato almeno un giro.")
        return

    print(f"[MERGE] Trovati {len(files)} giri da unire:")
    for f in files:
        print(f"        - {f}")

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"[MERGE] Dataset salvato in '{output_path}' ({len(merged)} campioni totali).\n")


class PilotaMLP:
    def __init__(self, dataset_path: str):
        self.model = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation='relu',
            solver='adam',
            learning_rate_init=MLP_LEARNING_RATE,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        self.scaler = StandardScaler()

        self.features_names = ['speedX', 'trackPos', 'angle'] + [f'track_{i}' for i in range(19)]
        self.targets_names  = ['steer', 'accel', 'brake', 'gear']

        # Stato smoothed (aggiornato ad ogni step di guida)
        self._smooth = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0}

        self._addestra_modello(dataset_path)

    def _addestra_modello(self, dataset_path: str) -> None:
        print("\n[1] Caricamento del dataset...")
        try:
            df = pd.read_csv(dataset_path, comment='#')
        except FileNotFoundError:
            print(f"ERRORE: File {dataset_path} non trovato. Registra prima i dati!")
            exit()

        print(f"    Trovati {len(df)} campioni di guida.")

        X = df[self.features_names].values
        y = df[self.targets_names].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("[2] Standardizzazione e addestramento della rete neurale in corso...")
        print(f"    Architettura: input({len(self.features_names)}) → "
              + " → ".join(str(n) for n in MLP_HIDDEN_LAYERS)
              + f" → output({len(self.targets_names)})")
        print(f"    Epoche massime : {MLP_MAX_ITER}  |  Early stopping attiva")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        print(f"    Addestramento completato in {self.model.n_iter_} epoche.")

        print("[3] Valutazione del modello (Mean Squared Error)...")
        y_pred = self.model.predict(X_test_scaled)
        mse_steer = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mse_accel = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        print(f"    Errore Sterzo : {mse_steer:.4f}  (più vicino a 0 è meglio)")
        print(f"    Errore Accel  : {mse_accel:.4f}  (più vicino a 0 è meglio)")
        print("\n>>> MODELLO PRONTO! IN ATTESA DI TORCS... <<<\n")

    def _safety_steer(self, track_pos: float, angle: float) -> float:
        """
        Correttore geometrico puro: calcola lo sterzo ideale basandosi solo
        su trackPos (quanto siamo fuori centro) e angle (angolo di deriva).
        Non usa la rete — serve da ancora quando il modello è incerto.
        """
        return -TRACKPOS_GAIN * track_pos - ANGLE_GAIN * angle

    def predici_azioni(self, sensors: dict) -> dict:
        """
        Pipeline completa:
          1. Predizione grezza della rete neurale
          2. Blending con il correttore geometrico (safety layer)
          3. Smoothing EMA per evitare comandi nervosi
        """
        track_pos = sensors.get('trackPos', 0.0)
        angle     = sensors.get('angle',    0.0)

        current_state = [
            sensors.get('speedX', 0.0),
            track_pos,
            angle
        ] + list(sensors.get('track', [200.0] * 19))

        current_state_scaled = self.scaler.transform([current_state])
        pred = self.model.predict(current_state_scaled)[0]

        # 1. Predizione grezza della rete
        net_steer = float(pred[0])
        net_accel = float(pred[1])
        net_brake = float(pred[2])
        net_gear  = int(np.clip(round(pred[3]), 1, 6))  # sempre in [1, 6]

        # 2. Safety layer: blend tra rete e correttore geometrico
        safe_steer    = self._safety_steer(track_pos, angle)
        blended_steer = (1.0 - SAFETY_WEIGHT) * net_steer + SAFETY_WEIGHT * safe_steer

        # 3. Smoothing EMA (evita oscillazioni e sterzo nervoso)
        self._smooth['steer'] += STEER_ALPHA * (blended_steer - self._smooth['steer'])
        self._smooth['accel'] += ACCEL_ALPHA * (net_accel      - self._smooth['accel'])
        self._smooth['brake'] += BRAKE_ALPHA * (net_brake      - self._smooth['brake'])

        return {
            'steer': float(np.clip(self._smooth['steer'], -1.0, 1.0)),
            'accel': float(np.clip(self._smooth['accel'],  0.0, 1.0)),
            'brake': float(np.clip(self._smooth['brake'],  0.0, 1.0)),
            'gear' : net_gear
        }

    def reset_smooth(self) -> None:
        """Resetta lo stato EMA all'inizio di ogni giro."""
        self._smooth = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0}


# ── Esecuzione Guida Autonoma ──────────────────────────────────────────────────

def main():
    merge_laps(LAPS_FOLDER, DATASET_FILE)

    ai_driver = PilotaMLP(DATASET_FILE)

    client = snakeoil3.Client(p=3001, vision=False)

    try:
        while True:
            client.get_servers_input()
            ai_driver.reset_smooth()
            print("=== Gara Iniziata (Guida Autonoma MLP) ===")

            for step in range(MAX_STEPS):
                sensors = client.S.d

                if abs(sensors.get('trackPos', 0.0)) > OFF_TRACK_RESET:
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
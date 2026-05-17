# =============================================================================
# pilotaBC.py — Pilota Autonomo con Behavioral Cloning (BC)
# =============================================================================

import time
import glob
import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import snakeoil as snakeoil3

# =============================================================================
# Configurazione
# =============================================================================

DATASET_FILE = "dataset.csv"
LAPS_FOLDER  = "csvTest"

EPOCHS            = 100
BATCH_SIZE        = 256
LEARNING_RATE     = 1e-3
CHECKPOINT_EVERY  = 50
LOAD_CHECKPOINT   = "checkpoints/bc_epoch100.pt"

FEATURE_NAMES = (
    ['speedX', 'speedY', 'trackPos', 'angle']
    + [f'track_{i}' for i in range(19)]
)
TARGET_NAMES = ['steer', 'accel', 'brake']

UPSHIFT_RPM    = {1: 9000, 2: 9000, 3: 9000, 4: 11000, 5: 14000}
DOWNSHIFT_RPM  = {2: 6000, 3: 8000, 4: 9000, 5: 10000, 6: 12000}
SHIFT_COOLDOWN = 0.5


# =============================================================================
# Unione dei giri CSV
# =============================================================================

def merge_laps(laps_folder: str, output_path: str) -> None:
    pattern = os.path.join(laps_folder, "lap_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERRORE] Nessun file lap_*.csv trovato in '{laps_folder}'.")
        print("         Registra almeno un giro con manualControlAutoShift.py prima.")
        return

    print(f"[1] Trovati {len(files)} giri da unire:")
    for f in files:
        print(f"      - {f}")

    dfs    = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"    Dataset salvato in '{output_path}' ({len(merged)} campioni totali).\n")


# =============================================================================
# Architettura MLP
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron per Behavioral Cloning.
    Input: 23 feature (speedX, speedY, trackPos, angle, track_0..18)
    Output: [steer (tanh), accel (sigmoid), brake (sigmoid)]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.head_steer = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.head_ab    = nn.Sequential(nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        steer    = self.head_steer(features)
        ab       = self.head_ab(features)
        return torch.cat([steer, ab], dim=1)


# =============================================================================
# Classe principale: Pilota BC
# =============================================================================

class PilotaBC:

    def __init__(self, dataset_path: str):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model  = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Dispositivo di calcolo: {self.device}")

        self._gear            = 1
        self._last_shift_time = time.time()

        self._addestra(dataset_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _addestra(self, dataset_path: str) -> None:

        # Carica sempre il dataset per fittare lo scaler
        print("[2] Caricamento dataset e fit scaler...")
        try:
            df = pd.read_csv(dataset_path, comment='#')
        except FileNotFoundError:
            print(f"[ERRORE] File '{dataset_path}' non trovato.")
            exit(1)

        missing = [c for c in FEATURE_NAMES + TARGET_NAMES if c not in df.columns]
        if missing:
            print(f"[ERRORE] Colonne mancanti nel dataset: {missing}")
            exit(1)

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df[TARGET_NAMES].values.astype(np.float32)

        # fit_transform sempre, checkpoint o no
        X = self.scaler.fit_transform(X).astype(np.float32)

        # Carica checkpoint se specificato 
        if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
            self.model = MLP(input_dim=len(FEATURE_NAMES)).to(self.device)
            self.model.load_state_dict(
                torch.load(LOAD_CHECKPOINT, map_location=self.device)
            )
            print(f"    Checkpoint caricato: {LOAD_CHECKPOINT}")
            print("\n    MODELLO PRONTO! IN ATTESA DI TORCS...\n")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_tr = torch.tensor(X_train); y_tr = torch.tensor(y_train)
        X_te = torch.tensor(X_test);  y_te = torch.tensor(y_test)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.model = MLP(input_dim=X_train.shape[1]).to(self.device)
        criterion  = nn.MSELoss()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        os.makedirs("checkpoints", exist_ok=True)

        print(f"[3] Training MLP per {EPOCHS} epoche (batch={BATCH_SIZE}, lr={LEARNING_RATE})...")
        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            epoch_loss /= len(X_train)

            if epoch % CHECKPOINT_EVERY == 0:
                ckpt_path = f"checkpoints/bc_epoch{epoch}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"    [Epoch {epoch:>3}/{EPOCHS}] Loss: {epoch_loss:.5f}  → checkpoint salvato: {ckpt_path}")
            elif epoch % 10 == 0:
                print(f"    [Epoch {epoch:>3}/{EPOCHS}] Loss train: {epoch_loss:.5f}")

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_te.to(self.device)).cpu().numpy()

        mse_steer = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mse_accel = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        mse_brake = mean_squared_error(y_test[:, 2], y_pred[:, 2])
        print(f"\n[4] Valutazione su test set:")
        print(f"    Sterzo : {mse_steer:.5f}")
        print(f"    Accel  : {mse_accel:.5f}")
        print(f"    Freno  : {mse_brake:.5f}")
        print("\n    MODELLO PRONTO! IN ATTESA DI TORCS...\n")

    # ------------------------------------------------------------------
    # Cambio automatico
    # ------------------------------------------------------------------

    def _aggiorna_marcia(self, rpm: float, speed: float) -> None:
        now = time.time()
        if speed < 10.0 and self._gear > 1:
            self._gear = 1
            self._last_shift_time = now
            return
        if now - self._last_shift_time < SHIFT_COOLDOWN:
            return
        if self._gear < 6 and rpm > UPSHIFT_RPM.get(self._gear, 18700):
            self._gear += 1
            self._last_shift_time = now
        elif self._gear > 1 and rpm < DOWNSHIFT_RPM.get(self._gear, 0):
            self._gear -= 1
            self._last_shift_time = now

    # ------------------------------------------------------------------
    # Predizione in tempo reale
    # ------------------------------------------------------------------

    def predici_azioni(self, sensors: dict) -> dict:
        track = sensors.get('track', [200.0] * 19)

        stato = np.array(
            [sensors.get('speedX',   0.0),
            sensors.get('speedY',   0.0),
            sensors.get('trackPos', 0.0),
            sensors.get('angle',    0.0)]
            + list(track),
            dtype=np.float32
        ).reshape(1, -1)

        stato_norm = self.scaler.transform(stato).astype(np.float32)
        tensor_in  = torch.tensor(stato_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor_in).cpu().numpy()[0]

        self._aggiorna_marcia(sensors.get('rpm', 0.0), sensors.get('speedX', 0.0))

        return {
            'steer': float(np.clip(pred[0], -1.0, 1.0)),
            'accel': float(np.clip(pred[1],  0.0, 1.0)),
            'brake': float(np.clip(pred[2],  0.0, 1.0)),
            'gear' : self._gear
        }


# =============================================================================
# Loop di guida autonoma
# =============================================================================

def main():
    merge_laps(LAPS_FOLDER, DATASET_FILE)
    pilota = PilotaBC(DATASET_FILE)
    client = snakeoil3.Client(p=3001, vision=False)

    try:
        while True:
            client.get_servers_input()
            print("═" * 55)
            print("  GUIDA AUTONOMA — Behavioral Cloning attivo")
            print("  CTRL+C per uscire")
            print("═" * 55)

            step = 0
            while True:
                sensors   = client.S.d
                track_pos = sensors.get('trackPos', 0.0)
                speed     = sensors.get('speedX',   0.0)

                if abs(track_pos) > 1.4:
                    print(f"\n  [!] Uscita di pista (trackPos={track_pos:.2f}) — reset sessione.")
                    client.R.d['meta'] = 1
                    client.respond_to_server()
                    time.sleep(2.0)
                    break

                actions = pilota.predici_azioni(sensors)

                client.R.d['steer']  = actions['steer']
                client.R.d['accel']  = actions['accel']
                client.R.d['brake']  = actions['brake']
                client.R.d['gear']   = actions['gear']
                client.R.d['clutch'] = 0.0
                client.R.d['meta']   = 0
                client.respond_to_server()

                if step % 50 == 0:
                    print(
                        f"  Marcia {actions['gear']}"
                        f" | {speed:.1f} km/h"
                        f" | steer {actions['steer']:+.3f}"
                        f" | accel {actions['accel']:.2f}"
                        f" | brake {actions['brake']:.2f}"
                        f" | trackPos {track_pos:+.3f}",
                        end="\r"
                    )

                client.get_servers_input()
                step += 1

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n  Sessione interrotta dall'utente.")
    except Exception as e:
        print(f"\n  Disconnesso da TORCS: {e}")


if __name__ == "__main__":
    main()
import time
import os
import snakeoil as snakeoil3
from pynput.keyboard import Key, Listener

#  Raccolta dati 
DATASET_DIR       = "Laps"   # Cartella di output dei CSV
LOG_EVERY_N_STEPS = 3
TRACK_LIMIT_RESET = 1.2
DAMAGE_THRESHOLD  = 20       # Danno accumulato (delta) oltre cui il giro viene scartato

#  Mappatura Marce Automatiche 
UPSHIFT_RPM = {
    1: 5500,   # Cambia in 2a a 5500 giri
    2: 7500,   
    3: 8500,   
    4: 9500,   
    5: 10500   
}

DOWNSHIFT_RPM = {
    2: 3500,   # Scala in 1a solo se scendi sotto i 3500 giri
    3: 4500,  
    4: 5500,   
    5: 6500,   
    6: 7500   
}

SHIFT_COOLDOWN = 0.5 

#  Elettronica di Sicurezza 
SOFT_REV_LIMIT_RPM = 13200    
HARD_REV_LIMIT_RPM = 13800    
TCS_SLIP_THRESHOLD = 3.0     

#  Sterzo 
STEER_INPUT_STEP = 1.1    # alzarne il valore rende il volante più sensibile 
MIN_STEER_FACTOR = 0.5    # alzarne il valore rende le curve più aggressive ad alta velocità
STEER_SMOOTH     = 0.16   # velocità con cui le ruote della machina girano
STEER_CENTERING  = 0.14   # velocità con cui viene riportato il volante nella posizione centrale
SPEED_STEER_DAMP = 185    # irrigidisce lo sterzo all'aumentare della velocità 

#  Acceleratore / Freno 
ACCEL_SMOOTH = 0.55  
BRAKE_SMOOTH = 0.50   


#  Controller 
class ManualController:

    def __init__(self):
        self._keys_pressed: set = set()
        self.state = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 1}
        
        self._last_shift_time = time.time()

        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def _on_press(self, key):   
        self._keys_pressed.add(key)

    def _on_release(self, key): 
        self._keys_pressed.discard(key)

    def _update_gear_auto(self, rpm: float, speed: float) -> None:
        now = time.time()
        
        if speed < 10.0 and self.state['gear'] > 1:
            self.state['gear'] = 1
            self._last_shift_time = now
            return

        if now - self._last_shift_time < SHIFT_COOLDOWN:
            return

        # Adatta gli RPM in base alla velocità (Modifica implementata)
        if speed < 20.0:
            UPSHIFT_RPM_MODIF = {k: v * 0.8 for k, v in UPSHIFT_RPM.items()}
        else:
            UPSHIFT_RPM_MODIF = UPSHIFT_RPM

        gear = self.state['gear']

        # Utilizza UPSHIFT_RPM_MODIF invece del dizionario base
        if gear < 6 and rpm > UPSHIFT_RPM_MODIF.get(gear, 13000):
            self.state['gear'] += 1
            self._last_shift_time = now
            
        elif gear > 1 and rpm < DOWNSHIFT_RPM.get(gear, 0):
            self.state['gear'] -= 1
            self._last_shift_time = now

    def update(self, sensors: dict) -> None:
        speed = sensors.get('speedX', 0.0)
        angle = sensors.get('angle',  0.0)
        rpm   = sensors.get('rpm', 0.0)

        # Gestione Cambio Automatico
        self._update_gear_auto(rpm, speed)

        # Acceleratore
        target_accel = 1.0 if Key.up in self._keys_pressed else 0.0
        self.state['accel'] += (target_accel - self.state['accel']) * ACCEL_SMOOTH
        self.state['accel']  = max(0.0, min(1.0, self.state['accel']))

        # Soft Rev Limiter
        if rpm > SOFT_REV_LIMIT_RPM:
            over_rev = rpm - SOFT_REV_LIMIT_RPM
            range_rev = HARD_REV_LIMIT_RPM - SOFT_REV_LIMIT_RPM
            reduction = min(0.7, over_rev / range_rev)
            self.state['accel'] *= (1.0 - reduction)

        # TCS (Permette più accelerazione in 1a marcia)
        wheel = sensors.get('wheelSpinVel', [0.0] * 4)
        if len(wheel) == 4:
            rear_spin = (wheel[2] + wheel[3]) - (wheel[0] + wheel[1])
            if rear_spin > TCS_SLIP_THRESHOLD:
                slip_penalty = min(0.5, (rear_spin - TCS_SLIP_THRESHOLD) * 0.1)
                self.state['accel'] *= (1.0 - slip_penalty)

        # Freno
        target_brake = 1.0 if Key.down in self._keys_pressed else 0.0
        self.state['brake'] += (target_brake - self.state['brake']) * BRAKE_SMOOTH
        self.state['brake']  = max(0.0, min(1.0, self.state['brake']))

        # Sterzo
        raw_steer = 0.0
        if Key.left  in self._keys_pressed: raw_steer += STEER_INPUT_STEP
        if Key.right in self._keys_pressed: raw_steer -= STEER_INPUT_STEP

        # Damping meno aggressivo per le alte velocità
        speed_factor = max(MIN_STEER_FACTOR, 1.0 - abs(speed) / SPEED_STEER_DAMP)
        raw_steer   *= speed_factor

        if abs(raw_steer) < 0.01:
            steer_target = -angle * STEER_CENTERING 
        else:
            steer_target = raw_steer - angle * 0.3

        self.state['steer'] += (steer_target - self.state['steer']) * STEER_SMOOTH
        
        if abs(self.state['steer']) < 0.015:
            self.state['steer'] = 0.0
            
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))

    def stop(self):
        self._listener.stop()


# Savataggio file in CSV

def _build_csv_header() -> str:
    track_cols = ",".join(f"track_{i}" for i in range(19))
    return f"timestamp,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage,{track_cols}\n"

def _build_csv_row(timestamp: float, actions: dict, sensors: dict) -> str:
    track = sensors.get('track', [0.0] * 19)
    return (
        f"{timestamp:.3f},"
        f"{actions['steer']:.5f},{actions['accel']:.5f},"
        f"{actions['brake']:.5f},{int(actions['gear'])},"
        f"{sensors.get('speedX',   0.0):.4f},"
        f"{sensors.get('trackPos', 0.0):.5f},"
        f"{sensors.get('angle',    0.0):.5f},"
        f"{sensors.get('rpm',      0.0):.1f},"
        f"{sensors.get('damage',   0.0):.1f},"
        + ",".join(f"{v:.4f}" for v in track) + "\n"
    )

def save_lap(lap_buffer_csv: list, lap_time: float) -> None:
    """
    Salva un giro pulito su disco come CSV.

    Il nome del file include il tempo ufficiale del giro (es. lap_87.43.csv).
    Se lap_time <= 0 il file viene marcato come 'partial' (salvataggio da Ctrl+C).
    Se un file con lo stesso nome esiste già, aggiunge un contatore per non sovrascriverlo.

    Parametri:
        lap_buffer_csv : lista di righe CSV accumulate durante il giro
        lap_time       : tempo ufficiale del giro da TORCS (lastLapTime)
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    time_str = f"{lap_time:.2f}" if lap_time > 0 else "partial"

    # Costruiamo il nome base e il percorso iniziale
    base_name = f"lap_{time_str}"
    csv_path  = os.path.join(DATASET_DIR, f"{base_name}.csv")

    # Se il file esiste già, incrementiamo un contatore finché non troviamo un nome libero
    counter = 1
    while os.path.exists(csv_path):
        csv_path = os.path.join(DATASET_DIR, f"{base_name}_{counter}.csv")
        counter += 1

    # Salviamo il file
    with open(csv_path, "w") as f:
        f.write(_build_csv_header())
        f.writelines(lap_buffer_csv)

    # Stampa di conferma aggiornata per gestire anche i giri "parziali"
    label_tempo = f"{lap_time:.2f}s" if lap_time > 0 else "Parziale"
    print(f"\n  ✓ Giro {label_tempo} → '{csv_path}' ({len(lap_buffer_csv)} campioni)")


# Avvio del giro
def run_lap(controller: ManualController, episode: int, t0: float) -> int:
    """
    Esegue un singolo giro su TORCS raccogliendo i dati in un buffer in memoria.

    Il CSV viene scritto su disco SOLO se il giro è valido:
      - Nessuna uscita di pista (|trackPos| <= TRACK_LIMIT_RESET)
      - Nessun danno accumulato oltre DAMAGE_THRESHOLD

    La fine del giro viene rilevata tramite 'lastLapTime' di TORCS.
    """
    buf_csv = []
    is_lap_valid = True
    last_damage = 0.0
    last_lap_time = 0.0
    step = 0

    client = snakeoil3.Client(p=3001, vision=False)

    # Prima lettura 
    client.get_servers_input()
    last_damage   = client.S.d.get('damage',      0.0)
    last_lap_time = client.S.d.get('lastLapTime', 0.0)

    try:
        while True:

            # LEGGI i sensori aggiornati
            client.get_servers_input()
            sensors   = client.S.d
            track_pos = sensors.get('trackPos', 0.0)
            speed     = sensors.get('speedX',   0.0)
            damage    = sensors.get('damage',   0.0)

            # ELABORA: validità giro
            if damage > last_damage:
                if is_lap_valid:
                    print(f"Giro invalidato (danno) — dati scartati.\n")
                    is_lap_valid = False

            if abs(track_pos) > TRACK_LIMIT_RESET:
                print(f"Fuori pista ({track_pos:.2f}) — reset in corso.\n")
                buf_csv       = []
                is_lap_valid  = True
                last_lap_time = 0.0
                client.R.d['meta'] = 1
                client.respond_to_server()
                break

            # ELABORA: aggiorna controller e prepara comandi
            controller.update(sensors)
            actions = controller.state

            # INVIA i comandi
            client.R.d.update({
                'steer': actions['steer'], 'accel': actions['accel'],
                'brake': actions['brake'], 'gear':  actions['gear'],
                'clutch': 0.0, 'meta': 0
            })
            client.respond_to_server()

            # ACCUMULA nel buffer 
            if step % LOG_EVERY_N_STEPS == 0 and speed > 5:
                buf_csv.append(_build_csv_row(time.time() - t0, actions, sensors))

                if len(buf_csv) % 50 == 0:
                    status = "✓" if is_lap_valid else "✗ INVALIDO"
                    print(
                        f"  [{status} | Campioni: {len(buf_csv):>5}]"
                        f" | Marcia {actions['gear']}"
                        f" | RPM: {sensors.get('rpm', 0.0):.0f}"
                        f" | {speed:.1f} km/h",
                        end="\r"
                    )

            # RILEVA fine giro tramite lastLapTime
            current_lap_time = sensors.get('lastLapTime', 0.0)

            if current_lap_time > 0 and current_lap_time != last_lap_time:
                if is_lap_valid and len(buf_csv) > 0:
                    save_lap(buf_csv, current_lap_time)
                else:
                    print(f"\n  ✗ Giro concluso ma scartato — nessun file salvato.")

                buf_csv       = []
                is_lap_valid  = True
                last_lap_time = current_lap_time

            last_damage = damage
            step       += 1

    except Exception as e:
        print(f"Connessione terminata: {e}\n")

    # Salvataggio residuo
    if is_lap_valid and len(buf_csv) > 0:
        save_lap(buf_csv, last_lap_time if last_lap_time > 0 else -1)
        return len(buf_csv)

    return 0


# Main 
def main():
    controller = ManualController()
    t0 = time.time()
    lap = 0
    total_rows = 0

    print("=" * 60)
    print("  ↑ ↓ ← → : Guida    |    CAMBIO : Automatico Mappato")
    print("  CTRL+C  : Esci     |    Fuori pista = Auto Reset")
    print("=" * 60)
    print(f"  Registrazione attiva — giri puliti salvati in '{DATASET_DIR}/'")

    try:
        while True:
            lap += 1
            print(f"\n  ── Giro {lap} ──")
            try:
                total_rows += run_lap(controller, lap, t0)
                time.sleep(1.0)
            except Exception as e:
                print(f"  In attesa di TORCS... ({e})")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n  Sessione interrotta.")

    finally:
        controller.stop()
        print(f"\n  Fine. Campioni totali salvati: {total_rows}")


if __name__ == "__main__":
    main()
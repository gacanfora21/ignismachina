import time
import os
import snakeoil as snakeoil3
import pygame # libreria controller

# Raccolta dati 
DATASET_DIR       = "Laps"   # Cartella di output dei CSV
LOG_EVERY_N_STEPS = 3        # Salva un campione ogni N step
TRACK_LIMIT_RESET = 1.3      # |trackPos| oltre cui il giro viene invalidato
DAMAGE_THRESHOLD  = 20       # Danno accumulato (delta) oltre cui il giro viene scartato

# Mappatura Marce Automatiche 
# RPM soglia per scalare la marcia su
UPSHIFT_RPM = {
    1: 5500,
    2: 7500,
    3: 8500,
    4: 9500,
    5: 10500
}

# RPM soglia per scalare la marcia giù
DOWNSHIFT_RPM = {
    2: 3500,
    3: 4500,
    4: 5500,
    5: 6500,
    6: 7500
}

SHIFT_COOLDOWN = 0.5  # Secondi minimi tra un cambio marcia e il successivo

#  Elettronica di Sicurezza 
SOFT_REV_LIMIT_RPM = 13200   # RPM oltre cui inizia a ridurre l'acceleratore
HARD_REV_LIMIT_RPM = 13800   # RPM oltre cui taglia completamente l'acceleratore
TCS_SLIP_THRESHOLD = 3.0     # Soglia di slittamento per il controllo trazione

#  Sterzo 
STEER_INPUT_STEP = 1     # Sensibilità del volante (valori più alti = più reattivo)
MIN_STEER_FACTOR = 0.45  # Angolo minimo applicato in curva ad alta velocità
STEER_SMOOTH     = 0.15  # Velocità di rotazione delle ruote
STEER_CENTERING  = 0.10  # Velocità di ritorno del volante al centro
SPEED_STEER_DAMP = 180   # Fattore di smorzamento sterzo all'aumentare della velocità

#  Acceleratore / Freno 
ACCEL_SMOOTH = 0.55  # Interpolazione apertura acceleratore (0=istantaneo, 1=lentissimo)
BRAKE_SMOOTH = 0.50  # Interpolazione pressione freno


class JoystickController:
    """
    Gestisce l'input da controller fisico (Xbox/PS4) tramite Pygame.
    Traduce gli assi analogici in comandi TORCS (steer, accel, brake)
    e calcola automaticamente la marcia corretta in base a RPM e velocità.
    """

    def __init__(self):
        """Inizializza Pygame, rileva il primo joystick collegato e imposta lo stato iniziale."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("ERRORE: Nessun joystick rilevato! Collegalo e riavvia.")
            exit()

        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        print(f"\n[+] Joystick Connesso: {self.joy.get_name()}\n")

        # Stato corrente dei comandi inviati a TORCS
        self.state = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 1}
        self._last_shift_time = time.time()  # Timestamp dell'ultimo cambio marcia

    def _update_gear_auto(self, rpm: float, speed: float) -> None:
        """
        Aggiorna la marcia automaticamente in base a RPM e velocità.

        - Forza la prima marcia sotto i 10 km/h
        - Scala verso l'alto quando si superano le soglie UPSHIFT_RPM
        - Scala verso il basso quando si scende sotto DOWNSHIFT_RPM
        - Riduce le soglie di upshift del 20% a bassa velocità (<20 km/h)
        - Rispetta il cooldown tra due cambi consecutivi
        """
        now = time.time()

        if speed < 10.0 and self.state['gear'] > 1:
            self.state['gear'] = 1
            self._last_shift_time = now
            return

        if now - self._last_shift_time < SHIFT_COOLDOWN:
            return

        # A bassa velocità abbassa le soglie di upshift per evitare strappi
        UPSHIFT_RPM_MODIF = (
            {k: v * 0.8 for k, v in UPSHIFT_RPM.items()}
            if speed < 20.0 else UPSHIFT_RPM
        )

        gear = self.state['gear']

        if gear < 6 and rpm > UPSHIFT_RPM_MODIF.get(gear, 13000):
            self.state['gear'] += 1
            self._last_shift_time = now
        elif gear > 1 and rpm < DOWNSHIFT_RPM.get(gear, 0):
            self.state['gear'] -= 1
            self._last_shift_time = now

    def update(self, sensors: dict) -> None:
        """
        Legge gli assi del controller e aggiorna self.state.

        Assi usati:
          - Asse 0: Levetta joestick sinistra = sterzo 
          - Asse 5: RT = acceleratore 
          - Asse 4: LT = freno      

        """
        pygame.event.clear()
        pygame.event.pump()  # Aggiorna lo stato interno di Pygame

        speed = sensors.get('speedX', 0.0)
        rpm   = sensors.get('rpm',    0.0)

        self._update_gear_auto(rpm, speed)

        steer_axis = -self.joy.get_axis(0) 

        if abs(steer_axis) < 0.05:  # deadzone del ±0.05
            steer_axis = 0.0
        self.state['steer'] = steer_axis

        accel_mapped = (self.joy.get_axis(5) + 1.0) / 2.0
        brake_mapped = (self.joy.get_axis(4) + 1.0) / 2.0

        if accel_mapped < 0.05: accel_mapped = 0.0
        if brake_mapped < 0.05: brake_mapped = 0.0

        self.state['accel'] = accel_mapped
        self.state['brake'] = brake_mapped

    def stop(self):
        """Chiude Pygame e rilascia il joystick."""
        pygame.quit()


# ── CSV  ────────────────────────────────────────────────────────────────

def _build_csv_header() -> str:
    """Restituisce la riga di intestazione del CSV."""
    track_cols = ",".join(f"track_{i}" for i in range(19))
    return f"time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage,{track_cols}\n"

def _build_csv_row(timestamp: float, actions: dict, sensors: dict) -> str:
    """
    Costruisce una riga CSV dal timestamp, dai comandi e dai sensori TORCS.

    Colonne: time | steer | accel | brake | gear |
             speedX | trackPos | angle | rpm | damage | track_0…track_18
    """
    track = sensors.get('track', [0.0] * 19)
    return (
        f"{timestamp:.3f},"
        f"{actions['steer']:.4f},{actions['accel']:.4f},"
        f"{actions['brake']:.4f},{int(actions['gear'])},"
        f"{sensors.get('speedX',   0.0):.2f},"
        f"{sensors.get('trackPos', 0.0):.4f},"
        f"{sensors.get('angle',    0.0):.4f},"
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
    csv_path = os.path.join(DATASET_DIR, f"{base_name}.csv")
    
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


# Aviva giro

def run_lap(controller: JoystickController, episode: int, t0: float) -> int:
    """
    Esegue un singolo giro su TORCS raccogliendo i dati in un buffer in memoria.

    Ciclo fondamentale: get_servers_input → elabora → respond_to_server.
    Invertire questo ordine causa un ritardo cumulativo sui comandi.

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
    last_damage = client.S.d.get('damage', 0.0)
    last_lap_time = client.S.d.get('lastLapTime', 0.0)

    try:
        while True:

            # LEGGI i sensori aggiornati
            client.get_servers_input()
            sensors   = client.S.d
            track_pos = sensors.get('trackPos', 0.0)
            speed     = sensors.get('speedX',   0.0)
            damage    = sensors.get('damage',   0.0)

            # ELABORA: validità giro ────────────────────────────────────────
            if damage > last_damage:
                if is_lap_valid:
                    print(f"\n Giro invalidato (danno) — dati scartati.")
                    is_lap_valid = False

            if abs(track_pos) > TRACK_LIMIT_RESET:
                print(f"\n Fuori pista ({track_pos:.2f}) — reset in corso.")
                buf_csv       = []
                is_lap_valid  = True
                last_lap_time = 0.0
                client.R.d['meta'] = 1
                client.respond_to_server()
                break

            # ELABORA: aggiorna controller e prepara comandi ────────────────
            controller.update(sensors)
            actions = controller.state

            # INVIA i comandi
            client.R.d.update({
                'steer':  actions['steer'],
                'accel':  actions['accel'],
                'brake':  actions['brake'],
                'gear':   actions['gear'],
                'clutch': 0.0,
                'meta':   0
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
        print(f"\n  Connessione terminata: {e}")

    # Salvataggio residuo
    if is_lap_valid and len(buf_csv) > 0:
        save_lap(buf_csv, last_lap_time if last_lap_time > 0 else -1)
        return len(buf_csv)

    return 0


# Main

def main():
    """
    Entry point. Gestisce il loop principale dei giri:
    inizializza il controller, esegue episodi consecutivi
    e stampa il totale dei campioni raccolti alla fine.
    Alla pressione di Ctrl+C salva il buffer parziale se il giro era valido
    e conteneva più di 100 campioni.
    """
    controller = JoystickController()
    t0 = time.time()
    lap = 0
    total_rows = 0

    print(f"Registrazione attiva — giri puliti salvati in '{DATASET_DIR}/'")

    try:
        while True:
            lap += 1
            try:
                total_rows += run_lap(controller, lap, t0)
                time.sleep(1.0)
            except Exception as e:
                print(f"In attesa di TORCS... ({e})")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nSessione interrotta.")

    finally:
        controller.stop()
        print(f"\nFine. Campioni totali salvati: {total_rows}")


if __name__ == "__main__":
    main()
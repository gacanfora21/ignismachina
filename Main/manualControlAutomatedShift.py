from pynput.keyboard import Key, Listener
import snakeoil as snakeoil3
import time
import os


class ArcadeController:
    def __init__(self):
        self.keys = set()

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }
        
        self.last_shift_time = time.time()
        # Ridotto a 0.3 per permettere scalate più veloci durante le frenate brusche
        self.shift_cooldown = 0.3 

        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    def press(self, key):
        self.keys.add(key)

    def release(self, key):
        self.keys.discard(key)

    def update(self, sensors):
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)
        rpm = sensors.get('rpm', 0)

        # ========================
        # CAMBIO AUTOMATICO OTTIMIZZATO
        # ========================
        UP_SHIFT_RPM = 8500
        DOWN_SHIFT_RPM = 5500  # Alzato per scalare prima ed evitare stalli del motore

        if self.state['gear'] < 1:
            self.state['gear'] = 1

        current_time = time.time()
        
        # CONTROLLO DI SICUREZZA: Se andiamo pianissimo (< 15 km/h), mettiamo subito in prima.
        if speed < 15.0 and self.state['gear'] > 1:
            self.state['gear'] = 1
            self.last_shift_time = current_time

        # LOGICA DI CAMBIO NORMALE
        elif current_time - self.last_shift_time > self.shift_cooldown:
            if self.state['gear'] < 6 and rpm > UP_SHIFT_RPM:
                self.state['gear'] += 1
                self.last_shift_time = current_time
            elif self.state['gear'] > 1 and rpm < DOWN_SHIFT_RPM:
                self.state['gear'] -= 1
                self.last_shift_time = current_time

        # ========================
        # ACCELERAZIONE REATTIVA
        # ========================
        target_accel = 1.0 if Key.up in self.keys else 0.0
        self.state['accel'] += (target_accel - self.state['accel']) * 0.3

        # ========================
        # FRENO REATTIVO
        # ========================
        target_brake = 1.0 if Key.down in self.keys else 0.0
        self.state['brake'] += (target_brake - self.state['brake']) * 0.6

        # ========================
        # STERZO REATTIVO
        # ========================
        steer_input = 0.0
        if Key.left in self.keys:
            steer_input += 0.8
        if Key.right in self.keys:
            steer_input -= 0.8

        # LIMITE VELOCITÀ SULLO STERZO
        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_input *= max_steer

        # SE NON STAI STERZANDO → VAI DRITTO
        if abs(steer_input) < 0.01:
            steer_target = 0.0
        else:
            stability = angle * 0.3
            steer_target = steer_input - stability

        # SMOOTH VELOCE
        self.state['steer'] += (steer_target - self.state['steer']) * 0.6

        # DEAD ZONE
        if abs(self.state['steer']) < 0.02:
            self.state['steer'] = 0.0

        # CLAMP (limita i valori per sicurezza)
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))
        self.state['gear'] = max(-1, min(6, self.state['gear']))
# ============================================================
# MAIN
# ============================================================

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()

    client.get_servers_input()

    print("========================================")
    print("🚗 Arcade driving mode attivo (No-Lag) 🚗")
    print("Frecce direzionali: Guida")
    print("Tasti W / S: Marcia Su / Marcia Giù")
    print("Premi CTRL+C nel terminale per salvare e uscire.")
    print("========================================")

    file_name = "manual.csv"
    
    # Controllo per l'Append
    csv_exists = os.path.isfile(file_name)
    log_csv = open(file_name, "a")
    
    # Intestazione solo se il file è nuovo
    if not csv_exists:
        track_headers = ",".join([f"track_{i}" for i in range(19)])
        log_csv.write(f"time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage,{track_headers}\n")

    t0 = time.time()
    step = 0
    LOG_STEP = 20 # Salva nel CSV circa 2.5 volte al secondo (a 50Hz)

    try:
        while True:
            S = client.S.d

            # 1. Aggiorna l'input del controller
            controller.update(S)
            a = controller.state

            # 2. Prepara la risposta per il server
            client.R.d['steer'] = a['steer']
            client.R.d['accel'] = a['accel']
            client.R.d['brake'] = a['brake']
            client.R.d['gear'] = a['gear']
            client.R.d['clutch'] = 0.0
            client.R.d['meta'] = 0

            # 3. Invia i comandi a TORCS
            client.respond_to_server()
            
            # 4. Aspetta i nuovi sensori dal server (Sostituisce il time.sleep!)
            client.get_servers_input()

            # 5. Logging dei dati sul CSV
            if step % LOG_STEP == 0:
                current_time = time.time() - t0
                track_sensors = S.get('track', [0.0] * 19)
                track_str = ",".join([str(x) for x in track_sensors])

                log_csv.write(
                    f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
                    f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
                    f"{S.get('rpm',0)},{S.get('damage',0)},{track_str}\n"
                )

            step += 1

    except KeyboardInterrupt:
        print("\nGuida interrotta dall'utente. Salvataggio dei log in corso...")
    finally:
        log_csv.close()
        print(f"Log salvati con successo nel file '{file_name}'. Uscita.")

if __name__ == "__main__":
    main()
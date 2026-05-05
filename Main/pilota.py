import math

# Definisco i parametri
TARGET_SPEED = 250       # Velocità target in km/h
STEER_GAIN = 30          # Sensibilità dello sterzo
CENTERING_GAIN = 0.20    # Forza di correzione verso il centro pista
BRAKE_THRESHOLD = 0.9    # Soglia d'angolo per la frenata
GEAR_SPEEDS = [0, 20, 40, 80, 100, 180] # Soglie per il cambio marcia
ENABLE_TRACTION_CONTROL = True



def calculate_steering(S):
    # Combina angolo e posizione laterale per mantenere l'auto centrata
    steer = (S['angle'] * STEER_GAIN / math.pi) - (S['trackPos'] * CENTERING_GAIN)
    return max(-1, min(1, steer))

def calculate_throttle(S, R):
    # Controllo velocità: accelera se sotto il target, altrimenti rallenta
    if S['speedX'] < TARGET_SPEED - (R['steer'] * 2.5):
        accel = min(1.0, R['accel'] + 0.4)
    else:
        accel = max(0.0, R['accel'] - 0.2)
    
    if S['speedX'] < 10:
        accel += 1 / (S['speedX'] + 0.1)
    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    # Frenata basata sulla curva del percorso
    return 0.3 if abs(S['angle']) > BRAKE_THRESHOLD else 0.0

def shift_gears(S):
    # Cambio automatico della marcia sulla base della velocità
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S['speedX'] > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    # Se le ruote slittano, riduco l'accelerazione
    if ENABLE_TRACTION_CONTROL:
        # Verifica slittamento tra ruote anteriori e posteriori
        if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - 
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])) > 2:
            accel -= 0.1
    return max(0.0, accel)



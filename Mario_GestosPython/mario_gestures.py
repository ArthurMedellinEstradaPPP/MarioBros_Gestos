# ----------------------------- Librerias ---------------------------------------------------

import cv2
import mediapipe as mp
import pyautogui
import time

# ------------- Controles ----------------

RIGHT = "d"
LEFT = "a"
UP = "w"
DOWN = "s"
JUMP = "space"
RUN = "shift"
START = "enter"
SELECT = "backspace"

# ------------- Estado de teclas ----------------

moving_right = False
moving_left = False
running = False
crouching = False
run_enabled  = True        # toggle: True = correr activado, False = desactivado
both_hands_prev = False    # estado de ambas manos en el frame anterior

last_jump = 0
jumping = False
jump_hold_start = 0
gesture_start   = 0          # cuando el usuario empezó a cerrar la mano
jump_hold_time  = 0.05       # se calcula dinámicamente en el loop
JUMP_COOLDOWN    = 0.5   # segundos entre saltos
JUMP_MIN_HOLD    = 0.05  # tiempo mínimo de hold (salto pequeño)
JUMP_MAX_HOLD    = 0.5   # tiempo máximo de hold (salto máximo)
JUMP_THRESHOLD   = 0.05  # distancia máxima para considerar dedos "juntos"
RUN_THRESHOLD    = 0.15  # distancia mínima para considerar mano "abierta"
CROUCH_THRESHOLD = 0.03  # distancia mínima para considerar "puño cerrado"

# --------------- Detección de gestos ---------------------------------

def detect_gestures(hand_landmarks):
    """Detecta salto, correr y agacharse a partir de los landmarks de la mano."""
    index_tip = hand_landmarks.landmark[8]   # punta del índice
    thumb_tip = hand_landmarks.landmark[4]   # punta del pulgar
    middle_tip = hand_landmarks.landmark[12] # punta del dedo medio
    wrist = hand_landmarks.landmark[0]       # muñeca

    # Distancia pulgar-índice (para gestos finos)
    pinch_distance = abs(index_tip.y - thumb_tip.y)

    # Apertura de la mano (medio vs muñeca, más robusto)
    hand_spread = abs(middle_tip.y - wrist.y)

    return {
        "jump":   pinch_distance <= JUMP_THRESHOLD,
        "run":    hand_spread > RUN_THRESHOLD,
        "crouch": pinch_distance <= CROUCH_THRESHOLD and hand_spread < 0.1,
    }

# --------------- Overlay de estado en pantalla -----------------------

def draw_status(frame, states: dict):
    """Dibuja en pantalla las teclas activas y la barra de carga del salto."""
    run_enabled = states.get("run_enabled", True)

    labels = {
        "RIGHT ▶":  states.get("right"),
        "LEFT  ◀":  states.get("left"),
        "JUMP  ↑":  states.get("jump"),
        "DOWN  ↓":  states.get("crouch"),
    }
    y = 30
    for text, active in labels.items():
        color = (0, 255, 0) if active else (80, 80, 80)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        y += 28

    # RUN con indicador de toggle
    if run_enabled:
        run_color = (0, 255, 0) if states.get("run") else (80, 80, 80)
        run_label = "RUN  💨 [ON] "
    else:
        run_color = (0, 100, 255)   # naranja = desactivado
        run_label = "RUN  💨 [OFF]"
    cv2.putText(frame, run_label, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, run_color, 2, cv2.LINE_AA)
    y += 28

    # Hint de ambas manos
    cv2.putText(frame, "[ ambas manos = toggle RUN ]", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    y += 20

    # Barra de carga del salto
    charge = states.get("jump_charge", 0.0)
    if charge > 0:
        bar_w = int(150 * charge)
        color = (0, int(255 * (1 - charge)), int(255 * charge))
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 14), color, -1)
        cv2.rectangle(frame, (10, y), (160, y + 14), (200, 200, 200), 1)
        cv2.putText(frame, "CARGA", (165, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)

# --------------- Codigo de MediaPipe ---------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ------------------------------- Camara ------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Iniciando control por gestos... (ESC para salir)")
time.sleep(2)

# ------------------------------ LOOP PRINCIPAL -----------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:          # FIX: manejar fallo de cámara
            print("Error: no se pudo leer el frame.")
            break

        frame = cv2.flip(frame, 1)

        # Redimensionar solo para MediaPipe (más rápido)
        small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        right_hand   = False
        left_hand    = False
        jump_detected   = False
        run_detected    = False
        crouch_detected = False

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                label = results.multi_handedness[i].classification[0].label

                # Dibujar landmarks en el frame original (tamaño completo)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gestures = detect_gestures(hand_landmarks)

                if gestures["jump"]:
                    jump_detected = True
                if gestures["run"]:
                    run_detected = True
                if gestures["crouch"]:
                    crouch_detected = True

                if label == "Right":
                    right_hand = True
                if label == "Left":
                    left_hand = True

        # ---- Toggle correr con ambas manos ----
        both_hands_now = right_hand and left_hand
        if both_hands_now and not both_hands_prev:
            # Flanco de subida: ambas manos aparecen juntas → toggle
            run_enabled = not run_enabled
            if not run_enabled and running:
                pyautogui.keyUp(RUN)
                running = False
        both_hands_prev = both_hands_now

        # ---- Mover derecha ----
        if right_hand and not moving_right:
            pyautogui.keyDown(RIGHT)
            moving_right = True
        elif not right_hand and moving_right:
            pyautogui.keyUp(RIGHT)
            moving_right = False

        # ---- Mover izquierda ----
        if left_hand and not moving_left:
            pyautogui.keyDown(LEFT)
            moving_left = True
        elif not left_hand and moving_left:
            pyautogui.keyUp(LEFT)
            moving_left = False

        # ---- Correr (solo si run_enabled) ----
        if run_enabled and run_detected and not running:
            pyautogui.keyDown(RUN)
            running = True
        elif (not run_detected or not run_enabled) and running:
            pyautogui.keyUp(RUN)
            running = False

        # ---- Agacharse ----
        if crouch_detected and not crouching:
            pyautogui.keyDown(DOWN)
            crouching = True
        elif not crouch_detected and crouching:
            pyautogui.keyUp(DOWN)
            crouching = False

        # ---- Salto dinámico: cuanto más tiempo mantienes el gesto, más alto salta ----
        now = time.time()

        # 1) Empezar a contar cuando se detecta el gesto (y no hay salto en curso)
        if jump_detected and not jumping and gesture_start == 0:
            if (now - last_jump) > JUMP_COOLDOWN:
                gesture_start = now

        # 2) Al soltar el gesto → disparar el salto con el hold acumulado
        if not jump_detected and gesture_start > 0 and not jumping:
            held      = now - gesture_start
            hold_time = max(min(held, JUMP_MAX_HOLD), JUMP_MIN_HOLD)
            pyautogui.keyDown(JUMP)
            jumping        = True
            jump_hold_start = now
            jump_hold_time  = hold_time   # guardamos cuánto mantener presionado
            last_jump      = now
            gesture_start  = 0

        # 3) Soltar la tecla cuando se cumple el hold_time calculado
        if jumping and (now - jump_hold_start) >= jump_hold_time:
            pyautogui.keyUp(JUMP)
            jumping        = False
            jump_hold_time = JUMP_MIN_HOLD  # reset

        # ---- Overlay de estado ----
        # Calcular carga visual (0.0 a 1.0)
        if gesture_start > 0:
            charge = min((now - gesture_start) / JUMP_MAX_HOLD, 1.0)
        else:
            charge = 0.0

        draw_status(frame, {
            "right":       moving_right,
            "left":        moving_left,
            "run":         running,
            "run_enabled": run_enabled,
            "jump":        jumping,
            "crouch":      crouching,
            "jump_charge": charge,
        })

        cv2.imshow("Control Mario", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

finally:
    # FIX: liberar recursos y asegurar que no queden teclas presionadas
    for key in [RIGHT, LEFT, RUN, DOWN, JUMP]:
        pyautogui.keyUp(key)
    cap.release()
    cv2.destroyAllWindows()
    print("Control cerrado correctamente.")

"""

Eye Blink -> Morse Code translator (real-time via webcam)
Author: (example) Bilal's Assistant
Requirements: opencv-python, mediapipe, numpy
pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ----------------------
# Morse code dictionary
# ----------------------
MORSE_CODE = {
    # letters
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    # numbers
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    # punctuation (common)
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":",
    "-.-.-.": ";", "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
    ".-..-.": "\"", "...-..-": "$", ".--.-.": "@",
    # special commands
    "........": "[BACKSPACE]"  # 8 dots for backspace/delete
}

# ----------------------
# Eye landmark indices for MediaPipe Face Mesh
# We'll pick 6 key points per eye to compute EAR:
# mapping: p1 (outer corner), p2 (upper inner), p3 (upper outer),
# p4 (inner corner), p5 (lower outer), p6 (lower inner)
# These indices are common selections for MediaPipe; they come from the full lists you provided.
# ----------------------
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# ----------------------
# Parameters (tunable)
# ----------------------
EAR_THRESHOLD = 0.15         # typical blink threshold (you should calibrate per-user)
EAR_SMOOTHING = 3            # frames for rolling average of EAR (reduced for faster response)
MIN_CONSEC_FRAMES = 1        # frames EAR must be below threshold to count as start (reduced)
DEBOUNCE_TIME = 0.08         # min seconds between consecutive detected blinks (reduced)
DOT_DASH_TIME = 0.3          # seconds: < -> dot, >= -> dash (reduced for easier control)
LETTER_GAP = 1.5             # seconds of silence to mark end-of-letter
WORD_GAP = 7.0               # seconds of silence to mark end-of-word
MIN_BLINK_DURATION = 0.03    # ignore extremely short durations (noise) (reduced)
MAX_BLINK_DURATION = 2.5     # ignore unrealistically long durations

# Code mode activation parameters
DOUBLE_BLINK_GAP = 0.5       # max time between two blinks to count as double-blink activation
CODE_MODE_TIMEOUT = 10.0     # seconds of inactivity before exiting code mode
ACTIVATION_BLINK_DURATION = 0.15  # max duration for activation blinks (quick blinks only)

# ----------------------
# Utility functions
# ----------------------
def euclidean(a, b):
    """Euclidean distance between two (x, y) points."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return np.linalg.norm(a - b)

def compute_ear(landmarks_px, eye_idx):
    """
    Compute Eye Aspect Ratio (EAR) for given eye.
    landmarks_px: list of (x,y) pixel coords for face landmarks (list length 468)
    eye_idx: list of 6 indices chosen for the eye
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Using the 6 points mapping defined above.
    """
    p1 = landmarks_px[eye_idx[0]]
    p2 = landmarks_px[eye_idx[1]]
    p3 = landmarks_px[eye_idx[2]]
    p4 = landmarks_px[eye_idx[3]]
    p5 = landmarks_px[eye_idx[4]]
    p6 = landmarks_px[eye_idx[5]]

    # safety: if any are None (face partially out of frame), return a high EAR so no blink
    if None in (p1, p2, p3, p4, p5, p6):
        return 1.0

    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 1.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def landmarks_to_pixel_coords(face_landmarks, width, height):
    """Convert MediaPipe normalized landmarks to pixel (x,y). Returns list of 468 (x,y)."""
    coords = []
    for lm in face_landmarks.landmark:
        x_px = min(int(lm.x * width), width - 1)
        y_px = min(int(lm.y * height), height - 1)
        coords.append((x_px, y_px))
    return coords

def morse_to_char(pattern):
    """Translate morse pattern to a character or special command. Returns '?' if unknown."""
    return MORSE_CODE.get(pattern, "?")

def process_morse_command(pattern, decoded_message):
    """Process morse pattern and return updated message. Handles special commands like backspace."""
    result = morse_to_char(pattern)
    
    if result == "[BACKSPACE]":
        # Remove last character if message exists
        if decoded_message:
            decoded_message = decoded_message[:-1]
            print(f"BACKSPACE executed: {pattern} -> removed last character")
        else:
            print(f"BACKSPACE ignored: {pattern} -> message already empty")
        return decoded_message
    else:
        # Regular character
        decoded_message += result
        print(f"Letter committed: {pattern} -> {result}")
        return decoded_message

# ----------------------
# Main loop
# ----------------------
def main(camera_index=0, target_width=640, target_height=480):
    global EAR_THRESHOLD  # Make it global so we can modify it with keyboard controls
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    # Optionally set FPS, but many webcams don't honor it
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check camera index and permissions.")
        return

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Try to import drawing utilities (optional for visualization)
    mp_drawing = None
    drawing_spec = None
    try:
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    except (AttributeError, ImportError):
        print("Drawing utilities not available - running without face mesh visualization")
        pass

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # offers more precise iris/eye points if available
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # EAR smoothing
    ear_buffer = deque(maxlen=EAR_SMOOTHING)

    # blink state
    blink_active = False
    consecutive_below = 0
    blink_start_time = None
    last_blink_time = 0
    last_signal_time = 0  # time of last dot or dash detected
    suppress_word_gap_until = 0  # suppress auto-space briefly after backspace

    # Code mode state (to differentiate intentional vs natural blinks)
    code_mode_active = False
    activation_blink_count = 0
    last_activation_blink_time = 0
    code_mode_start_time = 0
    pending_activation_blink = None  # store first blink timing for double-blink detection

    current_morse_symbol = ""  # e.g. ".-"
    decoded_message = ""       # readable text

    paused = False

    # FPS calculation
    last_frame_time = time.time()
    fps = 0.0

    print("Starting Eye Blink Morse Detector. Keys: 'q' quit, 'r' reset message, 'p' pause/unpause")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            h, w = frame.shape[:2]

            # To speed up, optionally resize (already requested via cap.set) - we'll use as-is
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not paused:
                results = face_mesh.process(rgb)
            else:
                results = None

            face_detected = False
            ear_value = None

            if results and results.multi_face_landmarks:
                face_detected = True
                # using first face
                face_landmarks = results.multi_face_landmarks[0]
                # convert to pixels
                coords = landmarks_to_pixel_coords(face_landmarks, w, h)

                # compute EAR for both eyes using our selected 6-point subset
                ear_r = compute_ear(coords, RIGHT_EYE_IDX)
                ear_l = compute_ear(coords, LEFT_EYE_IDX)
                ear_value = (ear_r + ear_l) / 2.0

                ear_buffer.append(ear_value)
                avg_ear = float(np.mean(ear_buffer)) if ear_buffer else ear_value

                # Draw face mesh and small markers for the 6 key points per eye (for debug)
                if mp_drawing is not None:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                    )

                # Draw the six selected points per eye and label them with better visualization
                # Right eye points (red)
                for i, idx in enumerate(RIGHT_EYE_IDX):
                    pt = coords[idx]
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)  # Red for right eye
                    cv2.circle(frame, pt, 5, (0, 0, 255), 1)   # Outer circle
                
                # Left eye points (blue)
                for i, idx in enumerate(LEFT_EYE_IDX):
                    pt = coords[idx]
                    cv2.circle(frame, pt, 3, (255, 0, 0), -1)  # Blue for left eye
                    cv2.circle(frame, pt, 5, (255, 0, 0), 1)   # Outer circle
                
                # Draw eye bounding boxes for better visualization
                right_eye_pts = [coords[idx] for idx in RIGHT_EYE_IDX]
                left_eye_pts = [coords[idx] for idx in LEFT_EYE_IDX]
                
                # Right eye bounding box
                if all(pt for pt in right_eye_pts):
                    right_x_coords = [pt[0] for pt in right_eye_pts]
                    right_y_coords = [pt[1] for pt in right_eye_pts]
                    cv2.rectangle(frame, 
                                (min(right_x_coords)-10, min(right_y_coords)-10),
                                (max(right_x_coords)+10, max(right_y_coords)+10),
                                (0, 0, 255), 1)
                    cv2.putText(frame, f"R: {ear_r:.3f}", (min(right_x_coords), min(right_y_coords)-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Left eye bounding box
                if all(pt for pt in left_eye_pts):
                    left_x_coords = [pt[0] for pt in left_eye_pts]
                    left_y_coords = [pt[1] for pt in left_eye_pts]
                    cv2.rectangle(frame, 
                                (min(left_x_coords)-10, min(left_y_coords)-10),
                                (max(left_x_coords)+10, max(left_y_coords)+10),
                                (255, 0, 0), 1)
                    cv2.putText(frame, f"L: {ear_l:.3f}", (min(left_x_coords), min(left_y_coords)-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # -------------------------
                # Blink detection state machine with Code Mode
                # -------------------------
                now = time.time()

                if avg_ear < EAR_THRESHOLD:
                    consecutive_below += 1
                else:
                    consecutive_below = 0

                # Start of blink: EAR below threshold for enough consecutive frames
                if (consecutive_below >= MIN_CONSEC_FRAMES) and (not blink_active):
                    # begin blink
                    blink_active = True
                    blink_start_time = now

                # End of blink: EAR goes back above threshold while blink was active
                if blink_active and (avg_ear >= EAR_THRESHOLD):
                    blink_end_time = now
                    blink_active = False
                    duration = blink_end_time - (blink_start_time or blink_end_time)

                    # Debounce: ignore blinks that are too close in time or too short/long
                    if (now - last_blink_time) > DEBOUNCE_TIME and MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                        
                        # Check if we're in code mode or trying to activate it
                        if not code_mode_active:
                            # NOT in code mode - check for activation sequence (double blink)
                            if duration <= ACTIVATION_BLINK_DURATION:  # Quick blink for activation
                                if pending_activation_blink is None:
                                    # First potential activation blink
                                    pending_activation_blink = now
                                    print("First activation blink detected. Blink again quickly to enter code mode.")
                                elif (now - pending_activation_blink) <= DOUBLE_BLINK_GAP:
                                    # Second blink within time window - activate code mode!
                                    code_mode_active = True
                                    code_mode_start_time = now
                                    activation_blink_count = 0
                                    pending_activation_blink = None
                                    print("🔴 CODE MODE ACTIVATED! Your blinks will now be interpreted as Morse code.")
                                    print("💡 Short blinks = dots (.), Long blinks = dashes (-)")
                                    print("⏰ Mode will auto-deactivate after 10 seconds of no blinks")
                                else:
                                    # Too much time passed, reset
                                    pending_activation_blink = now
                                    print("Activation timeout. First blink registered again.")
                            else:
                                # Normal long blink - just ignore it (natural blink)
                                pending_activation_blink = None
                                print(f"Natural blink ignored (duration: {duration:.3f}s)")
                        
                        else:
                            # IN CODE MODE - interpret as Morse code
                            symbol = "." if duration < DOT_DASH_TIME else "-"
                            current_morse_symbol += symbol
                            last_signal_time = now
                            code_mode_start_time = now  # Reset timeout
                            print(f"Code blink: duration={duration:.3f}s -> {symbol}")

                        last_blink_time = now

                # Check for code mode timeout
                if code_mode_active and (now - code_mode_start_time) > CODE_MODE_TIMEOUT:
                    code_mode_active = False
                    print("🟢 CODE MODE DEACTIVATED (timeout)")

                # Reset pending activation if too much time has passed
                if pending_activation_blink and (now - pending_activation_blink) > DOUBLE_BLINK_GAP:
                    pending_activation_blink = None
                
                # Letter and word gap detection (only in code mode)
                if code_mode_active:
                    # If we have not had any new blink for letter gap, commit current symbol
                    if current_morse_symbol and (time.time() - last_signal_time > LETTER_GAP):
                        # Identify if this symbol is a BACKSPACE command
                        is_backspace_cmd = (current_morse_symbol == "........") or (
                            MORSE_CODE.get(current_morse_symbol) == "[BACKSPACE]"
                        )
                        # Process morse command (handles backspace and regular characters)
                        decoded_message = process_morse_command(current_morse_symbol, decoded_message)
                        current_morse_symbol = ""
                        if is_backspace_cmd:
                            # Prevent immediate re-adding of a space due to word-gap logic
                            suppress_word_gap_until = time.time() + 1.0
                            # Also reset last_signal_time so gap timers start fresh
                            last_signal_time = time.time()

                    # Word gap: if no signal for WORD_GAP, add a space (if last char not space)
                    if (
                        decoded_message
                        and (time.time() - last_signal_time > WORD_GAP)
                        and (time.time() >= suppress_word_gap_until)
                    ):
                        # Only add if last char isn't a space to prevent multiples
                        if not decoded_message.endswith(" "):
                            decoded_message += " "
                            print("Word gap detected -> adding space")

            else:
                # No face detected
                ear_buffer.append(1.0)  # push a large EAR to avoid accidental blinks
                avg_ear = float(np.mean(ear_buffer)) if ear_buffer else 1.0
                # Optionally reset transient blink state if face gone
                blink_active = False
                consecutive_below = 0
                # note: we do not clear current morse or message — user may want to continue

            # -------------------------
            # Draw UI overlays
            # -------------------------
            # Main status box (larger)
            cv2.rectangle(frame, (0, 0), (w, 160), (0, 0, 0), -1)
            alpha = 0.7
            overlay = frame.copy()
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Helper function for text
            def add_text(text, y, color=(255,255,255), scale=0.7, thickness=1):
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            
            def add_large_text(text, y, color=(255,255,255), scale=1.2, thickness=2):
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

            # Title
            add_text("Eye Blink Morse Detector (Double-blink to activate code mode)", 18)
            
            # Code mode status (very prominent)
            if code_mode_active:
                mode_text = "🔴 CODE MODE ACTIVE"
                mode_color = (0, 0, 255)  # Red
                timeout_remaining = CODE_MODE_TIMEOUT - (time.time() - code_mode_start_time)
                timeout_text = f" (Timeout: {timeout_remaining:.1f}s)"
            elif pending_activation_blink:
                mode_text = "⚡ ACTIVATION PENDING - Blink again!"
                mode_color = (0, 255, 255)  # Yellow
                timeout_text = ""
            else:
                mode_text = "🟢 NORMAL MODE - Double-blink to activate"
                mode_color = (0, 255, 0)  # Green
                timeout_text = ""
            
            add_large_text(mode_text + timeout_text, 35, mode_color)
            
            # Large decoded message display
            add_large_text(f"DECODED MESSAGE: {decoded_message}", 65, (100, 255, 100))
            
            # Current morse pattern (only show in code mode)
            if code_mode_active and current_morse_symbol:
                add_text(f"Current Pattern: {current_morse_symbol}", 95, (200, 255, 200))
                # Show hint for special commands
                if current_morse_symbol == "........":
                    add_text("→ BACKSPACE ready (8 dots)", 110, (255, 255, 0))
            elif code_mode_active:
                add_text("Commands: ........ = Backspace | Press H for help", 95, (200, 200, 200))
            
            # Eye state information
            eye_state = "CLOSED" if avg_ear and avg_ear < EAR_THRESHOLD else "OPEN"
            eye_color = (0, 0, 255) if eye_state == "CLOSED" else (0, 255, 0)
            add_text(f"Eyes: {eye_state} | EAR: {avg_ear:.3f} | Threshold: {EAR_THRESHOLD:.3f}", 125, eye_color)
            
            # Blink status
            if code_mode_active:
                if blink_active:
                    blink_status = "Recording Code Blink..."
                    blink_color = (0, 255, 255)
                else:
                    blink_status = "Ready for Code Blinks"
                    blink_color = (255, 255, 255)
            else:
                if blink_active:
                    blink_status = "Natural Blink (Ignored)"
                    blink_color = (128, 128, 128)
                else:
                    blink_status = "Double-blink to activate code mode"
                    blink_color = (200, 200, 200)
            
            add_text(f"Status: {blink_status}", 150, blink_color)

            # Large output message box at bottom
            if decoded_message:
                msg_box_height = 80
                cv2.rectangle(frame, (0, h - msg_box_height), (w, h), (0, 50, 0), -1)
                overlay_bottom = frame.copy()
                cv2.addWeighted(overlay_bottom, 0.8, frame, 0.2, 0, frame)
                
                # Split long messages across lines
                words = decoded_message.split()
                line1 = " ".join(words[:5]) if len(words) > 5 else decoded_message
                line2 = " ".join(words[5:]) if len(words) > 5 else ""
                
                cv2.putText(frame, "OUTPUT:", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line1, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
                if line2:
                    cv2.putText(frame, line2, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)

            # Visual indicator for recent blink
            if last_blink_time > 0 and (time.time() - last_blink_time) < 1.0:
                recent_symbol = current_morse_symbol[-1] if current_morse_symbol else ""
                if recent_symbol:
                    symbol_text = "DOT (.)" if recent_symbol == "." else "DASH (-)" if recent_symbol == "-" else ""
                    cv2.putText(frame, symbol_text, (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

            # FPS display
            now_f = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now_f - last_frame_time)) if (now_f - last_frame_time) > 0 else fps
            last_frame_time = now_f
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA)

            # If face not detected show large warning
            if not face_detected:
                cv2.putText(frame, "NO FACE DETECTED!", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Position yourself in front of camera", (w//2-200, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Eye Blink Morse Code Detector", frame)

            # -------------------------
            # Keyboard controls
            # -------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_morse_symbol = ""
                decoded_message = ""
                print("Reset message and current sequence.")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('m'):
                # Manual code mode toggle
                code_mode_active = not code_mode_active
                if code_mode_active:
                    code_mode_start_time = time.time()
                    print("🔴 CODE MODE MANUALLY ACTIVATED")
                else:
                    print("🟢 CODE MODE MANUALLY DEACTIVATED")
                pending_activation_blink = None
            elif key == ord('h'):
                # Help
                print("\n=== EYE BLINK MORSE CODE HELP ===")
                print("🔸 Double-blink quickly to activate code mode")
                print("🔸 Or press 'M' to manually toggle code mode")
                print("🔸 In code mode: Short blinks = dots (.), Long blinks = dashes (-)")
                print("🔸 Code mode auto-deactivates after 10 seconds of no blinks")
                print("🔸 Normal blinks outside code mode are ignored")
                print("\n� SPECIAL MORSE COMMANDS:")
                print("🔹 ........ (8 dots) = BACKSPACE (delete last character)")
                print("🔹 .-.- (dot-dash-dot-dash) = SPACE (add explicit space)")
                print("\n⌨️  KEYBOARD CONTROLS:")
                print("🔹 Q=quit, R=reset, P=pause, C=calibrate, M=toggle mode, H=help")
                print("🔹 S=save message, T=auto-adjust threshold, ↑↓=adjust threshold")
            elif key == ord('c'):
                # Calibration mode
                print(f"Current EAR: {avg_ear:.3f}, Threshold: {EAR_THRESHOLD:.3f}")
                print("Keep your eyes OPEN and press UP arrow to increase threshold")
                print("Close your eyes and press DOWN arrow to decrease threshold")
                print("Current threshold works if eyes show OPEN when open, CLOSED when closed")
            elif key == 82:  # Up arrow key
                EAR_THRESHOLD += 0.01
                print(f"Threshold increased to: {EAR_THRESHOLD:.3f}")
            elif key == 84:  # Down arrow key
                EAR_THRESHOLD -= 0.01
                if EAR_THRESHOLD < 0.05:
                    EAR_THRESHOLD = 0.05
                print(f"Threshold decreased to: {EAR_THRESHOLD:.3f}")
            elif key == ord('t'):
                # Quick threshold adjustment based on current EAR
                if avg_ear:
                    EAR_THRESHOLD = avg_ear - 0.02
                    print(f"Auto-adjusted threshold to: {EAR_THRESHOLD:.3f} (based on current EAR)")
            elif key == ord('s'):
                # Save current message to file
                if decoded_message:
                    with open("morse_output.txt", "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {decoded_message}\n")
                    print(f"Saved message to morse_output.txt: {decoded_message}")
                else:
                    print("No message to save")

    finally:
        # cleanup
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

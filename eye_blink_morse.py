"""

Eye Blink -> Morse Code translator (real-time via webcam)
Author: (example) Bilal's Assistant
Requirements: opencv-python, mediapipe, numpy
pip install opencv-python mediapipe numpy
"""

import os
import time
from collections import deque

# MediaPipe can import TensorFlow internally. With newer protobuf packages,
# TensorFlow may crash on import unless protobuf uses the Python runtime.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

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
DOT_DASH_TIME = 0.45         # seconds: < -> dot, >= -> dash (slightly longer hold for dash)
BACKSPACE_HOLD_TIME = 3.0    # seconds: hold eyes closed in code mode to backspace
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
    """Process morse pattern and return updated message."""
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
# OpenCV dashboard UI
# ----------------------
UI_COLORS = {
    "bg": (27, 20, 18),
    "surface": (39, 30, 27),
    "surface_2": (33, 25, 22),
    "border": (74, 61, 55),
    "border_soft": (57, 45, 40),
    "text": (246, 240, 236),
    "muted": (176, 162, 154),
    "green": (160, 230, 100),
    "green_bg": (76, 95, 49),
    "red": (95, 92, 239),
    "red_bg": (70, 57, 87),
    "cyan": (238, 188, 32),
    "cyan_bg": (83, 69, 26),
    "yellow": (70, 220, 245),
    "white": (255, 255, 255),
}

def draw_round_rect(img, x1, y1, x2, y2, color, radius=16, thickness=-1, line_type=cv2.LINE_AA):
    """Draw a rounded rectangle with OpenCV primitives."""
    radius = max(0, min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness, line_type)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness, line_type)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness, line_type)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness, line_type)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness, line_type)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness, line_type)
        return

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, line_type)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, line_type)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, line_type)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, line_type)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, line_type)


def draw_text(img, text, x, y, scale=0.55, color=None, thickness=1, max_width=None):
    """Draw single-line text, trimming with ellipsis if a max width is supplied."""
    if color is None:
        color = UI_COLORS["text"]
    label = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if max_width is not None:
        original = label
        while label and cv2.getTextSize(label, font, scale, thickness)[0][0] > max_width:
            label = label[:-1]
        if label != original and len(label) > 3:
            label = label[:-3] + "..."
    cv2.putText(img, label, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_pill(img, x, y, text, color, bg_color, width=None, key=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width = cv2.getTextSize(text, font, 0.45, 1)[0][0]
    pill_width = width or max(90, text_width + (62 if key else 44))
    draw_round_rect(img, x, y, x + pill_width, y + 32, bg_color, radius=16)
    cv2.circle(img, (x + 18, y + 16), 5, color, -1, cv2.LINE_AA)
    draw_text(img, text, x + 32, y + 21, 0.45, UI_COLORS["text"], 1, pill_width - 40)
    if key:
        draw_round_rect(img, x + pill_width - 30, y + 6, x + pill_width - 8, y + 28, (59, 65, 76), radius=11)
        draw_text(img, key, x + pill_width - 23, y + 22, 0.38, UI_COLORS["muted"], 1)
    return pill_width


def place_image_fit(canvas, image, x, y, width, height, fill_color):
    """Fill a box without stretching; crop the overflow like a camera preview."""
    canvas[y:y + height, x:x + width] = fill_color
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    fitted_w = max(1, int(src_w * scale))
    fitted_h = max(1, int(src_h * scale))
    fitted = cv2.resize(image, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
    crop_x = max(0, (fitted_w - width) // 2)
    crop_y = max(0, (fitted_h - height) // 2)
    canvas[y:y + height, x:x + width] = fitted[crop_y:crop_y + height, crop_x:crop_x + width]


def draw_morse_symbols(img, pattern, x, y, color):
    if not pattern:
        draw_text(img, "Waiting for input", x, y + 18, 0.58, UI_COLORS["muted"], 1)
        return

    cursor = x
    for symbol in pattern[-8:]:
        if symbol == ".":
            cv2.circle(img, (cursor + 12, y + 14), 9, color, -1, cv2.LINE_AA)
            cursor += 38
        else:
            draw_round_rect(img, cursor, y + 5, cursor + 54, y + 23, color, radius=9)
            cursor += 72


def draw_event_stack(img, x, y, last_blink_time, current_morse_symbol, code_mode_active, blink_active):
    now = time.time()
    events = []
    if last_blink_time > 0 and (now - last_blink_time) < 1.2 and current_morse_symbol:
        symbol = current_morse_symbol[-1]
        events.append("Dash recorded" if symbol == "-" else "Dot recorded")
    if blink_active:
        events.append("Recording blink")
    if code_mode_active:
        events.append("Code mode ready")
    else:
        events.append("Natural blink guard")

    for index, event in enumerate(events[:3]):
        top = y + index * 58
        draw_round_rect(img, x, top, x + 206, top + 46, UI_COLORS["surface_2"], radius=16)
        draw_round_rect(img, x, top, x + 206, top + 46, UI_COLORS["border_soft"], radius=16, thickness=2)
        cv2.circle(img, (x + 24, top + 23), 8, UI_COLORS["green"] if index != 0 else UI_COLORS["cyan"], -1, cv2.LINE_AA)
        draw_text(img, event, x + 44, top + 29, 0.55, UI_COLORS["text"], 1, 145)


def build_dashboard(
    frame,
    face_detected,
    avg_ear,
    ear_threshold,
    code_mode_active,
    pending_activation_blink,
    blink_active,
    current_morse_symbol,
    decoded_message,
    fps,
    paused,
    code_mode_start_time,
    last_blink_time,
    show_help=False,
    debug_overlay=False,
):
    canvas_w, canvas_h = 1280, 820
    canvas = np.full((canvas_h, canvas_w, 3), UI_COLORS["bg"], dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, 78), (24, 27, 36), -1)
    cv2.line(canvas, (0, 78), (canvas_w, 78), UI_COLORS["border_soft"], 1, cv2.LINE_AA)

    margin = 32
    header_y = 48
    draw_round_rect(canvas, margin, 24, canvas_w - margin, 92, UI_COLORS["surface_2"], radius=18)
    draw_round_rect(canvas, margin, 24, canvas_w - margin, 92, UI_COLORS["border"], radius=18, thickness=2)
    draw_text(canvas, "Eye Blink Morse Translator", 58, header_y + 12, 0.78, UI_COLORS["text"], 1)

    mode_color = UI_COLORS["red"] if code_mode_active else UI_COLORS["muted"]
    mode_bg = UI_COLORS["red_bg"] if code_mode_active else UI_COLORS["surface"]
    mode_label = "Code Mode"
    if pending_activation_blink and not code_mode_active:
        mode_label = "Arming"
        mode_color = UI_COLORS["yellow"]
        mode_bg = UI_COLORS["cyan_bg"]

    draw_pill(canvas, 922, 39, "Camera", UI_COLORS["green"], UI_COLORS["green_bg"], 98)
    draw_pill(canvas, 1030, 39, mode_label, mode_color, mode_bg, 118)
    draw_text(canvas, "Help", 1180, 60, 0.5, UI_COLORS["muted"], 1)
    draw_round_rect(canvas, 1240, 40, 1262, 62, (45, 51, 64), radius=11)
    draw_text(canvas, "H", 1247, 57, 0.38, UI_COLORS["muted"], 1)

    camera_x, camera_y, camera_w, camera_h = 32, 112, 840, 500
    trans_x, trans_y, trans_w, trans_h = 894, 112, 354, 500

    draw_round_rect(canvas, camera_x, camera_y, camera_x + camera_w, camera_y + camera_h, UI_COLORS["surface"], radius=18)
    draw_round_rect(canvas, camera_x, camera_y, camera_x + camera_w, camera_y + camera_h, UI_COLORS["border"], radius=18, thickness=2)
    draw_text(canvas, "Camera Feed", camera_x + 24, camera_y + 40, 0.64, UI_COLORS["text"], 1)

    video_x, video_y, video_w, video_h = camera_x + 20, camera_y + 62, camera_w - 40, camera_h - 82
    draw_round_rect(canvas, video_x, video_y, video_x + video_w, video_y + video_h, (9, 11, 15), radius=10)
    place_image_fit(canvas, frame, video_x, video_y, video_w, video_h, (9, 11, 15))
    draw_round_rect(canvas, video_x, video_y, video_x + video_w, video_y + video_h, UI_COLORS["border_soft"], radius=10, thickness=2)

    face_label = "Face" if face_detected else "No Face"
    eyes_closed = avg_ear is not None and avg_ear < ear_threshold
    eye_label = "Eyes Closed" if eyes_closed else "Eyes Open"
    draw_pill(canvas, video_x + 16, video_y + 16, face_label, UI_COLORS["green"] if face_detected else UI_COLORS["red"], UI_COLORS["surface_2"], 104)
    draw_pill(canvas, video_x + 16, video_y + 54, eye_label, UI_COLORS["red"] if eyes_closed else UI_COLORS["green"], UI_COLORS["surface_2"], 122)
    if paused:
        draw_pill(canvas, video_x + video_w - 118, video_y + 16, "Paused", UI_COLORS["yellow"], UI_COLORS["surface_2"], 100)

    draw_round_rect(canvas, trans_x, trans_y, trans_x + trans_w, trans_y + trans_h, UI_COLORS["surface"], radius=18)
    draw_round_rect(canvas, trans_x, trans_y, trans_x + trans_w, trans_y + trans_h, UI_COLORS["border"], radius=18, thickness=2)
    draw_text(canvas, "Live Translation", trans_x + 28, trans_y + 44, 0.6, UI_COLORS["muted"], 1)

    message = decoded_message if decoded_message else "-"
    draw_text(canvas, message, trans_x + 28, trans_y + 114, 1.2, UI_COLORS["white"], 2, trans_w - 76)

    draw_event_stack(canvas, trans_x + trans_w - 196, trans_y + 4, last_blink_time, current_morse_symbol, code_mode_active, blink_active)

    draw_text(canvas, "Current Morse", trans_x + 28, trans_y + 354, 0.6, UI_COLORS["muted"], 1)
    draw_morse_symbols(canvas, current_morse_symbol, trans_x + 28, trans_y + 392, UI_COLORS["cyan"])

    if code_mode_active:
        timeout_remaining = max(0.0, CODE_MODE_TIMEOUT - (time.time() - code_mode_start_time))
        status = f"Listening - timeout in {timeout_remaining:.1f}s"
        status_color = UI_COLORS["red"]
    elif pending_activation_blink:
        status = "Activation pending - blink again"
        status_color = UI_COLORS["yellow"]
    else:
        status = "Normal mode - double blink to activate"
        status_color = UI_COLORS["muted"]
    draw_text(canvas, status, trans_x + 28, trans_y + 448, 0.48, status_color, 1, trans_w - 54)

    status_y = 632
    draw_round_rect(canvas, margin, status_y, canvas_w - margin, status_y + 56, UI_COLORS["surface_2"], radius=16)
    draw_round_rect(canvas, margin, status_y, canvas_w - margin, status_y + 56, UI_COLORS["border"], radius=16, thickness=2)
    cursor = margin + 24
    for text, color, bg, width in [
        ("Camera", UI_COLORS["green"] if face_detected else UI_COLORS["red"], UI_COLORS["green_bg"] if face_detected else UI_COLORS["red_bg"], 100),
        ("Face", UI_COLORS["green"] if face_detected else UI_COLORS["red"], UI_COLORS["green_bg"] if face_detected else UI_COLORS["red_bg"], 90),
        ("Open" if not eyes_closed else "Closed", UI_COLORS["green"] if not eyes_closed else UI_COLORS["red"], UI_COLORS["green_bg"] if not eyes_closed else UI_COLORS["red_bg"], 96),
        ("Code Mode" if code_mode_active else "Normal", mode_color, mode_bg, 112),
        (f"{fps:.0f} FPS", UI_COLORS["cyan"], UI_COLORS["cyan_bg"], 96),
        (f"EAR {avg_ear:.3f}", UI_COLORS["green"], UI_COLORS["green_bg"], 116),
        (f"TH {ear_threshold:.3f}", UI_COLORS["green"], UI_COLORS["green_bg"], 112),
        ("Debug" if debug_overlay else "Clean", UI_COLORS["yellow"] if debug_overlay else UI_COLORS["muted"], UI_COLORS["cyan_bg"] if debug_overlay else UI_COLORS["surface"], 104),
    ]:
        cursor += draw_pill(canvas, cursor, status_y + 12, text, color, bg, width) + 10

    toolbar_y = 712
    draw_round_rect(canvas, margin, toolbar_y, canvas_w - margin, toolbar_y + 86, UI_COLORS["surface_2"], radius=16)
    draw_round_rect(canvas, margin, toolbar_y, canvas_w - margin, toolbar_y + 86, UI_COLORS["border"], radius=16, thickness=2)
    actions = [
        ("Reset", "R", 110),
        ("Save", "S", 104),
        ("Calibrate", "C", 126),
        ("Auto", "T", 104),
        ("Help", "H", 104),
        ("Pause", "P", 112),
        ("Mode", "M", 104),
        ("Debug", "D", 112),
        ("Quit", "Q", 96),
    ]
    action_x = margin + 24
    for label, key, width in actions:
        draw_round_rect(canvas, action_x, toolbar_y + 20, action_x + width, toolbar_y + 66, UI_COLORS["surface"], radius=10)
        draw_round_rect(canvas, action_x, toolbar_y + 20, action_x + width, toolbar_y + 66, UI_COLORS["border"], radius=10, thickness=2)
        draw_text(canvas, label, action_x + 18, toolbar_y + 50, 0.5, UI_COLORS["text"], 1, width - 50)
        draw_round_rect(canvas, action_x + width - 32, toolbar_y + 30, action_x + width - 10, toolbar_y + 56, (58, 65, 76), radius=11)
        draw_text(canvas, key, action_x + width - 25, toolbar_y + 49, 0.38, UI_COLORS["muted"], 1)
        action_x += width + 12

    if not face_detected:
        draw_round_rect(canvas, video_x + 176, video_y + 128, video_x + video_w - 176, video_y + 210, UI_COLORS["surface_2"], radius=14)
        draw_round_rect(canvas, video_x + 176, video_y + 128, video_x + video_w - 176, video_y + 210, UI_COLORS["red_bg"], radius=14, thickness=2)
        draw_text(canvas, "No face detected", video_x + 226, video_y + 162, 0.78, UI_COLORS["white"], 2)
        draw_text(canvas, "Center yourself in the camera", video_x + 214, video_y + 192, 0.5, UI_COLORS["muted"], 1)

    if show_help:
        overlay = canvas.copy()
        help_x, help_y, help_w, help_h = 360, 184, 560, 340
        draw_round_rect(overlay, help_x, help_y, help_x + help_w, help_y + help_h, (19, 22, 29), radius=18)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        draw_round_rect(canvas, help_x, help_y, help_x + help_w, help_y + help_h, UI_COLORS["border"], radius=18, thickness=2)
        draw_text(canvas, "Keyboard Help", help_x + 30, help_y + 48, 0.82, UI_COLORS["white"], 2)
        help_items = [
            "Q  Quit application",
            "R  Reset message",
            "S  Save message",
            "P  Pause detection",
            "M  Toggle code mode",
            "C  Calibrate threshold",
            "T  Auto-adjust threshold",
            "Hold eyes closed 3s  Backspace",
            "D  Toggle face mesh overlay",
            "Up / Down  Adjust threshold",
        ]
        for index, item in enumerate(help_items):
            row_y = help_y + 92 + index * 28
            draw_text(canvas, item, help_x + 42, row_y, 0.54, UI_COLORS["text"], 1)
        draw_text(canvas, "Press H to close", help_x + 330, help_y + help_h - 26, 0.42, UI_COLORS["muted"], 1)

    return canvas
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
    long_close_backspace_done = False
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
    help_visible = False
    debug_overlay = False

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

                if debug_overlay and mp_drawing is not None:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                    )

                    for idx in RIGHT_EYE_IDX:
                        pt = coords[idx]
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
                        cv2.circle(frame, pt, 5, (0, 0, 255), 1)

                    for idx in LEFT_EYE_IDX:
                        pt = coords[idx]
                        cv2.circle(frame, pt, 3, (255, 0, 0), -1)
                        cv2.circle(frame, pt, 5, (255, 0, 0), 1)

                    right_eye_pts = [coords[idx] for idx in RIGHT_EYE_IDX]
                    left_eye_pts = [coords[idx] for idx in LEFT_EYE_IDX]

                    if all(pt for pt in right_eye_pts):
                        right_x_coords = [pt[0] for pt in right_eye_pts]
                        right_y_coords = [pt[1] for pt in right_eye_pts]
                        cv2.rectangle(frame, (min(right_x_coords) - 10, min(right_y_coords) - 10), (max(right_x_coords) + 10, max(right_y_coords) + 10), (0, 0, 255), 1)
                        cv2.putText(frame, f"R: {ear_r:.3f}", (min(right_x_coords), min(right_y_coords) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    if all(pt for pt in left_eye_pts):
                        left_x_coords = [pt[0] for pt in left_eye_pts]
                        left_y_coords = [pt[1] for pt in left_eye_pts]
                        cv2.rectangle(frame, (min(left_x_coords) - 10, min(left_y_coords) - 10), (max(left_x_coords) + 10, max(left_y_coords) + 10), (255, 0, 0), 1)
                        cv2.putText(frame, f"L: {ear_l:.3f}", (min(left_x_coords), min(left_y_coords) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

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

                # Long close in code mode is a deliberate backspace gesture.
                if (
                    code_mode_active
                    and blink_active
                    and blink_start_time is not None
                    and not long_close_backspace_done
                    and (now - blink_start_time) >= BACKSPACE_HOLD_TIME
                ):
                    if current_morse_symbol:
                        current_morse_symbol = ""
                        print("Pending Morse sequence cleared by long eye close")
                    if decoded_message:
                        decoded_message = decoded_message[:-1]
                        print("BACKSPACE executed: eyes closed for 3 seconds -> removed last character")
                    else:
                        print("BACKSPACE ignored: eyes closed for 3 seconds -> message already empty")
                    long_close_backspace_done = True
                    last_signal_time = now
                    last_blink_time = now
                    code_mode_start_time = now
                    suppress_word_gap_until = now + 1.0

                # End of blink: EAR goes back above threshold while blink was active
                if blink_active and (avg_ear >= EAR_THRESHOLD):
                    blink_end_time = now
                    blink_active = False
                    was_long_close_backspace = long_close_backspace_done
                    long_close_backspace_done = False
                    duration = blink_end_time - (blink_start_time or blink_end_time)

                    # Debounce: ignore blinks that are too close in time or too short/long
                    if (not was_long_close_backspace) and (now - last_blink_time) > DEBOUNCE_TIME and MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                        
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
                        decoded_message = process_morse_command(current_morse_symbol, decoded_message)
                        current_morse_symbol = ""

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
                long_close_backspace_done = False
                consecutive_below = 0
                # note: we do not clear current morse or message — user may want to continue

            # -------------------------
            # Draw dashboard UI
            # -------------------------
            now_f = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now_f - last_frame_time)) if (now_f - last_frame_time) > 0 else fps
            last_frame_time = now_f

            dashboard = build_dashboard(
                frame=frame,
                face_detected=face_detected,
                avg_ear=avg_ear,
                ear_threshold=EAR_THRESHOLD,
                code_mode_active=code_mode_active,
                pending_activation_blink=pending_activation_blink,
                blink_active=blink_active,
                current_morse_symbol=current_morse_symbol,
                decoded_message=decoded_message,
                fps=fps,
                paused=paused,
                code_mode_start_time=code_mode_start_time,
                last_blink_time=last_blink_time,
                show_help=help_visible,
                debug_overlay=debug_overlay,
            )
            cv2.imshow("Eye Blink Morse Code Detector", dashboard)
            # -------------------------
            # Keyboard controls
            # -------------------------
            raw_key = cv2.waitKeyEx(1)
            key = raw_key & 0xFF
            key_char = chr(key).lower() if 0 <= key < 256 else ""
            if key_char == 'q':
                break
            elif key_char == 'r':
                current_morse_symbol = ""
                decoded_message = ""
                print("Reset message and current sequence.")
            elif key_char == 'p':
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key_char == 'm':
                # Manual code mode toggle
                code_mode_active = not code_mode_active
                if code_mode_active:
                    code_mode_start_time = time.time()
                    print("CODE MODE MANUALLY ACTIVATED")
                else:
                    print("CODE MODE MANUALLY DEACTIVATED")
                pending_activation_blink = None
            elif key_char == 'h':
                help_visible = not help_visible
                # Help
                print("\n=== EYE BLINK MORSE CODE HELP ===")
                print("Double-blink quickly to activate code mode")
                print("Or press 'M' to manually toggle code mode")
                print("In code mode: Short blinks = dots (.), Long blinks = dashes (-)")
                print("Code mode auto-deactivates after 10 seconds of no blinks")
                print("Normal blinks outside code mode are ignored")
                print("\nSPECIAL MORSE COMMANDS:")
                print("Hold eyes closed for 3 seconds = BACKSPACE (delete last character)")
                print(".-.- (dot-dash-dot-dash) = SPACE (add explicit space)")
                print("\nKEYBOARD CONTROLS:")
                print("Q=quit, R=reset, P=pause, C=calibrate, M=toggle mode, H=help")
                print("S=save message, T=auto-adjust threshold, D=toggle face mesh, Up/Down=adjust threshold")
            elif key_char == 'd':
                debug_overlay = not debug_overlay
                print("Face mesh overlay ON" if debug_overlay else "Face mesh overlay OFF")
            elif key_char == 'c':
                # Calibration mode
                print(f"Current EAR: {avg_ear:.3f}, Threshold: {EAR_THRESHOLD:.3f}")
                print("Keep your eyes OPEN and press UP arrow to increase threshold")
                print("Close your eyes and press DOWN arrow to decrease threshold")
                print("Current threshold works if eyes show OPEN when open, CLOSED when closed")
            elif raw_key in (2490368, 65362):  # Up arrow key
                EAR_THRESHOLD += 0.01
                print(f"Threshold increased to: {EAR_THRESHOLD:.3f}")
            elif raw_key in (2621440, 65364):  # Down arrow key
                EAR_THRESHOLD -= 0.01
                if EAR_THRESHOLD < 0.05:
                    EAR_THRESHOLD = 0.05
                print(f"Threshold decreased to: {EAR_THRESHOLD:.3f}")
            elif key_char == 't':
                # Quick threshold adjustment based on current EAR
                if avg_ear:
                    EAR_THRESHOLD = avg_ear - 0.02
                    print(f"Auto-adjusted threshold to: {EAR_THRESHOLD:.3f} (based on current EAR)")
            elif key_char == 's':
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

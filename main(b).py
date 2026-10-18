# ============================================================
# NEXA - AI BASED EMERGENCY HAND SIGN DETECTION final prototype code
# ============================================================

import cv2
import time
import os
import shutil
import subprocess
import math
import traceback
from collections import deque

import numpy as np
import mediapipe as mp
import tensorflow as tf
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "emergency_gesture_model.keras"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

VIDEO_FOLDER = os.path.join(
    BASE_DIR,
    "emergency_clips"
)

BACKEND_URL = "http://127.0.0.1:8000/emergency"


# ============================================================
# CONFIGURATION
# ============================================================

# Existing ML model threshold for SIGN 2
EMERGENCY_THRESHOLD = 0.90

# Sign 2 must remain valid for this long
REQUIRED_TIME = 2.0

# Sign 2 must occur within this time after Sign 1
SEQUENCE_TIMEOUT = 5.0

# How long EMERGENCY should remain on screen
EMERGENCY_DISPLAY_TIME = 5.0

# Video before emergency
BUFFER_SECONDS = 5

# Video after emergency
RECORD_AFTER_SECONDS = 5

# Minimum time between recordings
RECORD_COOLDOWN = 10


# ============================================================
# SIGN 1 CONFIGURATION
#
# SIGN 1:
#
# Index   = OPEN
# Middle  = OPEN
# Ring    = OPEN
# Pinky   = OPEN
# Thumb   = FOLDED
# ============================================================

SIGN1_TIMEOUT = SEQUENCE_TIMEOUT

# Sign 1 must be seen on this many consecutive frames before
# the Sign 1 -> Sign 2 sequence starts. This removes one-frame
# false positives caused by landmark jitter.
SIGN1_CONFIRM_FRAMES = 3

# ---- Finger "open" thresholds (degrees / ratios) ----
# Angle at the PIP joint (MCP-PIP-DIP). ~180 = straight finger,
# ~60-100 = curled finger.
FINGER_PIP_ANGLE_MIN = 160.0

# Angle at the DIP joint (PIP-DIP-TIP).
FINGER_DIP_ANGLE_MIN = 150.0

# Fingertip must be at least this many times farther from the
# wrist than the finger's own MCP knuckle.
FINGER_TIP_TO_MCP_RATIO_MIN = 1.35

# ---- Thumb "folded" thresholds (ratios of hand size) ----
# hand size = distance wrist (0) -> middle-finger MCP (9)
#
# Thumb tip must be close to the palm centre ...
THUMB_PALM_RATIO_MAX = 0.60

# ... and close to the base of the index finger.
THUMB_INDEX_MCP_RATIO_MAX = 0.80


# ============================================================
# CREATE VIDEO FOLDER
# ============================================================

os.makedirs(
    VIDEO_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK FFMPEG
# ============================================================

FFMPEG_PATH = shutil.which("ffmpeg")

if FFMPEG_PATH is None:

    print()
    print("❌ FFmpeg was not found.")
    print("Please install FFmpeg and restart Cursor.")
    exit()

print()
print("FFmpeg found:")
print(FFMPEG_PATH)


# ============================================================
# COLORS - BGR
# ============================================================

BG = (15, 18, 25)

PANEL = (25, 30, 40)

WHITE = (235, 240, 245)

GRAY = (130, 140, 150)

CYAN = (255, 210, 40)

GREEN = (80, 220, 120)

ORANGE = (40, 170, 255)

RED = (70, 70, 255)


# ============================================================
# LOAD AI MODEL
# ============================================================

print()
print("Loading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "AI model loaded successfully!"
)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


hand_landmarker = (
    HandLandmarker.create_from_options(
        options
    )
)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(
        21,
        3
    )

    # Wrist becomes origin

    landmarks = (
        landmarks
        - landmarks[0]
    )

    # Scale

    scale = np.max(
        np.linalg.norm(
            landmarks,
            axis=1
        )
    )

    if scale == 0:

        return None

    landmarks = (
        landmarks / scale
    )

    return landmarks.flatten()


# ============================================================
# LANDMARK DISTANCE
#
# Used by the (unchanged) Sign 2 shape check.
# ============================================================

def landmark_distance(a, b):

    return math.sqrt(

        (a.x - b.x) ** 2 +

        (a.y - b.y) ** 2 +

        (a.z - b.z) ** 2

    )


# ============================================================
# SIGN 1 GEOMETRY HELPERS
#
# WHY THE OLD DETECTOR WAS UNRELIABLE:
#
# 1. MediaPipe returns x and y normalized separately by image
#    width and height. With a 900x600 frame this stretches the
#    x axis relative to y, so "distances" were distorted and
#    changed with hand rotation.
#
# 2. Fingers were judged only by "tip is farther from the wrist
#    than the PIP joint", which is also true for many curled or
#    half-curled fingers.
#
# 3. The thumb was judged with one loose distance test.
#
# NEW APPROACH:
#
# - All geometry is computed on landmarks converted to a common
#   pixel scale (x * width, y * height, z * width), so distances
#   and angles are not distorted by the frame's aspect ratio.
#
# - Fingers use joint ANGLES (PIP and DIP) plus a tip-to-knuckle
#   reach check. Joint angles and distance ratios do not depend
#   on which way the hand points or on mirroring of the image,
#   because they use no fixed x / y direction comparisons.
#
# - The thumb is judged by how close its tip is to the palm
#   centre and to the index-finger base, relative to hand size.
#
# LIMITATIONS (not tested against every hand / orientation):
#
# - MediaPipe z is an estimate and is noisy; angles shrink when
#   a finger points toward or away from the camera
#   (foreshortening), so an open finger aimed at the camera can
#   be rejected.
#
# - Thresholds are tuned by reasoning about typical hand
#   proportions, not by testing on a dataset. Different hand
#   sizes, camera angles or lighting may need the constants at
#   the top of the file adjusted.
#
# - A thumb pressed flat against the side of the hand can be
#   classified as folded, and a half-folded thumb near the
#   threshold may flicker between folded and not folded.
#
# - Poor hand tracking (blur, partial hand in frame) produces
#   bad landmarks, which produces bad results.
# ============================================================

def landmarks_to_points(
    hand,
    frame_width,
    frame_height
):

    return np.array(

        [
            [
                landmark.x * frame_width,
                landmark.y * frame_height,
                landmark.z * frame_width
            ]
            for landmark in hand
        ],

        dtype=np.float32

    )


def joint_angle(a, b, c):

    # Angle at point b, in degrees, between segments b->a and b->c

    ba = a - b

    bc = c - b

    norm_ba = np.linalg.norm(ba)

    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:

        return 0.0

    cosine = float(

        np.dot(ba, bc)
        / (norm_ba * norm_bc)

    )

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# CHECK WHETHER FINGER IS EXTENDED
#
# A finger is open only if ALL of these are true:
#
# 1. PIP joint is nearly straight
# 2. DIP joint is nearly straight
# 3. Fingertip reaches well beyond the finger's knuckle (MCP),
#    measured from the wrist
# 4. Fingertip is farther from the wrist than the PIP joint
# ============================================================

def finger_is_extended(
    points,
    mcp_index,
    pip_index,
    dip_index,
    tip_index
):

    wrist = points[0]

    mcp = points[mcp_index]

    pip = points[pip_index]

    dip = points[dip_index]

    tip = points[tip_index]


    pip_angle = joint_angle(
        mcp,
        pip,
        dip
    )

    dip_angle = joint_angle(
        pip,
        dip,
        tip
    )


    tip_to_wrist = float(
        np.linalg.norm(tip - wrist)
    )

    pip_to_wrist = float(
        np.linalg.norm(pip - wrist)
    )

    mcp_to_wrist = float(
        np.linalg.norm(mcp - wrist)
    )


    if mcp_to_wrist == 0:

        return False


    return bool(

        pip_angle >= FINGER_PIP_ANGLE_MIN

        and dip_angle >= FINGER_DIP_ANGLE_MIN

        and tip_to_wrist > pip_to_wrist

        and (
            tip_to_wrist / mcp_to_wrist
            >= FINGER_TIP_TO_MCP_RATIO_MIN
        )

    )


# ============================================================
# CHECK WHETHER THUMB IS FOLDED
#
# The thumb counts as folded only when its tip is close to the
# palm centre AND close to the base of the index finger.
# An extended thumb points away from the palm, so its tip is
# far from both points regardless of hand rotation.
# ============================================================

def thumb_is_folded(points):

    hand_size = float(

        np.linalg.norm(
            points[9] - points[0]
        )

    )

    if hand_size == 0:

        return False


    palm_center = np.mean(

        points[[0, 5, 9, 13, 17]],

        axis=0

    )

    thumb_tip = points[4]


    tip_to_palm = float(

        np.linalg.norm(
            thumb_tip - palm_center
        )

    ) / hand_size


    tip_to_index_mcp = float(

        np.linalg.norm(
            thumb_tip - points[5]
        )

    ) / hand_size


    return bool(

        tip_to_palm <= THUMB_PALM_RATIO_MAX

        and

        tip_to_index_mcp
        <= THUMB_INDEX_MCP_RATIO_MAX

    )


# ============================================================
# SIGN 1 DETECTOR
#
# Index   = OPEN
# Middle  = OPEN
# Ring    = OPEN
# Pinky   = OPEN
# Thumb   = FOLDED
#
# Strict result: every condition must be true at once.
# ============================================================

def detect_sign1(
    hand,
    frame_width=900,
    frame_height=600
):

    if hand is None or len(hand) < 21:

        return False


    points = landmarks_to_points(
        hand,
        frame_width,
        frame_height
    )


    # Index finger  : landmarks 5, 6, 7, 8
    index_open = finger_is_extended(
        points,
        5,
        6,
        7,
        8
    )

    # Middle finger : landmarks 9, 10, 11, 12
    middle_open = finger_is_extended(
        points,
        9,
        10,
        11,
        12
    )

    # Ring finger   : landmarks 13, 14, 15, 16
    ring_open = finger_is_extended(
        points,
        13,
        14,
        15,
        16
    )

    # Pinky         : landmarks 17, 18, 19, 20
    pinky_open = finger_is_extended(
        points,
        17,
        18,
        19,
        20
    )

    # Thumb         : landmarks 1, 2, 3, 4
    thumb_folded = thumb_is_folded(
        points
    )


    sign1_detected = (

        index_open

        and middle_open

        and ring_open

        and pinky_open

        and thumb_folded

    )


    return bool(sign1_detected)


# ============================================================
# SIGN 2 DETECTOR
#
# ALL FOUR FINGERS MUST BE CLOSED
# AND THUMB MUST BE CLOSED.
#
# This prevents Sign 1 from being accepted as Sign 2
# even when the ML model gives high confidence.
# ============================================================

def all_fingers_closed(hand):

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index_closed = (
        hand[8].y > hand[6].y
    )


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle_closed = (
        hand[12].y > hand[10].y
    )


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring_closed = (
        hand[16].y > hand[14].y
    )


    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    pinky_closed = (
        hand[20].y > hand[18].y
    )


    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = hand[4]

    thumb_mcp = hand[2]

    thumb_closed = (

        landmark_distance(
            thumb_tip,
            thumb_mcp
        )

        <

        0.12

    )


    # --------------------------------------------------------
    # ALL FINGERS MUST BE CLOSED
    # --------------------------------------------------------

    return (

        index_closed

        and middle_closed

        and ring_closed

        and pinky_closed

        and thumb_closed

    )


# ============================================================
# DRAW TEXT
# ============================================================

def text(
    frame,
    message,
    position,
    size=0.6,
    color=WHITE,
    thickness=1
):

    cv2.putText(

        frame,

        message,

        position,

        cv2.FONT_HERSHEY_SIMPLEX,

        size,

        color,

        thickness,

        cv2.LINE_AA

    )


# ============================================================
# DRAW PANEL
# ============================================================

def panel(
    frame,
    x1,
    y1,
    x2,
    y2
):

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        PANEL,

        -1

    )

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        (55, 65, 80),

        1

    )


# ============================================================
# SEND EMERGENCY ALERT
# ============================================================

def send_emergency_alert(
    video_filename
):

    try:

        data = {
            "video": video_filename
        }

        response = requests.post(

            BACKEND_URL,

            json=data,

            timeout=5

        )

        if response.status_code == 200:

            print()

            print(
                "🚨 Emergency alert sent to backend!"
            )

            return True


        print(
            "❌ Backend returned:",
            response.status_code
        )

        return False


    except requests.exceptions.RequestException as e:

        print(
            "❌ Could not connect to backend:",
            e
        )

        return False


# ============================================================
# CONVERT VIDEO TO H.264
# ============================================================

def convert_to_h264(
    raw_filepath,
    final_filepath
):

    print()

    print("=" * 60)

    print("🎬 Converting video to H.264...")

    print("=" * 60)


    command = [

        FFMPEG_PATH,

        "-y",

        "-i",
        raw_filepath,

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        final_filepath

    ]


    try:

        result = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True

        )


        if result.returncode != 0:

            print(
                "❌ FFmpeg conversion failed."
            )

            print(
                result.stderr
            )

            return False


        print(
            "✅ H.264 video created."
        )


        if os.path.exists(
            raw_filepath
        ):

            os.remove(
                raw_filepath
            )


        return True


    except Exception as e:

        print(
            "❌ FFmpeg error:",
            e
        )

        return False


# ============================================================
# FINALIZE RECORDING
# ============================================================

def finalize_recording(
    video_writer,
    raw_filepath,
    final_filepath
):

    # --------------------------------------------------------
    # RELEASE WRITER
    # --------------------------------------------------------

    if video_writer is not None:

        video_writer.release()

        print(
            "Video writer closed."
        )


    # --------------------------------------------------------
    # CHECK RAW FILE
    # --------------------------------------------------------

    if not os.path.exists(
        raw_filepath
    ):

        print(
            "❌ Raw video file does not exist."
        )

        return False


    raw_size = os.path.getsize(
        raw_filepath
    )


    print(
        f"Raw video size: "
        f"{raw_size / 1024 / 1024:.2f} MB"
    )


    if raw_size == 0:

        print(
            "❌ Raw video is empty."
        )

        return False


    # --------------------------------------------------------
    # CONVERT
    # --------------------------------------------------------

    success = convert_to_h264(

        raw_filepath,

        final_filepath

    )


    if not success:

        return False


    # --------------------------------------------------------
    # VERIFY FINAL FILE
    # --------------------------------------------------------

    if not os.path.exists(
        final_filepath
    ):

        print(
            "❌ Final video was not created."
        )

        return False


    final_size = os.path.getsize(
        final_filepath
    )


    print(
        f"Final video size: "
        f"{final_size / 1024 / 1024:.2f} MB"
    )


    if final_size == 0:

        print(
            "❌ Final video is empty."
        )

        return False


    print()

    print(
        "🎉 Emergency video completely saved!"
    )

    print(
        "📁",
        final_filepath
    )

    return True


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "❌ ERROR: Could not open webcam."
    )

    hand_landmarker.close()

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print()

print(
    "Camera started."
)

print(
    "Press Q or ESC to quit."
)


# ============================================================
# FPS
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0 or fps > 120:

    fps = 30


print(
    "Camera FPS:",
    fps
)


# ============================================================
# VIDEO BUFFER
# ============================================================

max_buffer_frames = int(
    BUFFER_SECONDS * fps
)

frame_buffer = deque(
    maxlen=max_buffer_frames
)


# ============================================================
# RECORDING VARIABLES
# ============================================================

recording = False

video_writer = None

recording_end_time = None

last_saved_time = 0

current_video_filename = None

current_raw_filepath = None

current_final_filepath = None


# ============================================================
# TWO-SIGN SEQUENCE VARIABLES
# ============================================================

SEQUENCE_WAITING = 0

SEQUENCE_SIGN1_DETECTED = 1


sequence_state = SEQUENCE_WAITING

sequence_start_time = None

# Consecutive frames in which Sign 1 was seen
sign1_frames = 0


# ============================================================
# EMERGENCY VARIABLES
# ============================================================

emergency_start_time = None

emergency_verified = False

# Stores when emergency was verified
emergency_verified_time = None


# ============================================================
# START TIME
# ============================================================

start_time = time.time()


# ============================================================
# OPENCV WINDOW
# ============================================================

WINDOW_NAME = "NEXA - Emergency AI"


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1200,
    700
)

cv2.moveWindow(
    WINDOW_NAME,
    50,
    50
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        ret, camera_frame = cap.read()


        if not ret:

            print(
                "❌ Failed to read camera."
            )

            break


        # ====================================================
        # MIRROR
        # ====================================================

        camera_frame = cv2.flip(
            camera_frame,
            1
        )


        # ====================================================
        # RESIZE
        # ====================================================

        camera_frame = cv2.resize(

            camera_frame,

            (900, 600)

        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb_frame = cv2.cvtColor(

            camera_frame,

            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame

        )


        timestamp_ms = int(

            (
                time.time()
                - start_time
            )
            * 1000

        )


        result = (

            hand_landmarker

            .detect_for_video(

                mp_image,

                timestamp_ms

            )

        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        probability = 0.0

        hand_detected = False

        sign1_detected = False

        sign2_shape_detected = False


        # ====================================================
        # HAND DETECTION
        # ====================================================

        if result.hand_landmarks:

            hand_detected = True

            hand = (
                result.hand_landmarks[0]
            )


            # =================================================
            # SIGN 1
            #
            # Geometry is evaluated in pixel scale, so the
            # actual frame size is passed in.
            # =================================================

            sign1_detected = detect_sign1(

                hand,

                camera_frame.shape[1],

                camera_frame.shape[0]

            )


            # =================================================
            # SIGN 2 SHAPE
            # =================================================

            sign2_shape_detected = (
                all_fingers_closed(hand)
            )


            # =================================================
            # CREATE LANDMARK ARRAY
            # =================================================

            landmarks = []


            for landmark in hand:

                landmarks.append(

                    [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]

                )


            # =================================================
            # NORMALIZE
            # =================================================

            features = normalize_landmarks(
                landmarks
            )


            # =================================================
            # MODEL PREDICTION
            # =================================================

            if features is not None:

                features = np.array(
                    features
                ).reshape(
                    1,
                    63
                )


                probability = float(

                    model.predict(

                        features,

                        verbose=0

                    )[0][0]

                )


            # =================================================
            # DRAW HAND LANDMARKS
            # =================================================

            h, w, _ = (
                camera_frame.shape
            )


            points = []


            for landmark in hand:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                points.append(
                    (x, y)
                )


            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17)

            ]


            for start, end in connections:

                cv2.line(

                    camera_frame,

                    points[start],

                    points[end],

                    CYAN,

                    2,

                    cv2.LINE_AA

                )


            for point in points:

                cv2.circle(

                    camera_frame,

                    point,

                    5,

                    WHITE,

                    -1,

                    cv2.LINE_AA

                )


        else:

            hand_detected = False

            sign1_detected = False

            sign2_shape_detected = False


        # ====================================================
        # CONSECUTIVE SIGN 1 FRAME COUNTER
        # ====================================================

        if sign1_detected:

            sign1_frames += 1

        else:

            sign1_frames = 0


        # ====================================================
        # ADD FRAME TO BUFFER
        # ====================================================

        frame_buffer.append(
            camera_frame.copy()
        )


        # ====================================================
        # CURRENT TIME
        # ====================================================

        current_time = time.time()


        # ====================================================
        # RESET EMERGENCY DISPLAY STATUS
        #
        # After 5 seconds, return to normal operation.
        #
        # This only resets the on-screen emergency state and
        # the detection sequence. It does NOT touch the
        # recording variables, the video writer, the FFmpeg
        # conversion, or the backend alert, so those finish
        # normally in the RECORDING section below.
        # ====================================================

        if (

            emergency_verified

            and

            emergency_verified_time is not None

            and

            (
                current_time
                - emergency_verified_time
            )

            >= EMERGENCY_DISPLAY_TIME

        ):

            emergency_verified = False

            emergency_verified_time = None

            sequence_state = (
                SEQUENCE_WAITING
            )

            sequence_start_time = None

            emergency_start_time = None

            sign1_frames = 0


            print()

            print(
                "✅ Emergency status cleared."
            )

            print(
                "System returned to normal monitoring."
            )


        # ====================================================
        # TWO-SIGN SEQUENCE
        # ====================================================

        if (

            sequence_state
            == SEQUENCE_WAITING

            and

            not emergency_verified

        ):

            emergency_start_time = None


            # ------------------------------------------------
            # SIGN 1 must be stable for several frames
            # ------------------------------------------------

            if (

                sign1_detected

                and

                sign1_frames
                >= SIGN1_CONFIRM_FRAMES

            ):

                sequence_state = (
                    SEQUENCE_SIGN1_DETECTED
                )


                sequence_start_time = (
                    current_time
                )


                print()

                print("=" * 60)

                print(
                    "✋ SIGN 1 DETECTED"
                )

                print(
                    "Waiting for emergency SIGN 2..."
                )

                print("=" * 60)


        # ====================================================
        # STATE 1
        # SIGN 1 WAS DETECTED
        # ====================================================

        elif (

            sequence_state
            == SEQUENCE_SIGN1_DETECTED

            and

            not emergency_verified

        ):

            sequence_elapsed = (

                current_time
                - sequence_start_time

            )


            # ------------------------------------------------
            # SIGN 2 VALIDITY
            #
            # BOTH CONDITIONS MUST BE TRUE:
            #
            # 1. ML confidence >= 90%
            # 2. ALL fingers are physically closed
            #
            # High ML probability alone is NOT enough.
            # ------------------------------------------------

            sign2_valid = (

                probability
                >= EMERGENCY_THRESHOLD

                and

                sign2_shape_detected

            )


            # ------------------------------------------------
            # TIMEOUT
            #
            # The 5 second window is for WAITING for Sign 2.
            # A verification already in progress is not cut
            # off by the timeout; it ends only when it
            # completes or when Sign 2 becomes invalid.
            # ------------------------------------------------

            if (

                emergency_start_time is None

                and

                sequence_elapsed
                > SEQUENCE_TIMEOUT

            ):

                print()

                print(
                    "⏱️ Sequence timeout."
                )

                print(
                    "SIGN 2 was not detected."
                )


                sequence_state = (
                    SEQUENCE_WAITING
                )

                sequence_start_time = None

                emergency_start_time = None

                sign1_frames = 0


            # ------------------------------------------------
            # SIGN 2 VALID
            # ------------------------------------------------

            elif sign2_valid:

                if emergency_start_time is None:

                    emergency_start_time = (
                        current_time
                    )


                    print()

                    print(
                        "🚨 SIGN 2 DETECTED!"
                    )

                    print(
                        "Starting final verification..."
                    )


                elapsed = (

                    current_time
                    - emergency_start_time

                )


                # ------------------------------------------------
                # 2 SECOND VERIFICATION COMPLETE
                # ------------------------------------------------

                if elapsed >= REQUIRED_TIME:

                    emergency_verified = True

                    emergency_verified_time = (
                        current_time
                    )


                    print()

                    print("=" * 60)

                    print(
                        "🚨 EMERGENCY VERIFIED!"
                    )

                    print(
                        "SIGN 1 → SIGN 2 sequence completed."
                    )

                    print("=" * 60)


                    # ====================================================
                    # START RECORDING
                    # ====================================================

                    if (

                        not recording

                        and

                        (
                            current_time
                            - last_saved_time
                        )
                        >= RECORD_COOLDOWN

                    ):

                        timestamp = time.strftime(
                            "%Y%m%d_%H%M%S"
                        )


                        current_video_filename = (

                            f"emergency_"
                            f"{timestamp}.mp4"

                        )


                        current_final_filepath = os.path.join(

                            VIDEO_FOLDER,

                            current_video_filename

                        )


                        current_raw_filepath = os.path.join(

                            VIDEO_FOLDER,

                            f"temp_{timestamp}.mp4"

                        )


                        # ------------------------------------------------
                        # VIDEO WRITER
                        # ------------------------------------------------

                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )


                        video_writer = cv2.VideoWriter(

                            current_raw_filepath,

                            fourcc,

                            fps,

                            (
                                camera_frame.shape[1],
                                camera_frame.shape[0]
                            )

                        )


                        # ------------------------------------------------
                        # WRITE BUFFER
                        # ------------------------------------------------

                        for buffered_frame in frame_buffer:

                            video_writer.write(
                                buffered_frame
                            )


                        recording = True


                        recording_end_time = (

                            current_time
                            + RECORD_AFTER_SECONDS

                        )


                        last_saved_time = (
                            current_time
                        )


                        print()

                        print(
                            "🎥 Recording emergency video..."
                        )

                        print(
                            "File:",
                            current_final_filepath
                        )


            # ------------------------------------------------
            # SIGN 2 NOT VALID
            #
            # Verification timer restarts from zero.
            # ------------------------------------------------

            else:

                emergency_start_time = None


        # ====================================================
        # RECORDING
        #
        # Independent of the emergency display state. Clearing
        # the EMERGENCY indicator never stops this block.
        # ====================================================

        if recording:

            if video_writer is not None:

                video_writer.write(
                    camera_frame
                )


            if (

                recording_end_time is not None

                and

                current_time
                >= recording_end_time

            ):

                print()

                print(
                    "⏹ Recording finished."
                )


                recording_success = (
                    finalize_recording(

                        video_writer,

                        current_raw_filepath,

                        current_final_filepath

                    )
                )


                video_writer = None

                recording = False

                recording_end_time = None


                # ------------------------------------------------
                # SEND ALERT AFTER VIDEO IS SAVED
                # ------------------------------------------------

                if recording_success:

                    send_emergency_alert(
                        current_video_filename
                    )


                current_video_filename = None

                current_raw_filepath = None

                current_final_filepath = None


        # ====================================================
        # CREATE NEXA UI
        # ====================================================

        frame = np.full(

            (700, 1200, 3),

            BG,

            dtype=np.uint8

        )


        # ====================================================
        # HEADER
        # ====================================================

        text(
            frame,
            "NEXA",
            (30, 45),
            1.1,
            CYAN,
            2
        )

        text(
            frame,
            "EMERGENCY AI",
            (120, 43),
            0.65,
            WHITE,
            1
        )

        cv2.circle(
            frame,
            (1090, 32),
            7,
            GREEN,
            -1
        )

        text(
            frame,
            "SYSTEM ONLINE",
            (1110, 38),
            0.5,
            GREEN,
            1
        )

        cv2.line(
            frame,
            (30, 65),
            (1170, 65),
            (50, 60, 70),
            1
        )


        # ====================================================
        # CAMERA PANEL
        # ====================================================

        panel(
            frame,
            30,
            90,
            930,
            640
        )

        frame[
            40:640,
            30:930
        ] = camera_frame

        cv2.rectangle(
            frame,
            (30, 90),
            (930, 640),
            CYAN,
            1
        )


        # ====================================================
        # RIGHT PANEL
        # ====================================================

        panel(
            frame,
            950,
            90,
            1170,
            640
        )

        text(
            frame,
            "DETECTION",
            (970, 125),
            0.55,
            CYAN,
            1
        )


        # ====================================================
        # STATUS
        #
        # SAFE is shown whenever Sign 1 is not currently being
        # shown and no sequence/verification/emergency is
        # actively being handled. With no hand in view the
        # status is also SAFE, unless an emergency was verified
        # or a recording is in progress.
        # ====================================================

        if emergency_verified:

            status = "EMERGENCY"

            status_color = RED


        elif (

            sequence_state == SEQUENCE_SIGN1_DETECTED

            and

            emergency_start_time is not None

        ):

            status = "VERIFYING SIGN 2"

            status_color = ORANGE


        elif not hand_detected and not recording:

            status = "SAFE"

            status_color = GREEN


        elif sequence_state == SEQUENCE_SIGN1_DETECTED:

            if sign1_detected:

                status = "SIGN 1 DETECTED"

            else:

                status = "WAITING SIGN 2"

            status_color = ORANGE


        elif sign1_detected:

            status = "SIGN 1"

            status_color = ORANGE


        else:

            status = "SAFE"

            status_color = GREEN


        text(
            frame,
            status,
            (970, 175),
            0.72,
            status_color,
            2
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        text(
            frame,
            "MODEL CONFIDENCE",
            (970, 220),
            0.42,
            GRAY,
            1
        )

        text(
            frame,
            f"{probability * 100:.1f}%",
            (970, 255),
            0.85,
            WHITE,
            2
        )


        bar_x = 970

        bar_y = 275

        bar_width = 170

        bar_height = 12


        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (50, 55, 65),
            -1
        )


        filled = int(
            bar_width
            * min(probability, 1)
        )


        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (
                bar_x + filled,
                bar_y + bar_height
            ),
            status_color,
            -1
        )


        # ====================================================
        # SEQUENCE / VERIFICATION
        # ====================================================

        text(
            frame,
            "SEQUENCE",
            (970, 330),
            0.45,
            GRAY,
            1
        )


        # ----------------------------------------------------
        # WAITING FOR SIGN 1
        # ----------------------------------------------------

        if sequence_state == SEQUENCE_WAITING:

            text(
                frame,
                "WAITING FOR SIGN 1",
                (970, 365),
                0.50,
                GRAY,
                1
            )

            text(
                frame,
                "OPEN 4 + THUMB FOLDED",
                (970, 395),
                0.38,
                GRAY,
                1
            )


        # ----------------------------------------------------
        # EMERGENCY VERIFIED (COUNTDOWN TO RESET)
        # ----------------------------------------------------

        elif emergency_verified:

            if emergency_verified_time is not None:

                reset_remaining = max(

                    EMERGENCY_DISPLAY_TIME
                    - (
                        time.time()
                        - emergency_verified_time
                    ),

                    0.0

                )

            else:

                reset_remaining = 0.0


            text(
                frame,
                "EMERGENCY VERIFIED",
                (970, 365),
                0.50,
                RED,
                1
            )

            text(
                frame,
                f"RESET IN {reset_remaining:.1f}s",
                (970, 395),
                0.42,
                WHITE,
                1
            )


        # ----------------------------------------------------
        # SIGN 1 DETECTED
        # ----------------------------------------------------

        else:

            if sequence_start_time is not None:

                sequence_elapsed = (

                    time.time()
                    - sequence_start_time

                )

            else:

                sequence_elapsed = 0


            sequence_display = min(

                sequence_elapsed,

                SEQUENCE_TIMEOUT

            )


            # ------------------------------------------------
            # SIGN 2 VERIFYING
            # ------------------------------------------------

            if emergency_start_time is not None:

                text(
                    frame,
                    "SIGN 2 VERIFYING",
                    (970, 365),
                    0.52,
                    ORANGE,
                    1
                )


                verify_elapsed = (

                    time.time()
                    - emergency_start_time

                )


                verify_display = min(

                    verify_elapsed,

                    REQUIRED_TIME

                )


                text(
                    frame,
                    f"{verify_display:.1f} / "
                    f"{REQUIRED_TIME:.1f}s",
                    (970, 400),
                    0.65,
                    WHITE,
                    1
                )


                cv2.rectangle(
                    frame,
                    (970, 415),
                    (1140, 427),
                    (50, 55, 65),
                    -1
                )


                verify_progress = min(

                    verify_display
                    / REQUIRED_TIME,

                    1.0

                )


                verify_filled = int(

                    170
                    * verify_progress

                )


                cv2.rectangle(
                    frame,
                    (970, 415),
                    (
                        970 + verify_filled,
                        427
                    ),
                    ORANGE,
                    -1
                )


            # ------------------------------------------------
            # WAITING FOR SIGN 2
            # ------------------------------------------------

            else:

                text(
                    frame,
                    "✓ SIGN 1 DETECTED",
                    (970, 365),
                    0.50,
                    GREEN,
                    1
                )

                text(
                    frame,
                    "WAITING FOR SIGN 2",
                    (970, 395),
                    0.42,
                    WHITE,
                    1
                )

                text(
                    frame,
                    f"{sequence_display:.1f} / "
                    f"{SEQUENCE_TIMEOUT:.1f}s",
                    (970, 425),
                    0.55,
                    WHITE,
                    1
                )


                cv2.rectangle(
                    frame,
                    (970, 440),
                    (1140, 452),
                    (50, 55, 65),
                    -1
                )


                sequence_progress = min(

                    sequence_display
                    / SEQUENCE_TIMEOUT,

                    1.0

                )


                sequence_filled = int(

                    170
                    * sequence_progress

                )


                cv2.rectangle(
                    frame,
                    (970, 440),
                    (
                        970 + sequence_filled,
                        452
                    ),
                    ORANGE,
                    -1
                )


        # ====================================================
        # SYSTEM STATUS
        # ====================================================

        text(
            frame,
            "SYSTEM",
            (970, 480),
            0.45,
            GRAY,
            1
        )


        systems = [

            ("CAMERA", 510),

            ("AI MODEL", 540),

            ("HAND AI", 570),

            ("SEQUENCE", 600)

        ]


        for name, y in systems:

            cv2.circle(
                frame,
                (980, y),
                5,
                GREEN,
                -1
            )

            text(
                frame,
                name,
                (995, y + 5),
                0.42,
                WHITE,
                1
            )

            text(
                frame,
                "ONLINE",
                (1090, y + 5),
                0.38,
                GREEN,
                1
            )


        # ====================================================
        # RECORDING STATUS
        # ====================================================

        if recording:

            cv2.rectangle(
                frame,
                (970, 610),
                (1150, 640),
                RED,
                2
            )

            text(
                frame,
                "● RECORDING",
                (980, 632),
                0.43,
                RED,
                2
            )


        elif emergency_verified:

            cv2.rectangle(
                frame,
                (970, 610),
                (1150, 640),
                RED,
                2
            )

            text(
                frame,
                "ALERT TRIGGERED",
                (980, 632),
                0.40,
                RED,
                2
            )


        # ====================================================
        # FOOTER
        # ====================================================

        text(
            frame,
            "AI-Based Emergency Hand Sign Detection",
            (30, 680),
            0.4,
            GRAY,
            1
        )

        text(
            frame,
            "Q / ESC  •  EXIT",
            (1060, 680),
            0.4,
            GRAY,
            1
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if (

            key == ord("q")

            or key == ord("Q")

            or key == 27

        ):

            print()

            print(
                "🛑 Exit requested."
            )

            break


# ============================================================
# CTRL+C HANDLING
# ============================================================

except KeyboardInterrupt:

    print()

    print(
        "🛑 Ctrl+C detected."
    )


except Exception as e:

    print()

    print(
        "❌ Unexpected error:"
    )

    print(e)

    traceback.print_exc()


# ============================================================
# CLEANUP
# ============================================================

finally:

    print()

    print(
        "Cleaning up..."
    )


    # --------------------------------------------------------
    # RELEASE VIDEO WRITER
    # --------------------------------------------------------

    if video_writer is not None:

        video_writer.release()

        video_writer = None


    # --------------------------------------------------------
    # RELEASE CAMERA
    # --------------------------------------------------------

    cap.release()


    # --------------------------------------------------------
    # CLOSE WINDOWS
    # --------------------------------------------------------

    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # CLOSE MEDIAPIPE
    # --------------------------------------------------------

    hand_landmarker.close()


    print(
        "Camera released."
    )

    print(
        "OpenCV windows closed."
    )

    print(
        "MediaPipe closed."
    )

    print()

    print(
        "System stopped."
    )
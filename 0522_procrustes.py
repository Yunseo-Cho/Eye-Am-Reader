import cv2
import numpy as np
import time
import winsound
import mediapipe as mp

calibration_data = {"left_top": None, "right_bottom": None}
book_bbox = None
TRIGGER_REQUIRED_FRAMES = 10
base_face_pts = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def get_pupil_center(gray_eye_roi):
    equalized = cv2.equalizeHist(gray_eye_roi)
    blurred = cv2.GaussianBlur(equalized, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def get_face_reference_points(landmarks, iw, ih):
    return np.array([
        (int(landmarks[1].x * iw), int(landmarks[1].y * ih)),
        (int(landmarks[33].x * iw), int(landmarks[33].y * ih)),
        (int(landmarks[263].x * iw), int(landmarks[263].y * ih)),
        (int(landmarks[13].x * iw), int(landmarks[13].y * ih)),
    ], dtype=np.float32)

def normalize_pupil_with_landmarks(pupil, base_pts, current_pts):
    if pupil is None or base_pts is None or current_pts is None:
        return None

    base_mean = np.mean(base_pts, axis=0)
    curr_mean = np.mean(current_pts, axis=0)
    base_centered = base_pts - base_mean
    curr_centered = current_pts - curr_mean

    base_norm = np.linalg.norm(base_centered)
    curr_norm = np.linalg.norm(curr_centered)
    if base_norm == 0 or curr_norm == 0:
        return None
    base_scaled = base_centered / base_norm
    curr_scaled = curr_centered / curr_norm

    U, _, Vt = np.linalg.svd(curr_scaled.T @ base_scaled)
    R = (U @ Vt).T

    scale = base_norm / curr_norm
    transform_matrix = scale * R

    transformed = transform_matrix @ (np.array(pupil) - curr_mean).T + base_mean
    return tuple(transformed.astype(int))

def is_gaze_on_trigger_zone(pupil, bbox, margin_bottom=0.25, margin_left=0.25, margin_right=0.25):
    if bbox is None or pupil is None:
        return None
    x, y = pupil
    w = bbox["right"] - bbox["left"]
    h = bbox["bottom"] - bbox["top"]
    
    if y < bbox["bottom"] - 0.35 * (bbox["bottom"] - bbox["top"]):
        return None


    if x < bbox["left"] + margin_left * w and y > bbox["bottom"] - margin_bottom * h:
        return "BOTTOM_RIGHT"
    elif x > bbox["right"] - margin_right * w and y > bbox["bottom"] - margin_bottom * h:
        return "BOTTOM_LEFT"
    return None


def compute_book_bbox(pt1, pt2):
    return {
        "left": min(pt1[0], pt2[0]),
        "right": max(pt1[0], pt2[0]),
        "top": min(pt1[1], pt2[1]),
        "bottom": max(pt1[1], pt2[1]),
    }

def draw_trigger_zones(frame, bbox, margin_bottom=0.15, margin_left=0.25, margin_right=0.25):
    if not bbox:
        return
    w = bbox["right"] - bbox["left"]
    h = bbox["bottom"] - bbox["top"]

    left_box = (
        (int(bbox["left"]), int(bbox["bottom"] - h * margin_bottom)),
        (int(bbox["left"] + w * margin_left), bbox["bottom"])
    )
    right_box = (
        (int(bbox["right"] - w * margin_right), int(bbox["bottom"] - h * margin_bottom)),
        (int(bbox["right"]), bbox["bottom"])
    )

    cv2.rectangle(frame, left_box[0], left_box[1], (0, 0, 255), 2)
    cv2.rectangle(frame, right_box[0], right_box[1], (255, 0, 0), 2)

mode = "CALIBRATING"
triggered = False
calib_frames = 0
left_top_accum = []
right_bottom_accum = []
prev_pupil_global = None
alpha = 0.1
last_zone = None
same_zone_count = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    pupil_global = None
    current_face_pts = None

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        current_face_pts = get_face_reference_points(landmarks, iw, ih)
        eye_pupils = []

        for eye_idx in [RIGHT_EYE_IDX, LEFT_EYE_IDX]:
            eye_pts = np.array([(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in eye_idx])
            x, y, w_roi, h_roi = cv2.boundingRect(eye_pts)

            pad_w = int(w_roi * 1.8)
            pad_h = int(h_roi * 1.8)

            cx, cy = x + w_roi // 2, y + h_roi // 2
            x1 = max(cx - pad_w // 2, 0)
            y1 = max(cy - pad_h // 2, 0)
            x2 = min(cx + pad_w // 2, iw)
            y2 = min(cy + pad_h // 2, ih)

            zoomed = frame[y1:y2, x1:x2]
            if zoomed.size == 0:
                continue

            eye_gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
            pupil = get_pupil_center(eye_gray)
            if pupil:
                scale_x = iw / (x2 - x1)
                scale_y = ih / (y2 - y1)
                global_pupil = (int(pupil[0] * scale_x), int(pupil[1] * scale_y))
                eye_pupils.append(global_pupil)

        if len(eye_pupils) == 2:
            raw_pupil = (
                (eye_pupils[0][0] + eye_pupils[1][0]) // 2,
                (eye_pupils[0][1] + eye_pupils[1][1]) // 2,
            )

            if mode == "CALIBRATING":
                pupil_global = raw_pupil
            elif base_face_pts is not None and current_face_pts is not None:
                norm_pupil = normalize_pupil_with_landmarks(raw_pupil, base_face_pts, current_face_pts)
                if norm_pupil:
                    if prev_pupil_global is None:
                        pupil_global = norm_pupil
                    else:
                        pupil_global = (
                            int(prev_pupil_global[0] * (1 - alpha) + norm_pupil[0] * alpha),
                            int(prev_pupil_global[1] * (1 - alpha) + norm_pupil[1] * alpha)
                        )
                    prev_pupil_global = pupil_global

    if mode == "CALIBRATING":
        if calib_frames < 150:
            cv2.putText(frame, "Look at the TOP-LEFT of the book", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if pupil_global:
                left_top_accum.append(pupil_global)
        elif calib_frames < 300:
            cv2.putText(frame, "Look at the BOTTOM-RIGHT of the book", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if pupil_global:
                right_bottom_accum.append(pupil_global)

        calib_frames += 1

        if calib_frames == 300:
            if len(left_top_accum) < 5 or len(right_bottom_accum) < 5:
                mode = "CALIBRATING"
                calib_frames = 0
                left_top_accum = []
                right_bottom_accum = []
                continue

            calibration_data["left_top"] = tuple(np.mean(left_top_accum, axis=0).astype(int))
            calibration_data["right_bottom"] = tuple(np.mean(right_bottom_accum, axis=0).astype(int))
            book_bbox = compute_book_bbox(calibration_data["left_top"], calibration_data["right_bottom"])
            base_face_pts = current_face_pts.copy()
            mode = "RUNNING"

    elif mode == "RUNNING":
        if book_bbox:
            cv2.rectangle(frame, (book_bbox["left"], book_bbox["top"]),
                          (book_bbox["right"], book_bbox["bottom"]), (0, 255, 0), 2)
            draw_trigger_zones(frame, book_bbox)

        if pupil_global:
            zone = is_gaze_on_trigger_zone(pupil_global, book_bbox)

            if zone == last_zone and zone is not None:
                same_zone_count += 1
            else:
                same_zone_count = 1
                last_zone = zone

            if same_zone_count >= TRIGGER_REQUIRED_FRAMES and not triggered:
                triggered = True
                winsound.Beep(1000, 300)
                cv2.putText(frame, f"Triggered: {zone}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if current_face_pts is not None:
                    base_face_pts = current_face_pts.copy()

            if zone is None:
                same_zone_count = 0
                triggered = False

    frame_idx += 1
    if frame_idx % 60 == 0 and current_face_pts is not None and mode == "RUNNING":
        base_face_pts = current_face_pts.copy()

    cv2.imshow("Eye Tracker (Landmark-Normalized)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


# Shared video helpers

def ffprobe_video_info(path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,codec_name,codec_tag_string",
        "-of", "json", path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    data = json.loads(p.stdout)
    return data["streams"][0]


def parse_ffmpeg_fraction(frac: str) -> float:
    if not frac or frac == "0/0":
        return 0.0
    num, den = frac.split("/")
    num = float(num)
    den = float(den)
    return num / den if den != 0 else 0.0


def transcode_to_cfr(input_path: str, target_fps: int, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"fps={target_fps}",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed:\n{p.stderr}")


def ensure_cfr_input(input_path: str, preferred_fps: int = 60) -> tuple[str, float]:
    info = ffprobe_video_info(input_path)
    avg_fps = parse_ffmpeg_fraction(info.get("avg_frame_rate", "0/0"))
    r_fps = parse_ffmpeg_fraction(info.get("r_frame_rate", "0/0"))
    unreliable = (avg_fps < 1.0) or (abs(avg_fps - r_fps) > 1.0)
    common = [24, 25, 30, 50, 60, 90, 100, 120, 240]
    if avg_fps >= 1.0:
        target = min(common, key=lambda x: abs(x - avg_fps))
    else:
        target = preferred_fps
    must_convert = input_path.lower().endswith(".mov") or unreliable
    if not must_convert:
        return input_path, avg_fps
    tmp_dir = tempfile.mkdtemp(prefix="cfr_")
    cfr_path = os.path.join(tmp_dir, f"input_cfr_{target}.mp4")
    transcode_to_cfr(input_path, target, cfr_path)
    info2 = ffprobe_video_info(cfr_path)
    fps2 = parse_ffmpeg_fraction(info2.get("avg_frame_rate", "0/0"))
    if fps2 < 1.0:
        fps2 = float(target)
    return cfr_path, fps2


def make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    for fourcc_str in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            return out
    raise RuntimeError("Could not open VideoWriter with avc1/H264/mp4v")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 0.0
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = clamp(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def vec_angle_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def normalize_vec(dx: float, dy: float) -> tuple[float, float]:
    n = float(np.hypot(dx, dy)) + 1e-6
    return dx / n, dy / n


def dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def cross_mag2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(abs(a[0] * b[1] - a[1] * b[0]))


def lm_xy(lm, width: int, height: int) -> np.ndarray:
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def lm_vis(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def load_balltrack(balltrack_json_path: str | None) -> dict[int, tuple[int, int]] | None:
    if not balltrack_json_path or not os.path.isfile(balltrack_json_path):
        return None
    with open(balltrack_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", {})
    out: dict[int, tuple[int, int]] = {}
    for k, v in frames.items():
        out[int(k)] = (int(v["cx"]), int(v["cy"]))
    return out


def make_followthrough_curve(start_px: tuple[int, int], ideal_dir: tuple[float, float], width: int, height: int) -> list[tuple[int, int]]:
    """Build a smooth ideal follow-through curve starting from release.

    The curve is a quadratic Bezier whose end point follows the ideal direction,
    with a small downward continuation so the path reads as a body trajectory
    instead of a rigid straight arrow.
    """
    sx, sy = start_px
    dx, dy = ideal_dir
    tangent_len = max(120, int(0.18 * max(width, height)))
    end_len = max(180, int(0.28 * max(width, height)))

    control = np.array([sx + dx * tangent_len, sy + dy * tangent_len], dtype=np.float32)
    end = np.array([sx + dx * end_len, sy + dy * end_len + 0.08 * height], dtype=np.float32)
    start = np.array([sx, sy], dtype=np.float32)

    points: list[tuple[int, int]] = []
    for t in np.linspace(0.0, 1.0, 24):
        p = ((1 - t) ** 2) * start + 2 * (1 - t) * t * control + (t ** 2) * end
        px = int(clamp(float(p[0]), 0, width - 1))
        py = int(clamp(float(p[1]), 0, height - 1))
        points.append((px, py))
    return points


def draw_trajectory(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int, label: str | None = None) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, points[0], thickness + 1, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(frame, points[-2], points[-1], color, thickness, cv2.LINE_AA, tipLength=0.35)
    if label:
        tx = int(points[0][0] + 10)
        ty = int(max(20, points[0][1] - 12))
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


class EventDetector:
    RELEASE_DIST_THRESH = 60
    RELEASE_CONSEC = 3
    RELEASE_VEL_MIN = 5.0

    def __init__(self, fps: float, ball_track: dict[int, tuple[int, int]] | None = None, img_width: int = 1, img_height: int = 1, bowling_arm: str = "right"):
        bowling_arm = (bowling_arm or "right").strip().lower()
        if bowling_arm not in {"left", "right"}:
            bowling_arm = "right"
        self.fps = float(fps)
        self.events = {"BFC": None, "FFC": None, "RELEASE": None}
        self.release_method = None
        self._ball_track = ball_track or {}
        self._img_w = img_width
        self._img_h = img_height
        self._wrist_y: deque[float] = deque(maxlen=5)
        self._la_y: deque[float] = deque(maxlen=7)
        self._ra_y: deque[float] = deque(maxlen=7)
        self._runup_started = False
        self._runup_phase = True
        self._mid_hip_prev = None
        self._hip_v_prev = None
        self._contacts: list[tuple[int, str]] = []
        self._release_consec_count = 0
        self._prev_ball_pos = None
        self.bowling_arm = bowling_arm

    def update_runup_phase(self, frame_idx: int, lh, rh, la, ra) -> None:
        ankle_distance = abs(float(la.x - ra.x))
        if ankle_distance > 0.05:
            self._runup_started = True
        mid_hip = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2], dtype=np.float32)
        if self._mid_hip_prev is not None:
            v = mid_hip - self._mid_hip_prev
            forward_v = float(v[1])
            if self._hip_v_prev is not None and self._runup_phase:
                if abs(self._hip_v_prev) > 0.01 and abs(forward_v) < 0.002:
                    self._runup_phase = False
            self._hip_v_prev = forward_v
        self._mid_hip_prev = mid_hip

    def get_bowling_wrist_and_other(self, landmarks):
        if self.bowling_arm == "left":
            return landmarks[15], landmarks[16], 15
        return landmarks[16], landmarks[15], 16

    def update_release(self, frame_idx: int, landmarks) -> None:
        if self.events["RELEASE"] is not None:
            return

        if self.events["FFC"] is None:
            return

        min_after_ffc = max(1, int(0.04 * self.fps))
        if frame_idx < int(self.events["FFC"]) + min_after_ffc:
            return

        wrist, other_wrist, wrist_idx = self.get_bowling_wrist_and_other(landmarks)
        self._wrist_y.append(float(wrist.y))
        wrist_px = np.array([wrist.x * self._img_w, wrist.y * self._img_h], dtype=np.float32)

        ball = self._ball_track.get(frame_idx)
        if ball is not None:
            ball_px = np.array(ball, dtype=np.float32)
            dist = float(np.linalg.norm(ball_px - wrist_px))

            ball_moving = False
            moving_away_from_wrist = False
            if self._prev_ball_pos is not None:
                disp = ball_px - self._prev_ball_pos
                disp_mag = float(np.linalg.norm(disp))
                ball_moving = disp_mag >= self.RELEASE_VEL_MIN

                prev_dist = float(np.linalg.norm(self._prev_ball_pos - wrist_px))
                moving_away_from_wrist = (dist - prev_dist) > 2.0

            self._prev_ball_pos = ball_px

            other_px = np.array([other_wrist.x * self._img_w, other_wrist.y * self._img_h], dtype=np.float32)
            other_dist = float(np.linalg.norm(ball_px - other_px))
            wrist_in_release_zone = float(wrist.y) < 0.58
            correct_arm = dist < other_dist

            if dist > self.RELEASE_DIST_THRESH and ball_moving and moving_away_from_wrist and wrist_in_release_zone and correct_arm:
                self._release_consec_count += 1
            else:
                self._release_consec_count = 0

            if self._release_consec_count >= self.RELEASE_CONSEC:
                self.events["RELEASE"] = frame_idx - self.RELEASE_CONSEC + 1
                self.release_method = f"ball_wrist_{self.bowling_arm}"
                return

        if len(self._wrist_y) < 5:
            return
        y0 = self._wrist_y[0]
        y4 = self._wrist_y[-1]
        vel = y4 - y0
        if y4 < 0.50 and vel > 0.06:
            self.events["RELEASE"] = frame_idx
            self.release_method = f"wrist_kinematic_{self.bowling_arm}"

    def _ankle_contact_candidate(self, y_hist: deque[float]) -> bool:
        if len(y_hist) < 7:
            return False
        ys = np.array(y_hist, dtype=np.float32)
        v = np.diff(ys)
        recent = v[-3:]
        return float(np.max(np.abs(recent))) < 0.0015

    def update_foot_contacts(self, frame_idx: int, la, ra) -> None:
        self._la_y.append(float(la.y))
        self._ra_y.append(float(ra.y))
        if self._runup_phase:
            return
        if self.events["BFC"] is None:
            if self._ankle_contact_candidate(self._la_y):
                self.events["BFC"] = frame_idx
                self._contacts.append((frame_idx, "L"))
                return
            if self._ankle_contact_candidate(self._ra_y):
                self.events["BFC"] = frame_idx
                self._contacts.append((frame_idx, "R"))
                return
        if self.events["BFC"] is not None and self.events["FFC"] is None:
            if frame_idx - self.events["BFC"] < int(0.06 * self.fps):
                return
            if self._ankle_contact_candidate(self._la_y):
                self.events["FFC"] = frame_idx
                self._contacts.append((frame_idx, "L"))
                return
            if self._ankle_contact_candidate(self._ra_y):
                self.events["FFC"] = frame_idx
                self._contacts.append((frame_idx, "R"))
                return

    def update(self, frame_idx: int, landmarks, lh, rh, la, ra) -> dict[str, int | None]:
        self.update_runup_phase(frame_idx, lh, rh, la, ra)
        self.update_foot_contacts(frame_idx, la, ra)
        self.update_release(frame_idx, landmarks)
        return self.events

    def contact_side_for(self, event_name: str) -> str | None:
        f = self.events.get(event_name)
        if f is None:
            return None
        for fr, side in self._contacts:
            if fr == f:
                return side
        return None


def drills_and_feedback(metrics: dict[str, Any]) -> dict[str, list[str]]:
    feedback = []
    drills = []
    lat = metrics.get("lateral_flexion_release_deg")
    if lat is not None:
        if lat >= 50:
            feedback.append("Excessive lateral flexion at release; consider reducing side-bend to lower back load.")
            drills += ["Tall-posture shadow bowling", "Side plank (trunk control)", "Alignment stick drill"]
        elif lat >= 40:
            feedback.append("Moderate lateral flexion at release; could improve upright posture for efficiency.")
            drills += ["Wall-lean posture drill", "Core anti-lateral-flexion holds"]
    sep = metrics.get("hip_shoulder_separation_ffc_deg")
    if sep is not None and sep < 25:
        feedback.append("Low hip-shoulder separation at FFC; may be limiting trunk-driven pace.")
        drills += ["Separation shadow reps (pause at FFC)", "Medicine ball rotational throws"]
    elbow = metrics.get("elbow_extension_change_deg")
    if elbow is not None and elbow > 20:
        feedback.append("Large elbow extension change; possible higher chucking risk. Consider checking with a qualified coach.")
        drills += ["Straight-arm band drill (light)", "Mirror work for arm path consistency"]

    ft = metrics.get("follow_through_label")
    if ft == "bad":
        feedback.append("After release, body weight appears to go sideways instead of transferring through the target line.")
        drills += [
            "Bound-and-hold follow-through drill",
            "Straight-line follow-through walk-throughs",
            "Front-leg bracing with chest-over-knee drill",
        ]
    elif ft == "moderate":
        feedback.append("Follow-through is slightly off line after release; aim to continue momentum more directly through the target.")
        drills += ["Target-line follow-through drill", "Step-through bowling drill"]
    elif ft == "good":
        feedback.append("Follow-through direction looks efficient after release, with weight transferring well in the delivery direction.")

    if not feedback:
        feedback.append("No major red flags detected on these metrics. Continue building consistency and strength.")
    drills_unique = list(dict.fromkeys(drills))[:6]
    return {"feedback": feedback, "drills": drills_unique}


def analyze_video(
    video_path: str,
    output_dir: str,
    balltrack_json_path: str | None = None,
    bowling_arm: str = "right",
    preferred_fps: int = 60,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    video_path = str(Path(video_path))
    stem = Path(video_path).stem
    output_path = os.path.join(output_dir, f"analysed_{stem}.mp4")

    cfr_input_path, fps = ensure_cfr_input(video_path, preferred_fps=preferred_fps)
    ball_track = load_balltrack(balltrack_json_path)
    cap = cv2.VideoCapture(cfr_input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfr_input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = make_writer(output_path, float(fps), width, height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    angle_buffer: deque[float] = deque(maxlen=7)
    crossover_count = 0
    crossover_frames: list[int] = []
    RUNUP_THRESHOLD = 0.02
    crossed_prev = False
    current_frame = 0
    written = 0

    event_detector = EventDetector(fps=float(fps), ball_track=ball_track, img_width=width, img_height=height, bowling_arm=bowling_arm)
    events: dict[str, int | None] = {"BFC": None, "FFC": None, "RELEASE": None}

    spine_tilt_by_frame: dict[int, float] = {}
    spine_tilt_smoothed_by_frame: dict[int, float] = {}
    hip_shoulder_sep_by_frame: dict[int, float] = {}
    shoulder_line_angle_by_frame: dict[int, float] = {}
    hip_line_angle_by_frame: dict[int, float] = {}
    elbow_angle_by_frame: dict[int, float] = {}
    landmarks_snapshot: dict[int, dict[str, Any]] = {}
    midhip_y_by_frame: dict[int, float] = {}
    la_x_by_frame: dict[int, float] = {}
    ra_x_by_frame: dict[int, float] = {}
    la_y_by_frame: dict[int, float] = {}
    ra_y_by_frame: dict[int, float] = {}
    follow_through_alignment_by_frame: dict[int, float] = {}
    follow_through_sideways_error_by_frame: dict[int, float] = {}
    release_body_center: tuple[float, float] | None = None
    release_body_center_px: tuple[int, int] | None = None
    followthrough_locked_label: str | None = None
    followthrough_locked_score: float | None = None
    followthrough_sideways_error: float | None = None
    followthrough_eval_done = False
    followthrough_direction_vector: tuple[float, float] | None = None
    followthrough_ideal_curve: list[tuple[int, int]] = []
    post_release_positions: list[tuple[int, tuple[float, float]]] = []

    metrics: dict[str, Any] = {"fps": float(fps), "events": {}, "bowling_arm": bowling_arm}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        right_cross = False
        left_cross = False
        raw_angle = None

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = result.pose_landmarks.landmark
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            bowling_arm_norm = bowling_arm.strip().lower()
            if bowling_arm_norm == "left":
                rw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                re = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            else:
                rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            lheel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
            rheel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]
            lfoot = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            rfoot = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            events = event_detector.update(current_frame, lm, lh, rh, la, ra)
            release_frame = events.get("RELEASE")

            mid_hip_y = float((lh.y + rh.y) / 2)
            midhip_y_by_frame[current_frame] = mid_hip_y
            la_x_by_frame[current_frame] = float(la.x)
            ra_x_by_frame[current_frame] = float(ra.x)
            la_y_by_frame[current_frame] = float(la.y)
            ra_y_by_frame[current_frame] = float(ra.y)

            if event_detector._runup_started and event_detector._runup_phase:
                mid_hip_x = (lh.x + rh.x) / 2
                if ra.x < (mid_hip_x - RUNUP_THRESHOLD):
                    right_cross = True
                if la.x > (mid_hip_x + RUNUP_THRESHOLD):
                    left_cross = True

            mid_shoulder = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
            mid_hip = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
            spine_vector = mid_shoulder - mid_hip
            vertical_vector = np.array([0, -1])
            norm_spine = np.linalg.norm(spine_vector)
            if norm_spine > 1e-6:
                cos_theta = np.dot(spine_vector, vertical_vector) / (norm_spine * np.linalg.norm(vertical_vector))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                raw_angle = float(np.degrees(np.arccos(cos_theta)))
            else:
                raw_angle = 0.0
            spine_tilt_by_frame[current_frame] = raw_angle

            shoulder_vec = np.array([rs.x - ls.x, rs.y - ls.y], dtype=np.float32)
            hip_vec = np.array([rh.x - lh.x, rh.y - lh.y], dtype=np.float32)
            shoulder_ang = vec_angle_deg(shoulder_vec)
            hip_ang = vec_angle_deg(hip_vec)
            shoulder_line_angle_by_frame[current_frame] = shoulder_ang
            hip_line_angle_by_frame[current_frame] = hip_ang
            sep = abs(shoulder_ang - hip_ang)
            sep = min(sep, 360 - sep)
            hip_shoulder_sep_by_frame[current_frame] = float(sep)

            rw_px = lm_xy(rw, width, height)
            re_px = lm_xy(re, width, height)
            rs_px = lm_xy(rs, width, height)
            elbow_ang = angle_deg(rs_px, re_px, rw_px)
            elbow_angle_by_frame[current_frame] = float(elbow_ang)

            if events.get("BFC") == current_frame or events.get("FFC") == current_frame or events.get("RELEASE") == current_frame:
                landmarks_snapshot[current_frame] = {
                    "ls": {"x": float(ls.x), "y": float(ls.y), "v": lm_vis(ls)},
                    "rs": {"x": float(rs.x), "y": float(rs.y), "v": lm_vis(rs)},
                    "lh": {"x": float(lh.x), "y": float(lh.y), "v": lm_vis(lh)},
                    "rh": {"x": float(rh.x), "y": float(rh.y), "v": lm_vis(rh)},
                    "la": {"x": float(la.x), "y": float(la.y), "v": lm_vis(la)},
                    "ra": {"x": float(ra.x), "y": float(ra.y), "v": lm_vis(ra)},
                    "lheel": {"x": float(lheel.x), "y": float(lheel.y), "v": lm_vis(lheel)},
                    "rheel": {"x": float(rheel.x), "y": float(rheel.y), "v": lm_vis(rheel)},
                    "lfoot": {"x": float(lfoot.x), "y": float(lfoot.y), "v": lm_vis(lfoot)},
                    "rfoot": {"x": float(rfoot.x), "y": float(rfoot.y), "v": lm_vis(rfoot)},
                }

            if events.get("BFC") == current_frame:
                cv2.putText(frame, "BFC", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)
            if events.get("FFC") == current_frame:
                cv2.putText(frame, "FFC", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)
            if events.get("RELEASE") == current_frame:
                cv2.putText(frame, "RELEASE", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)

            if release_frame is not None and current_frame == release_frame:
                release_body_center = (float((lh.x + rh.x) / 2), float((lh.y + rh.y) / 2))
                release_body_center_px = (int(release_body_center[0] * width), int(release_body_center[1] * height))
                b_now = ball_track.get(current_frame) if ball_track is not None else None
                b_next = ball_track.get(current_frame + 1) if ball_track is not None else None
                b_prev = ball_track.get(current_frame - 1) if ball_track is not None else None
                if b_now is not None and b_next is not None:
                    ideal_dx = float(b_next[0] - b_now[0])
                    ideal_dy = float(b_next[1] - b_now[1])
                elif b_prev is not None and b_now is not None:
                    ideal_dx = float(b_now[0] - b_prev[0])
                    ideal_dy = float(b_now[1] - b_prev[1])
                else:
                    ideal_dx = 0.0
                    ideal_dy = 1.0
                followthrough_direction_vector = normalize_vec(ideal_dx, ideal_dy)
                followthrough_ideal_curve = make_followthrough_curve(release_body_center_px, followthrough_direction_vector, width, height)
                post_release_positions = []
                followthrough_eval_done = False
                followthrough_locked_label = None
                followthrough_locked_score = None
                followthrough_sideways_error = None

            if raw_angle is not None and (release_frame is None or current_frame <= release_frame):
                angle_buffer.append(raw_angle)
                smoothed_angle = sum(angle_buffer) / len(angle_buffer)
                spine_tilt_smoothed_by_frame[current_frame] = float(smoothed_angle)

                if smoothed_angle < 40:
                    feedback = "Good alignment"
                    color = (0, 255, 0)
                elif 40 <= smoothed_angle < 50:
                    feedback = "Moderate lateral flexion"
                    color = (0, 165, 255)
                else:
                    feedback = "Excessive lateral flexion - Injury Risk"
                    color = (0, 0, 255)

                angle_text = f"Spine Tilt: {smoothed_angle:.1f} deg | {feedback}"
                cv2.putText(frame, angle_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)

            elif release_frame is not None and current_frame > release_frame:
                current_center = (float((lh.x + rh.x) / 2), float((lh.y + rh.y) / 2))
                if release_body_center is not None:
                    post_release_positions.append((current_frame, current_center))

                if (
                    release_body_center is not None
                    and followthrough_direction_vector is not None
                    and not followthrough_eval_done
                    and len(post_release_positions) >= 6
                ):
                    _, end_center = post_release_positions[-1]
                    move_dx = float(end_center[0] - release_body_center[0])
                    move_dy = float(end_center[1] - release_body_center[1])
                    actual_dir = normalize_vec(move_dx, move_dy)
                    ideal_dir = followthrough_direction_vector

                    alignment = dot2(actual_dir, ideal_dir)
                    sideways_error = cross_mag2(actual_dir, ideal_dir)

                    followthrough_locked_score = float(alignment)
                    followthrough_sideways_error = float(sideways_error)
                    follow_through_alignment_by_frame[current_frame] = float(alignment)
                    follow_through_sideways_error_by_frame[current_frame] = float(sideways_error)

                    if alignment >= 0.85 and sideways_error <= 0.25:
                        followthrough_locked_label = "good"
                    elif alignment >= 0.60 and sideways_error <= 0.50:
                        followthrough_locked_label = "moderate"
                    else:
                        followthrough_locked_label = "bad"

                    followthrough_eval_done = True

                if followthrough_locked_label == "good":
                    ft_feedback = "Follow-through direction is good"
                    ft_color = (0, 255, 0)
                elif followthrough_locked_label == "moderate":
                    ft_feedback = "Follow-through slightly off line"
                    ft_color = (0, 165, 255)
                elif followthrough_locked_label == "bad":
                    ft_feedback = "Follow-through going sideways"
                    ft_color = (0, 0, 255)
                else:
                    ft_feedback = "Evaluating follow-through..."
                    ft_color = (255, 255, 255)

                cv2.putText(frame, f"Follow-through: {ft_feedback}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ft_color, 3, cv2.LINE_AA)

                if release_body_center_px is not None and followthrough_ideal_curve:
                    draw_trajectory(frame, followthrough_ideal_curve, (0, 255, 0), 3, "Ideal trajectory")

                if release_body_center_px is not None and len(post_release_positions) >= 2:
                    actual_curve = [release_body_center_px]
                    for _, pos in post_release_positions:
                        actual_curve.append((int(pos[0] * width), int(pos[1] * height)))
                    # Keep the trajectory anchored at the release point.
                    # Downsample long trails, but never drop the first point.
                    if len(actual_curve) > 18:
                        interior = actual_curve[1:]
                        step = max(1, len(interior) // 16)
                        sampled = interior[::step]
                        actual_curve = [release_body_center_px] + sampled[-16:]
                    draw_trajectory(frame, actual_curve, (0, 0, 255), 3, "Actual trajectory")

        crossed_now = right_cross or left_cross
        if crossed_now and not crossed_prev:
            crossover_count += 1
            crossover_frames.append(current_frame)
        crossed_prev = crossed_now

        if crossover_count > 5:
            cv2.putText(frame, "Run-up Issue: Leg Cross-Over Detected", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        out.write(frame)
        written += 1
        current_frame += 1

    cap.release()
    out.release()
    pose.close()

    events = event_detector.events
    events_out = {}
    for k, v in events.items():
        if v is not None:
            ev = {"frame": int(v), "time_s": float(v / fps)}
            if k == "RELEASE" and event_detector.release_method:
                ev["method"] = event_detector.release_method
            events_out[k] = ev
    metrics["events"] = events_out
    metrics["balltrack_loaded"] = ball_track is not None

    def sample_at(frame_idx: int | None, series: dict[int, float]) -> float | None:
        if frame_idx is None:
            return None
        return float(series.get(frame_idx)) if frame_idx in series else None

    rel = events.get("RELEASE")
    ffc = events.get("FFC")
    bfc = events.get("BFC")

    metrics["lateral_flexion_bfc_deg"] = sample_at(bfc, spine_tilt_by_frame)
    metrics["lateral_flexion_ffc_deg"] = sample_at(ffc, spine_tilt_by_frame)
    metrics["lateral_flexion_release_deg"] = sample_at(rel, spine_tilt_by_frame)
    metrics["lateral_flexion_bfc_smoothed_deg"] = sample_at(bfc, spine_tilt_smoothed_by_frame)
    metrics["lateral_flexion_ffc_smoothed_deg"] = sample_at(ffc, spine_tilt_smoothed_by_frame)
    metrics["lateral_flexion_release_smoothed_deg"] = sample_at(rel, spine_tilt_smoothed_by_frame)
    metrics["hip_shoulder_separation_ffc_deg"] = sample_at(ffc, hip_shoulder_sep_by_frame)

    if ffc is not None:
        w = int(0.10 * fps)
        vals = [hip_shoulder_sep_by_frame[i] for i in range(max(0, ffc - w), ffc + w + 1) if i in hip_shoulder_sep_by_frame]
        metrics["hip_shoulder_separation_ffc_peak_deg"] = float(np.max(vals)) if vals else None
    else:
        metrics["hip_shoulder_separation_ffc_peak_deg"] = None

    metrics["shoulder_line_angle_ffc_deg"] = sample_at(ffc, shoulder_line_angle_by_frame)
    metrics["hip_line_angle_ffc_deg"] = sample_at(ffc, hip_line_angle_by_frame)
    metrics["shoulder_line_angle_release_deg"] = sample_at(rel, shoulder_line_angle_by_frame)
    metrics["hip_line_angle_release_deg"] = sample_at(rel, hip_line_angle_by_frame)

    if rel is not None:
        start = max(0, int(ffc)) if ffc is not None else max(0, int(rel - 0.25 * fps))
        end = int(rel)
        angles = [elbow_angle_by_frame[i] for i in range(start, end + 1) if i in elbow_angle_by_frame]
        if angles:
            metrics["elbow_angle_min_deg"] = float(np.min(angles))
            metrics["elbow_angle_max_deg"] = float(np.max(angles))
            metrics["elbow_extension_change_deg"] = float(np.max(angles) - np.min(angles))
        else:
            metrics["elbow_angle_min_deg"] = None
            metrics["elbow_angle_max_deg"] = None
            metrics["elbow_extension_change_deg"] = None
    else:
        metrics["elbow_angle_min_deg"] = None
        metrics["elbow_angle_max_deg"] = None
        metrics["elbow_extension_change_deg"] = None

    metrics["trunk_lean_ffc_smoothed_deg"] = metrics.get("lateral_flexion_ffc_smoothed_deg")
    metrics["trunk_lean_release_smoothed_deg"] = metrics.get("lateral_flexion_release_smoothed_deg")

    metrics["alignment_label"] = None
    if metrics["trunk_lean_release_smoothed_deg"] is not None:
        lean = metrics["trunk_lean_release_smoothed_deg"]
        if lean >= 50:
            metrics["alignment_label"] = "falling_away_high"
        elif lean >= 40:
            metrics["alignment_label"] = "falling_away_moderate"
        else:
            metrics["alignment_label"] = "upright_ok"

    metrics["back_foot_angle_bfc_deg"] = None
    metrics["front_foot_angle_ffc_deg"] = None
    metrics["stride_length_ffc_px"] = None
    metrics["stride_length_ffc_norm"] = None
    metrics["bfc_side"] = None
    metrics["ffc_side"] = None

    if bfc is not None and bfc in landmarks_snapshot:
        snap = landmarks_snapshot[bfc]
        metrics["bfc_side"] = event_detector.contact_side_for("BFC")
        lheel_px = np.array([snap["lheel"]["x"] * width, snap["lheel"]["y"] * height], dtype=np.float32)
        lfoot_px = np.array([snap["lfoot"]["x"] * width, snap["lfoot"]["y"] * height], dtype=np.float32)
        rheel_px = np.array([snap["rheel"]["x"] * width, snap["rheel"]["y"] * height], dtype=np.float32)
        rfoot_px = np.array([snap["rfoot"]["x"] * width, snap["rfoot"]["y"] * height], dtype=np.float32)
        lang = vec_angle_deg(lfoot_px - lheel_px)
        rang = vec_angle_deg(rfoot_px - rheel_px)
        lvis = (snap["lheel"]["v"] + snap["lfoot"]["v"]) / 2
        rvis = (snap["rheel"]["v"] + snap["rfoot"]["v"]) / 2
        metrics["back_foot_angle_bfc_deg"] = float(lang if lvis >= rvis else rang)

    if ffc is not None and ffc in landmarks_snapshot:
        snap = landmarks_snapshot[ffc]
        metrics["ffc_side"] = event_detector.contact_side_for("FFC")
        lheel_px = np.array([snap["lheel"]["x"] * width, snap["lheel"]["y"] * height], dtype=np.float32)
        lfoot_px = np.array([snap["lfoot"]["x"] * width, snap["lfoot"]["y"] * height], dtype=np.float32)
        rheel_px = np.array([snap["rheel"]["x"] * width, snap["rheel"]["y"] * height], dtype=np.float32)
        rfoot_px = np.array([snap["rfoot"]["x"] * width, snap["rfoot"]["y"] * height], dtype=np.float32)
        lang = vec_angle_deg(lfoot_px - lheel_px)
        rang = vec_angle_deg(rfoot_px - rheel_px)
        lvis = (snap["lheel"]["v"] + snap["lfoot"]["v"]) / 2
        rvis = (snap["rheel"]["v"] + snap["rfoot"]["v"]) / 2
        metrics["front_foot_angle_ffc_deg"] = float(lang if lvis >= rvis else rang)
        if ffc in la_x_by_frame and ffc in ra_x_by_frame and ffc in la_y_by_frame and ffc in ra_y_by_frame:
            a = np.array([la_x_by_frame[ffc] * width, la_y_by_frame[ffc] * height], dtype=np.float32)
            b = np.array([ra_x_by_frame[ffc] * width, ra_y_by_frame[ffc] * height], dtype=np.float32)
            metrics["stride_length_ffc_px"] = float(np.linalg.norm(a - b))
            metrics["stride_length_ffc_norm"] = float(np.linalg.norm(np.array([la_x_by_frame[ffc] - ra_x_by_frame[ffc], la_y_by_frame[ffc] - ra_y_by_frame[ffc]], dtype=np.float32)))

    metrics["com_forward_travel_post_release_norm"] = None
    metrics["back_leg_recovery_score"] = None
    metrics["follow_through_alignment"] = None
    metrics["follow_through_sideways_error"] = None
    metrics["follow_through_label"] = None
    if rel is not None:
        post = int(0.5 * fps)
        endf = min(max(0, current_frame - 1), rel + post)
        if rel in midhip_y_by_frame and endf in midhip_y_by_frame:
            metrics["com_forward_travel_post_release_norm"] = float(midhip_y_by_frame[endf] - midhip_y_by_frame[rel])
        ffc_side = event_detector.contact_side_for("FFC")
        if ffc_side is not None and rel in la_x_by_frame and rel in ra_x_by_frame and endf in la_x_by_frame and endf in ra_x_by_frame:
            if ffc_side == "L":
                front_x_rel = la_x_by_frame[rel]
                back_x_rel = ra_x_by_frame[rel]
                back_x_end = ra_x_by_frame[endf]
            else:
                front_x_rel = ra_x_by_frame[rel]
                back_x_rel = la_x_by_frame[rel]
                back_x_end = la_x_by_frame[endf]
            metrics["back_leg_recovery_score"] = float((back_x_end - back_x_rel) / (abs(front_x_rel - back_x_rel) + 1e-6))

        if followthrough_locked_score is not None:
            metrics["follow_through_alignment"] = float(followthrough_locked_score)
        if followthrough_sideways_error is not None:
            metrics["follow_through_sideways_error"] = float(followthrough_sideways_error)
        if followthrough_locked_label is not None:
            metrics["follow_through_label"] = str(followthrough_locked_label)

    metrics["ratings"] = {}
    lat_rel = metrics.get("lateral_flexion_release_smoothed_deg")
    if lat_rel is not None:
        metrics["ratings"]["lateral_flexion_release"] = "excessive" if lat_rel >= 50 else "moderate" if lat_rel >= 40 else "good"
    sep_ffc = metrics.get("hip_shoulder_separation_ffc_peak_deg")
    if sep_ffc is not None:
        metrics["ratings"]["hip_shoulder_separation"] = "low" if sep_ffc < 25 else "medium" if sep_ffc < 40 else "high"
    el_ext = metrics.get("elbow_extension_change_deg")
    if el_ext is not None:
        metrics["ratings"]["elbow_extension_change"] = "high" if el_ext > 20 else "normal"
    ft_label = metrics.get("follow_through_label")
    if ft_label is not None:
        metrics["ratings"]["follow_through"] = "good" if ft_label == "good" else "moderate" if ft_label == "moderate" else "poor"

    coach = drills_and_feedback(metrics)
    report = {
        "input_path": video_path,
        "output_path": output_path,
        "metrics": metrics,
        "coach": coach,
        "crossover_count": crossover_count,
        "crossover_frames": crossover_frames,
    }

    report_path = os.path.join(output_dir, f"analysed_{stem}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if cfr_input_path != video_path:
        shutil.rmtree(os.path.dirname(cfr_input_path), ignore_errors=True)

    return {
        "annotated_video_path": output_path,
        "report_json_path": report_path,
        "metrics": metrics,
        "coach": coach,
        "frames_written": written,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web-safe bowling action analysis")
    parser.add_argument("video_path")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--balltrack-json", default=None)
    parser.add_argument("--bowling-arm", choices=["left", "right"], default=None)
    args = parser.parse_args()

    bowling_arm = args.bowling_arm
    while bowling_arm not in {"left", "right"}:
        bowling_arm = input("Is the bowler left or right handed? Type 'left' or 'right': ").strip().lower()

    result = analyze_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        balltrack_json_path=args.balltrack_json,
        bowling_arm=bowling_arm,
    )
    print(json.dumps(result, indent=2))

"""
BowlFast.AI — back-view fast bowling analysis (single file, spec-aligned).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import os
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# --- Landmarks (spec §3) ---
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
NOSE = 0

COMMON_FPS_TARGETS = (30, 60, 120, 240)
# Every frame: fast run-up + close camera can move the bowler outside a 2-frame skip window.
YOLO_PERSON_EVERY = 1
YOLO_BALL_EVERY = 2
BALL_YOLO_CONF_PRIMARY = 0.5
# Lowered from 0.4 → 0.25: the downstream Kalman + parabola gates are
# strict, so let the model surface more candidates and filter geometrically.
BALL_YOLO_CONF_LOW = 0.40
BALL_YOLO_CONF_FLIGHT = 0.35
BALL_TRACK_FAIL_CONF = 0.4
BALL_YOLO_IMGSZ_FULL = 1280
BALL_YOLO_IMGSZ_ROI = 640
BALL_YOLO_IOU = 0.5
BALL_ROI_FRAC = 0.25
BALL_ROI_MIN_PX = 480
BALL_ROI_EDGE_MARGIN = 16
BALL_ROI_FAIL_LIMIT = 3
BALL_BOOTSTRAP_WRIST_PX = 100.0
BALL_PRE_RELEASE_WRIST_PX = 220.0
# Static false-positive rejection (post-release flight only). A candidate is
# treated as a fixed background blob if YOLO has fired within
# BALL_STATIC_RADIUS_PX of it on at least BALL_STATIC_MIN_FRAMES distinct frames
# inside the rolling window. A genuine in-flight ball moves too fast to cluster.
BALL_STATIC_WINDOW = 90
BALL_STATIC_RADIUS_PX = 12
BALL_STATIC_MIN_FRAMES = 6
BALL_CREASE_Y_FRAC = 0.78
BALL_EDGE_X_FRAC = 0.14
BALL_CREASE_WRIST_EXEMPT_PX = 120.0
# High alpha so the smoothed crop keeps up with sprinting in net / stump-close shots.
EMA_BOX_ALPHA = 0.84
BOX_EXPAND_W = 0.35
BOX_EXPAND_H = 0.28
# FIX 2: Reduced from 2.8 → 2.0 to switch to a clearly better detection sooner.
BOWLER_SCORE_SWITCH_MARGIN = 2.0

_HSV_WHITE_LO = np.array([0, 0, 230], dtype=np.uint8)
_HSV_WHITE_HI = np.array([180, 20, 255], dtype=np.uint8)
_HSV_RED_LO_1 = np.array([0, 110, 80], dtype=np.uint8)
_HSV_RED_HI_1 = np.array([10, 255, 255], dtype=np.uint8)
_HSV_RED_LO_2 = np.array([170, 110, 80], dtype=np.uint8)
_HSV_RED_HI_2 = np.array([180, 255, 255], dtype=np.uint8)
_HSV_PINK_LO = np.array([150, 80, 120], dtype=np.uint8)
_HSV_PINK_HI = np.array([175, 220, 255], dtype=np.uint8)


def parse_ffmpeg_fraction(frac: str) -> float:
    if not frac or frac == "0/0":
        return 0.0
    a, b = frac.split("/")
    nf, df = float(a), float(b)
    return nf / df if df else 0.0


def ffprobe_video_stream(path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json", path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    return json.loads(p.stdout)["streams"][0]


def nearest_common_fps(avg_fps: float) -> int:
    return min(COMMON_FPS_TARGETS, key=lambda t: abs(float(t) - avg_fps))


def transcode_to_cfr(src: str, target_fps: int, dst: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"fps={target_fps}", "-vsync", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        dst,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}")


def ensure_cfr_input(video_path: str) -> tuple[str, float, str | None]:
    path = str(Path(video_path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    info = ffprobe_video_stream(path)
    avg = parse_ffmpeg_fraction(info.get("avg_frame_rate", "0/0"))
    r = parse_ffmpeg_fraction(info.get("r_frame_rate", "0/0"))
    bad = avg < 1.0 or abs(avg - r) > 1.0
    if not (path.lower().endswith(".mov") or bad):
        return path, float(avg), None
    tgt = nearest_common_fps(avg) if avg >= 1.0 else 60
    tmp = tempfile.mkdtemp(prefix="bowlfast_cfr_")
    out = os.path.join(tmp, "cfr.mp4")
    transcode_to_cfr(path, tgt, out)
    info2 = ffprobe_video_stream(out)
    fps2 = parse_ffmpeg_fraction(info2.get("avg_frame_rate", "0/0"))
    return out, float(fps2 if fps2 >= 1.0 else tgt), tmp


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lm_vis(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def _ideal_pitch_forward_dir(
    anchor_x: int,
    anchor_y: int,
    fw: int,
    fh: int,
    *,
    flatness: float = 0.34,
) -> tuple[float, float]:
    """Unit vector straight down the pitch from the release anchor (in-lane).

    Screen y decreases toward the batsman. No run-up diagonal — lateral
    component is always zero. `flatness` is reserved for tip placement in draw.
    """
    del anchor_x, anchor_y, fw, fh, flatness
    return (0.0, -1.0)


def _ideal_arrow_tip(
    anchor_x: int,
    anchor_y: int,
    length_px: int,
    *,
    flatness: float = 0.34,
) -> tuple[int, int]:
    """Screen tip for the green ideal arrow: same lane x, shallow pitch rise."""
    grade = max(0.20, min(0.55, flatness))
    rise = int(length_px * grade)
    return (anchor_x, max(8, anchor_y - rise))


def norm_vec(dx: float, dy: float) -> tuple[float, float]:
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-9:
        return 0.0, 1.0
    return dx / n, dy / n


def angle_between_deg(v: np.ndarray, ref: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v), np.linalg.norm(ref)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v, ref) / (n1 * n2))
    return float(np.degrees(np.arccos(clamp(c, -1.0, 1.0))))


def cross_z(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def rect_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _exclude_runup_nearly_stationary_boxes(
    boxes: list[tuple[int, int, int, int]],
    phase: str,
    prev_box: tuple[int, int, int, int] | None,
    min_shift_px: float = 5.0,
) -> list[tuple[int, int, int, int]]:
    """Drop candidates whose centroid barely moved vs prev (WK/batter); keep all if empty."""
    if phase != "RUN_UP" or prev_box is None or not boxes:
        return boxes
    pcx, pcy = rect_center(prev_box)
    out = [
        b
        for b in boxes
        if math.hypot(rect_center(b)[0] - pcx, rect_center(b)[1] - pcy)
        >= min_shift_px
    ]
    return out if out else boxes


def clamp_box(
    box: tuple[int, int, int, int], fw: int, fh: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
    if x2 <= x1 + 4:
        x2 = min(fw - 1, x1 + 5)
    if y2 <= y1 + 4:
        y2 = min(fh - 1, y1 + 5)
    return x1, y1, x2, y2


def make_search_gate(
    prev_box: tuple[int, int, int, int],
    fw: int,
    fh: int,
    lost_frames: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = prev_box
    cx, cy = rect_center(prev_box)
    bw, bh = x2 - x1, y2 - y1
    # FIX 5: Widened base scale from 1.75 → 2.1 so large strides don't
    # push the bowler outside the gate and force a wrong candidate.
    scale = 2.1 + 0.2 * min(12, lost_frames)
    gx1 = int(cx - 0.5 * bw * scale)
    gx2 = int(cx + 0.5 * bw * scale)
    gy1 = int(cy - 0.55 * bh * scale)
    gy2 = int(cy + 0.65 * bh * scale)
    return clamp_box((gx1, gy1, gx2, gy2), fw, fh)


def box_inside_gate(
    box: tuple[int, int, int, int], gate: tuple[int, int, int, int]
) -> bool:
    cx, cy = rect_center(box)
    gx1, gy1, gx2, gy2 = gate
    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


def _box_norm_center(
    box: tuple[int, int, int, int], fw: int, fh: int
) -> tuple[float, float]:
    cx, cy = rect_center(box)
    return cx / max(1.0, float(fw)), cy / max(1.0, float(fh))


def _on_entry_side(nx: float, entry_side: str) -> bool:
    if entry_side == "left":
        return nx < 0.52
    return nx > 0.48


def _in_umpire_corral(nx: float, ny: float) -> bool:
    """Stumps / umpire stand — centre of frame, mid height."""
    return 0.34 <= nx <= 0.66 and 0.18 <= ny <= 0.88


def iou_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(aa + ba - inter)


def _pose_reuse_ok(
    box: tuple[int, int, int, int],
    prev_box: tuple[int, int, int, int] | None,
    fw: int,
    fh: int,
) -> bool:
    """Allow last-frame pose reuse only when the person crop has not jumped far."""
    if prev_box is None:
        return True
    cx1, cy1 = rect_center(box)
    cx0, cy0 = rect_center(prev_box)
    max_jump = 0.11 * math.hypot(float(fw), float(fh))
    return math.hypot(cx1 - cx0, cy1 - cy0) <= max_jump


def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for tag in ("avc1", "H264", "mp4v"):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if wr.isOpened():
            return wr
    raise RuntimeError("VideoWriter failed")


def expand_smooth_box(
    box: tuple[int, int, int, int],
    fw: int, fh: int,
    prev_smooth: tuple[float, float, float, float] | None,
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    bw2 = bw * (1.0 + BOX_EXPAND_W)
    bh2 = bh * (1.0 + BOX_EXPAND_H)
    nx1 = int(round(cx - bw2 / 2))
    ny1 = int(round(cy - bh2 / 2))
    nx2 = int(round(cx + bw2 / 2))
    ny2 = int(round(cy + bh2 / 2))
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(fw - 1, nx2), min(fh - 1, ny2)
    if nx2 <= nx1 + 8:
        nx2 = min(fw - 1, nx1 + 9)
    if ny2 <= ny1 + 8:
        ny2 = min(fh - 1, ny1 + 9)
    cur = (float(nx1), float(ny1), float(nx2), float(ny2))
    if prev_smooth is None:
        out = cur
    else:
        a = EMA_BOX_ALPHA
        out = tuple(
            a * cur[i] + (1.0 - a) * prev_smooth[i] for i in range(4)
        )
    ibox = (
        int(round(out[0])),
        int(round(out[1])),
        int(round(out[2])),
        int(round(out[3])),
    )
    return ibox, out


def create_ball_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.006
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.2
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def create_ball_kalman_ca() -> cv2.KalmanFilter:
    """6-state constant-acceleration Kalman: [x, y, vx, vy, ax, ay].

    Used post-release where the ball follows a parabolic trajectory and
    a constant-velocity model lags the true position each frame.
    """
    kf = cv2.KalmanFilter(6, 2)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32
    )
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
    kf.processNoiseCov[4, 4] = 0.1
    kf.processNoiseCov[5, 5] = 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.2
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


def _ball_roi_box(
    cx: int, cy: int, fw: int, fh: int
) -> tuple[int, int, int, int]:
    """Return an ROI (x0, y0, x1, y1) centered on (cx, cy) for windowed YOLO."""
    side = max(BALL_ROI_MIN_PX, int(BALL_ROI_FRAC * max(fw, fh)))
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(fw, cx + half)
    y1 = min(fh, cy + half)
    return x0, y0, x1, y1


def _ball_roi_near_edge(
    cx: int, cy: int, fw: int, fh: int
) -> bool:
    m = BALL_ROI_EDGE_MARGIN
    return cx < m or cy < m or cx > fw - m or cy > fh - m


def hsv_ball_candidates(frame: np.ndarray) -> list[tuple[int, int, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_w = cv2.inRange(hsv, _HSV_WHITE_LO, _HSV_WHITE_HI)
    mask_r = cv2.bitwise_or(
        cv2.inRange(hsv, _HSV_RED_LO_1, _HSV_RED_HI_1),
        cv2.inRange(hsv, _HSV_RED_LO_2, _HSV_RED_HI_2),
    )
    mask_p = cv2.inRange(hsv, _HSV_PINK_LO, _HSV_PINK_HI)
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_w, mask_r), mask_p)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    out: list[tuple[int, int, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 350:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        cx, cy = x + bw // 2, y + bh // 2
        out.append((cx, cy, min(0.95, area / 200.0)))
    return out


def motion_ball_candidates(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, float]]:
    """Color-agnostic fallback: small bright blobs that moved between frames.

    Operates inside `roi` if provided, otherwise the full frame. Returns
    centers in full-frame coordinates with a heuristic confidence.
    """
    if prev_gray is None:
        return []
    h, w = frame.shape[:2]
    if roi is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = roi
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    cur_gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if (
        prev_gray.shape[0] != h
        or prev_gray.shape[1] != w
    ):
        return []
    prev_crop = prev_gray[y0:y1, x0:x1]
    diff = cv2.absdiff(cur_gray, prev_crop)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 6 or area > 1200:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) > 60:
            continue
        ar = bw / float(bh) if bh else 0.0
        if ar < 0.4 or ar > 2.5:
            continue
        cx = x0 + x + bw // 2
        cy = y0 + y + bh // 2
        out.append((cx, cy, min(0.45, 0.18 + area / 600.0)))
    return out


def parabola_inlier_frames(
    pts: list[tuple[int, int, int]],
) -> set[int]:
    if len(pts) < 6:
        return {p[0] for p in pts}
    f = np.array([p[0] for p in pts], dtype=np.float64)
    cx = np.array([p[1] for p in pts], dtype=np.float64)
    cy = np.array([p[2] for p in pts], dtype=np.float64)
    try:
        py = np.polyfit(f, cy, 2)
        fit_y = py[0] * f**2 + py[1] * f + py[2]
        ry = np.abs(cy - fit_y)
        sy = float(np.std(ry)) + 1e-6
        px = np.polyfit(f, cx, 1)
        fit_x = px[0] * f + px[1]
        rx = np.abs(cx - fit_x)
        sx = float(np.std(rx)) + 1e-6
        m = (ry < 2.0 * sy) & (rx < 2.0 * sx)
        return {pts[i][0] for i, ok in enumerate(m) if ok}
    except Exception:
        return {p[0] for p in pts}


def load_bowler_exclusion_zones(path: str | None) -> list[dict[str, float]]:
    """Load static fielder exclusion disks from a bowlfast-style calibration JSON."""
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw = data.get("exclusion_zones") or []
    out: list[dict[str, float]] = []
    for z in raw:
        if not isinstance(z, dict):
            continue
        try:
            out.append(
                {
                    "cx": float(z["cx"]),
                    "cy": float(z["cy"]),
                    "radius": float(z["radius"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _exclusion_zone_penalty(
    nx: float, ny: float, zones: list[dict[str, float]]
) -> float:
    for z in zones:
        if math.hypot(nx - z["cx"], ny - z["cy"]) < z["radius"]:
            return 10.0
    return 0.0


def _deep_slip_fringe_penalty(nx: float, ny: float, phase: str) -> float:
    """Standing slips: far left/right and relatively high in frame (distant)."""
    if phase not in ("DELIVERY", "RELEASE", "FOLLOWTHROUGH"):
        return 0.0
    pen = 0.0
    if (nx < 0.12 or nx > 0.88) and ny < 0.50:
        pen += 4.2
    elif (nx < 0.18 or nx > 0.82) and ny < 0.40:
        pen += 2.6
    if phase in ("DELIVERY", "RELEASE"):
        if (nx < 0.08 or nx > 0.92) and ny < 0.55:
            pen += 2.2
    return pen


def _bowling_arm_spatial_bonus(
    nx: float, ny: float, bowling_arm: str, phase: str, before_ffc: bool
) -> float:
    """Back-view prior: RFM often approaches from camera-left and fills lower frame;
    left-arm mirrors to the right. Complements --entry-side when both are set correctly.
    """
    if not before_ffc and phase not in ("RUN_UP", "JUMP"):
        return 0.0
    b = 0.0
    if bowling_arm == "right":
        b += max(0.0, 0.52 - nx) * 2.45
        if ny > 0.36:
            b += (ny - 0.36) * 1.65
    else:
        b += max(0.0, nx - 0.48) * 2.45
        if ny > 0.36:
            b += (ny - 0.36) * 1.65
    return b


@dataclass
class BowlerTracker:
    person_model: YOLO
    bowling_arm: str
    entry_side: str
    fw: int
    fh: int
    prev_box: tuple[int, int, int, int] | None = None
    smooth_box: tuple[float, float, float, float] | None = None
    ffc_frame: int | None = None
    frozen_offscreen: bool = False
    last_good_score: float = 0.0
    freeze_frames: int = 0
    lost_frames: int = 0
    # FIX 4: Track consecutive low-IOU frames before committing to frozen state
    _consecutive_low_iou: int = field(default=0, init=False)
    # FIX 6A: Count YOLO frames processed — used to apply bottom-of-frame
    # bias only during the early run-up before the bowler grows in frame.
    _bootstrap_frames: int = field(default=0, init=False)
    prev_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    box_area_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=12)
    )
    exclusion_zones: list[dict[str, float]] = field(default_factory=list)

    def _find_dominant_bowler_box(
        self,
        boxes: list[tuple[int, int, int, int]],
        phase: str,
    ) -> tuple[int, int, int, int] | None:
        """Single clear largest person (approaching bowler) vs static slips."""
        if phase not in ("RUN_UP", "DELIVERY", "JUMP"):
            return None
        if len(boxes) < 1:
            return None

        def area(b: tuple[int, int, int, int]) -> int:
            return max(1, (b[2] - b[0]) * (b[3] - b[1]))

        by_area = sorted(boxes, key=area, reverse=True)
        largest = by_area[0]
        la = area(largest)
        if len(by_area) >= 2 and la < area(by_area[1]) * 1.9:
            return None

        cx, cy = rect_center(largest)
        nx, ny = cx / max(1.0, float(self.fw)), cy / max(1.0, float(self.fh))
        if self.entry_side == "left" and nx > 0.65:
            return None
        if self.entry_side == "right" and nx < 0.35:
            return None
        if (nx < 0.12 or nx > 0.88) and ny < 0.52:
            return None
        arm = self.bowling_arm.strip().lower()
        if phase == "RUN_UP":
            if arm == "right" and nx > 0.74:
                return None
            if arm == "left" and nx < 0.26:
                return None
        if _exclusion_zone_penalty(nx, ny, self.exclusion_zones) > 0:
            return None
        return largest

    def _score_box(
        self,
        box: tuple[int, int, int, int],
        before_ffc: bool,
        ball_pos: tuple[int, int] | None,
        phase: str,
        movement: float,
    ) -> float:
        x1, y1, x2, y2 = box
        h, wbox = y2 - y1, x2 - x1
        cx, cy = rect_center(box)
        nx = cx / max(1.0, self.fw)
        ny = cy / max(1.0, self.fh)
        score = 0.0

        if self.entry_side == "left" and nx > 0.55:
            score -= 12.0
        elif self.entry_side == "right" and nx < 0.45:
            score -= 12.0

        score += min(1.0, h / max(1.0, 0.58 * self.fh)) * 2.6
        score += min(1.0, wbox / max(1.0, 0.22 * self.fw)) * 0.65

        score -= _exclusion_zone_penalty(nx, ny, self.exclusion_zones)

        if _in_umpire_corral(nx, ny):
            score -= 7.5

        if phase in ("RUN_UP", "JUMP") or before_ffc:
            if phase == "RUN_UP":
                if self.entry_side == "left":
                    if nx > 0.58:
                        score -= 7.0
                    else:
                        score += max(0.0, 1.0 - nx) * 3.5
                else:
                    if nx < 0.42:
                        score -= 7.0
                    else:
                        score += max(0.0, nx) * 3.5
            else:
                if self.entry_side == "left":
                    score += max(0.0, 1.0 - nx) * 2.8
                else:
                    score += max(0.0, nx) * 2.8
            centrality = 1.0 - min(1.0, abs(nx - 0.5) / 0.42)
            score -= 2.35 * centrality
            score -= max(0.0, ny - 0.76) * 3.2
            if ball_pos is not None:
                bx, by = ball_pos
                dist = math.hypot(cx - bx, cy - by)
                score += max(0.0, 1.0 - dist / max(1.0, 0.48 * self.fw)) * 3.6
            if self._bootstrap_frames < 20:
                score += ny * 3.5
        elif phase in ("DELIVERY", "RELEASE", "FOLLOWTHROUGH"):
            if self.entry_side == "left":
                if nx > 0.55:
                    score -= 5.0
                elif _on_entry_side(nx, self.entry_side):
                    score += 2.5
            else:
                if nx < 0.45:
                    score -= 5.0
                elif _on_entry_side(nx, self.entry_side):
                    score += 2.5
            centrality = 1.0 - min(1.0, abs(nx - 0.5) / 0.42)
            score -= 1.25 * centrality
            score -= _deep_slip_fringe_penalty(nx, ny, phase)

        score += _bowling_arm_spatial_bonus(
            nx, ny, self.bowling_arm.strip().lower(), phase, before_ffc
        )

        if phase == "RUN_UP":
            if movement < 3.0:
                score -= 10.0
            elif movement < 7.5:
                score -= 8.0
            else:
                score += min(movement / 20.0, 3.0)
            if len(self.box_area_history) >= 3:
                med_area = float(statistics.median(self.box_area_history))
                cand_area = float((x2 - x1) * (y2 - y1))
                if cand_area < 0.85 * med_area:
                    score -= 8.0

        if len(self.box_area_history) >= 4:
            areas = list(self.box_area_history)
            recent_avg = sum(areas[-3:]) / 3.0
            older_avg = sum(areas[:3]) / 3.0
            if older_avg > 0:
                growth_rate = (recent_avg - older_avg) / older_avg
                if growth_rate > 0.03:
                    score += min(growth_rate * 15.0, 4.0)
                elif growth_rate < -0.02:
                    score -= 3.0
                elif abs(growth_rate) < 0.01 and phase == "RUN_UP":
                    score -= 2.5

        if self.prev_box is not None:
            pnx, pny = _box_norm_center(self.prev_box, self.fw, self.fh)
            prev_on_umpire = _in_umpire_corral(pnx, pny)
            iou_w = 7.2 if not before_ffc else 4.6
            if prev_on_umpire:
                iou_w = min(iou_w, 2.2)
            drift_w = 4.2 if not before_ffc else 1.65
            drift_scale = 0.17 * self.fw if not before_ffc else 0.30 * self.fw
            score += iou_box(box, self.prev_box) * iou_w
            pcx, pcy = rect_center(self.prev_box)
            drift = math.hypot(cx - pcx, cy - pcy)
            score += max(0.0, 1.0 - drift / max(1.0, drift_scale)) * drift_w
            if (
                prev_on_umpire
                and _on_entry_side(nx, self.entry_side)
                and not _in_umpire_corral(nx, ny)
            ):
                score += 4.0

        return float(score)

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        phase: str,
        events: dict[str, int | None],
        ball_pos: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        before_ffc = events.get("FFC") is None
        if events.get("FFC") is not None and self.ffc_frame is None:
            self.ffc_frame = int(events["FFC"])

        if self.frozen_offscreen and self.prev_box is not None:
            pnx, pny = _box_norm_center(self.prev_box, self.fw, self.fh)
            if not _in_umpire_corral(pnx, pny):
                ibox, self.smooth_box = expand_smooth_box(
                    self.prev_box, self.fw, self.fh, self.smooth_box
                )
                return ibox
            self.frozen_offscreen = False

        if frame_idx % YOLO_PERSON_EVERY != 0 and self.prev_box is not None:
            ibox, self.smooth_box = expand_smooth_box(
                self.prev_box, self.fw, self.fh, self.smooth_box
            )
            return ibox

        res = self.person_model(frame, conf=0.35, classes=[0], verbose=False)[0]
        min_area = 0.02 * self.fw * self.fh
        raw_boxes: list[tuple[int, int, int, int]] = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                raw_boxes.append((x1, y1, x2, y2))

        if phase == "RUN_UP":
            raw_boxes = [
                b
                for b in raw_boxes
                if rect_center(b)[1] < int(self.fh * 0.74)
            ]
        raw_boxes = _exclude_runup_nearly_stationary_boxes(
            raw_boxes, phase, self.prev_box
        )

        self._bootstrap_frames += 1

        dominant_box = self._find_dominant_bowler_box(raw_boxes, phase)
        # Latch when no track yet, or after long loss (e.g. stuck on slip / dropout).
        allow_dominant = dominant_box is not None and (
            self.prev_box is None
            or (
                phase in ("RUN_UP", "DELIVERY", "JUMP")
                and self.lost_frames >= 28
            )
        )
        if allow_dominant:
            self._consecutive_low_iou = 0
            self.freeze_frames = 0
            self.lost_frames = 0
            self.last_good_score = 0.0
            dx1, dy1, dx2, dy2 = dominant_box
            self.box_area_history.append((dx2 - dx1) * (dy2 - dy1))
            self.prev_box = dominant_box
            ibox, self.smooth_box = expand_smooth_box(
                dominant_box, self.fw, self.fh, self.smooth_box
            )
            dcx, dcy = rect_center(dominant_box)
            dgkey = f"{dcx // 40 * 40}_{dcy // 40 * 40}"
            self.prev_positions[dgkey] = (dcx, dcy)
            return ibox

        candidates = raw_boxes

        if self.prev_box is not None and candidates:
            gate = make_search_gate(self.prev_box, self.fw, self.fh, self.lost_frames)
            gated = [b for b in candidates if box_inside_gate(b, gate)]
            if gated:
                candidates = gated

        persons: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for box in candidates:
            x1, y1, x2, y2 = box
            cx_i = (x1 + x2) // 2
            cy_i = (y1 + y2) // 2
            persons.append((cx_i, cy_i, box))

        height = self.fh
        if phase == "RUN_UP":
            persons = [p for p in persons if p[1] < int(height * 0.73)]
        elif phase in ("DELIVERY", "RELEASE"):
            persons = [p for p in persons if p[1] < int(height * 0.88)]

        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for cx_i, cy_i, box in persons:
            gkey = f"{cx_i // 40 * 40}_{cy_i // 40 * 40}"
            if gkey in self.prev_positions:
                px0, py0 = self.prev_positions[gkey]
                movement = math.hypot(cx_i - px0, cy_i - py0)
            else:
                movement = 0.0
            sc = self._score_box(
                box, before_ffc, ball_pos, phase, movement
            )
            scored.append((sc, box))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_box: tuple[int, int, int, int] | None = None
        best_sc = -1e9
        if scored:
            best_sc, best_box = scored[0]

        # Score-based freeze only after FFC — during run-up it trapped the box on
        # static clutter when the real bowler had already left the ROI.
        if (
            not before_ffc
            and self.prev_box is not None
            and best_box is not None
            and self.last_good_score > 0
            and best_sc < self.last_good_score - BOWLER_SCORE_SWITCH_MARGIN
            and self.freeze_frames < 6
        ):
            self.freeze_frames += 1
            self.lost_frames = min(30, self.lost_frames + 1)
            ibox, self.smooth_box = expand_smooth_box(
                self.prev_box, self.fw, self.fh, self.smooth_box
            )
            for cx_i, cy_i, box in persons:
                gkey = f"{cx_i // 40 * 40}_{cy_i // 40 * 40}"
                self.prev_positions[gkey] = (cx_i, cy_i)
            if self.prev_box is not None:
                bx1, by1, bx2, by2 = self.prev_box
                self.box_area_history.append((bx2 - bx1) * (by2 - by1))
            return ibox

        # Before FFC, avoid jumping to wrong static fielder if IoU is tiny.
        if (
            self.prev_box is not None
            and best_box is not None
            and before_ffc
            and iou_box(best_box, self.prev_box) < 0.18
        ):
            prev_sc = self._score_box(
                self.prev_box, before_ffc, ball_pos, phase, 0.0
            )
            if best_sc < prev_sc + BOWLER_SCORE_SWITCH_MARGIN:
                best_box = self.prev_box
                best_sc = prev_sc
                self.freeze_frames = min(6, self.freeze_frames + 1)

        if best_box is not None:
            self.freeze_frames = 0
            self.lost_frames = 0
            self.last_good_score = best_sc

        # FIX 4: Don't permanently freeze on a single low-IOU frame.
        # Require CONSECUTIVE_LOW_IOU_THRESHOLD bad frames before locking,
        # so a brief occlusion or detection hiccup doesn't strand the box.
        if not before_ffc and self.prev_box is not None and best_box is not None:
            iou_v = iou_box(best_box, self.prev_box)
            pnx, pny = _box_norm_center(self.prev_box, self.fw, self.fh)
            bnx, bny = _box_norm_center(best_box, self.fw, self.fh)
            escape_umpire = (
                _in_umpire_corral(pnx, pny)
                and _on_entry_side(bnx, self.entry_side)
                and not _in_umpire_corral(bnx, bny)
                and best_sc >= self.last_good_score - 0.5
            )
            if iou_v < 0.2 and not escape_umpire:
                self._consecutive_low_iou += 1
                best_box = self.prev_box
                if self._consecutive_low_iou > 10:
                    self.frozen_offscreen = True
            else:
                self._consecutive_low_iou = 0
                if escape_umpire:
                    self.frozen_offscreen = False

        if best_box is None:
            if self.prev_box is None:
                w, h = self.fw, self.fh
                roi_w = int(w * 0.45)
                x1 = 0 if self.entry_side == "left" else w - roi_w
                best_box = (x1, 0, min(w, x1 + roi_w), h)
            else:
                self.lost_frames = min(30, self.lost_frames + 1)
                best_box = self.prev_box

        for cx_i, cy_i, box in persons:
            gkey = f"{cx_i // 40 * 40}_{cy_i // 40 * 40}"
            self.prev_positions[gkey] = (cx_i, cy_i)

        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            self.box_area_history.append((bx2 - bx1) * (by2 - by1))

        self.prev_box = best_box
        ibox, self.smooth_box = expand_smooth_box(
            best_box, self.fw, self.fh, self.smooth_box
        )
        return ibox


def extract_pose_landmarks(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    pose_est: Any,
    fw: int,
    fh: int,
    entry_side: str | None = None,
) -> list[Any] | None:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    cx_box = 0.5 * (x1 + x2)
    es = (entry_side or "").strip().lower()
    if es == "left" and cx_box > round(0.55 * fw):
        return None
    if es == "right" and cx_box < round(0.45 * fw):
        return None
    roi = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pr = pose_est.process(rgb)
    if not pr.pose_landmarks:
        return None
    rh, rw = roi.shape[:2]
    out: list[Any] = []
    for lm in pr.pose_landmarks.landmark:
        px = (lm.x * rw + x1) / max(1, fw)
        py = (lm.y * rh + y1) / max(1, fh)
        out.append(SimpleNamespace(x=px, y=py, visibility=lm.visibility))
    return out


def _smooth_y_series(
    hist: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    if len(hist) < 3:
        return list(hist)
    out: list[tuple[int, float]] = []
    for i in range(len(hist)):
        s = max(0, i - 1)
        e = min(len(hist), i + 2)
        ys = [hist[j][1] for j in range(s, e)]
        out.append((hist[i][0], sum(ys) / len(ys)))
    return out


def _savgol_coeffs(window: int, polyorder: int = 2) -> np.ndarray:
    """Compute Savitzky-Golay smoothing coefficients via least squares."""
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    return pinv[0]


def smooth_series(
    hist: list[tuple[int, float]],
    fps: float,
    window_s: float = 0.15,
    polyorder: int = 2,
) -> list[tuple[int, float]]:
    """Savitzky-Golay smoother (works on irregular frame indices by treating
    the sample sequence as uniformly spaced — typical case here since the
    pre-pass appends one sample per processed frame whenever the landmark is
    visible).

    Falls back to the simple 3-tap mean if the series is too short.
    """
    n = len(hist)
    if n < 5:
        return _smooth_y_series(hist)
    win = max(5, int(round(window_s * fps)))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return _smooth_y_series(hist)
    poly = min(polyorder, win - 1)
    coeffs = _savgol_coeffs(win, poly)
    ys = np.array([y for _, y in hist], dtype=np.float64)
    half = win // 2
    padded = np.concatenate(
        [np.full(half, ys[0]), ys, np.full(half, ys[-1])]
    )
    smooth = np.convolve(padded, coeffs[::-1], mode="valid")
    smooth = smooth[: n]
    return [(hist[i][0], float(smooth[i])) for i in range(n)]


def smoothed_velocity(
    smoothed: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Frame-to-frame first derivative of an already-smoothed y series.

    Returned units are 'normalised y per frame'. Edges use one-sided diffs.
    """
    n = len(smoothed)
    if n < 2:
        return [(f, 0.0) for f, _ in smoothed]
    out: list[tuple[int, float]] = []
    for i in range(n):
        if i == 0:
            v = smoothed[1][1] - smoothed[0][1]
        elif i == n - 1:
            v = smoothed[-1][1] - smoothed[-2][1]
        else:
            v = (smoothed[i + 1][1] - smoothed[i - 1][1]) / 2.0
        out.append((smoothed[i][0], float(v)))
    return out


def compute_ankle_baseline(
    smoothed: list[tuple[int, float]],
    fps: float,
    window_s: float = 3.0,
) -> float | None:
    """Estimate the 'ground level' (floor) y for an ankle from the run-up.

    Looks at the first `window_s` seconds, finds local maxima (foot-strike
    instants — the foot is at its lowest screen position, i.e. y is largest),
    and returns the median of the top-quartile of those values. Returns None
    if not enough samples.
    """
    if not smoothed:
        return None
    n_window = min(len(smoothed), max(8, int(window_s * fps)))
    seg = smoothed[:n_window]
    ys = np.array([y for _, y in seg], dtype=np.float64)
    if len(ys) < 5:
        return float(np.median(ys))
    maxima: list[float] = []
    for i in range(2, len(ys) - 2):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and (
            ys[i] > ys[i - 2] - 1e-9 or ys[i] > ys[i + 2] - 1e-9
        ):
            maxima.append(float(ys[i]))
    if len(maxima) < 3:
        return float(np.percentile(ys, 80))
    arr = np.array(maxima, dtype=np.float64)
    cutoff = float(np.percentile(arr, 50))
    top = arr[arr >= cutoff]
    return float(np.median(top)) if top.size else float(np.median(arr))


def _local_max_at(sm: list[tuple[int, float]], i: int) -> bool:
    yi = sm[i][1]
    lo = max(0, i - 3)
    hi = min(len(sm), i + 2)
    for j in range(lo, hi):
        if sm[j][1] > yi + 1e-9:
            return False
    return True


def try_detect_bfc(
    hist: list[tuple[int, float]],
    fps: float,
    min_frame: int = 0,
) -> int | None:
    if len(hist) < 8:
        return None
    for i in range(3, len(hist) - 4):
        fi, yi = hist[i]
        if fi < min_frame:
            continue
        _, y0 = hist[i - 3]
        _, y1 = hist[i - 2]
        _, y2 = hist[i - 1]
        _, yp = hist[i + 1]
        if not (yi >= y1 >= y2 >= y0):
            continue
        if not (yi >= y1 and yi >= yp):
            continue
        slab = [hist[j][1] for j in range(i, min(i + 4, len(hist)))]
        if max(slab) - min(slab) > 0.003:
            continue
        return fi
    return None


def try_detect_ffc(
    hist: list[tuple[int, float]],
    bfc: int,
    fps: float,
    bfc_y: float | None = None,
) -> int | None:
    if len(hist) < 8:
        return None
    max_f = bfc + int(fps * 1.5)
    for i in range(3, len(hist) - 4):
        fi, yi = hist[i]
        if fi <= bfc or fi > max_f:
            continue
        _, y0 = hist[i - 3]
        _, y1 = hist[i - 2]
        _, y2 = hist[i - 1]
        _, yp = hist[i + 1]
        if not (yi >= y1 >= y2 >= y0):
            continue
        if not (yi >= y1 and yi >= yp):
            continue
        slab = [hist[j][1] for j in range(i, min(i + 4, len(hist)))]
        if max(slab) - min(slab) > 0.003:
            continue
        return fi
    return None


@dataclass
class PhaseDetector:
    fps: float
    bowling_arm: str
    fw: int
    fh: int
    b_ankle_idx: int = field(init=False)
    nb_ankle_idx: int = field(init=False)
    b_hist: "deque[tuple[int, float]]" = field(default_factory=deque)
    nb_hist: "deque[tuple[int, float]]" = field(default_factory=deque)
    phase: str = "RUN_UP"
    events: dict[str, int | None] = field(
        default_factory=lambda: {
            "IMPULSE": None,
            "JUMP_END": None,
            "LANDING": None,
            "BFC": None,
            "FFC": None,
            "RELEASE": None,
        }
    )
    ball_sep_streak: int = 0
    ball_sep_disp_seen: bool = False
    wrist_min_y: float | None = None
    wrist_rise_frames: int = 0
    prev_ball_dist: float | None = None
    wrist_y_hist: list[tuple[int, float]] = field(default_factory=list)
    # When set (from a pre-pass over the whole video), `update()` skips its
    # online detectors and instead reveals events as the frame counter passes
    # them. This makes phase boundaries deterministic and well-timed.
    pinned_events: dict[str, int | None] | None = None

    def __post_init__(self) -> None:
        if self.bowling_arm == "right":
            self.b_ankle_idx, self.nb_ankle_idx = RIGHT_ANKLE, LEFT_ANKLE
        else:
            self.b_ankle_idx, self.nb_ankle_idx = LEFT_ANKLE, RIGHT_ANKLE
        ml = int(self.fps * 4)
        self.b_hist = deque(maxlen=ml)
        self.nb_hist = deque(maxlen=ml)

    def _bfc_smoothed_y(self, bfc: int) -> float | None:
        sm = _smooth_y_series(list(self.b_hist))
        for f, y in sm:
            if f == bfc:
                return y
        return None

    def _compute_impulse(self, bfc: int) -> int | None:
        sm = _smooth_y_series(list(self.b_hist))
        win_size = max(4, int(self.fps * 1.2))
        pre = [(f, y) for f, y in sm if f < bfc][-win_size:]
        if len(pre) < 4:
            return None
        apex_idx = min(range(len(pre)), key=lambda k: pre[k][1])
        pre_apex = pre[: apex_idx + 1]
        if len(pre_apex) < 3:
            return None
        prev_max_idx = max(
            range(len(pre_apex)), key=lambda k: pre_apex[k][1]
        )
        if prev_max_idx >= len(pre_apex) - 1:
            return None
        prev_max_y = pre_apex[prev_max_idx][1]
        for k in range(prev_max_idx + 1, len(pre_apex)):
            if pre_apex[k][1] < prev_max_y - 0.005:
                return pre_apex[k][0]
        return None

    def _pinned_phase_for(self, frame_idx: int) -> str:
        pin = self.pinned_events or {}
        evi = self.events.get("IMPULSE") or pin.get("IMPULSE")
        evl = (
            self.events.get("LANDING")
            or self.events.get("JUMP_END")
            or pin.get("LANDING")
            or pin.get("JUMP_END")
        )
        evb = self.events.get("BFC") or pin.get("BFC")
        evf = self.events.get("FFC") or pin.get("FFC")
        evr = self.events.get("RELEASE") or pin.get("RELEASE")
        if evr is not None and frame_idx >= evr:
            return "FOLLOWTHROUGH"
        if evf is not None and frame_idx >= evf:
            return "RELEASE"
        if evb is not None and frame_idx >= evb:
            return "DELIVERY"
        if evl is not None and frame_idx >= evl:
            return "DELIVERY"
        if evi is not None and frame_idx >= evi:
            return "JUMP"
        return "RUN_UP"

    def _update_pinned(self, frame_idx: int) -> None:
        assert self.pinned_events is not None
        for key in ("IMPULSE", "JUMP_END", "LANDING", "BFC", "FFC", "RELEASE"):
            f = self.pinned_events.get(key)
            if (
                f is not None
                and frame_idx >= f
                and self.events.get(key) is None
            ):
                self.events[key] = f
        self.phase = self._pinned_phase_for(frame_idx)

    def update(
        self,
        frame_idx: int,
        lms: list[Any] | None,
        ball_px: tuple[int, int] | None,
        ball_spd: float,
        ball_conf: float,
    ) -> None:
        if self.pinned_events is not None:
            self._update_pinned(frame_idx)
            return

        if lms is None:
            return

        def yank(idx: int) -> float | None:
            lm = lms[idx]
            return float(lm.y) if lm_vis(lm) > 0.5 else None

        by = yank(self.b_ankle_idx)
        if by is not None:
            self.b_hist.append((frame_idx, by))
        nby = yank(self.nb_ankle_idx)
        if nby is not None:
            self.nb_hist.append((frame_idx, nby))

        widx = RIGHT_WRIST if self.bowling_arm == "right" else LEFT_WRIST
        wlm = lms[widx]

        if self.events["BFC"] is None:
            bfc = try_detect_bfc(list(self.b_hist), self.fps, 0)
            if bfc is not None:
                self.events["BFC"] = bfc
                self.phase = "DELIVERY"
                impulse = self._compute_impulse(bfc)
                if impulse is not None and impulse < bfc:
                    self.events["IMPULSE"] = impulse
        elif self.events["FFC"] is None:
            bfc = self.events["BFC"]
            assert bfc is not None
            bfc_y = self._bfc_smoothed_y(int(bfc))
            ffc = try_detect_ffc(
                list(self.nb_hist), int(bfc), self.fps, bfc_y=bfc_y
            )
            if ffc is not None:
                self.events["FFC"] = ffc
                self.phase = "RELEASE"
        elif self.events["RELEASE"] is None:
            ffc = int(self.events["FFC"])
            if frame_idx <= ffc:
                return
            released = False
            if lm_vis(wlm) > 0.5:
                self.wrist_y_hist.append((frame_idx, float(wlm.y)))
            if (
                ball_px is not None
                and lm_vis(wlm) > 0.5
                and ball_conf >= 0.35
            ):
                wx = wlm.x * self.fw
                wy = wlm.y * self.fh
                bx, by = ball_px
                d = math.hypot(bx - wx, by - wy)
                if self.prev_ball_dist is not None:
                    if d > self.prev_ball_dist + 10.0 and ball_spd > 4.0:
                        self.ball_sep_streak += 1
                        if d > 0.05 * self.fh:
                            self.ball_sep_disp_seen = True
                    else:
                        self.ball_sep_streak = 0
                        self.ball_sep_disp_seen = False
                self.prev_ball_dist = d
                if self.ball_sep_streak >= 3 and self.ball_sep_disp_seen:
                    released = True
            wrist_margin = int(self.fps * 0.08)
            if (
                not released
                and lm_vis(wlm) > 0.5
                and frame_idx >= ffc + wrist_margin
            ):
                wy = float(wlm.y)
                if self.wrist_min_y is None or wy < self.wrist_min_y:
                    self.wrist_min_y = wy
                    self.wrist_rise_frames = 0
                elif (
                    self.wrist_min_y is not None
                    and wy >= self.wrist_min_y + 0.02
                ):
                    self.wrist_rise_frames += 1
                    if self.wrist_rise_frames >= 4:
                        released = True
                else:
                    self.wrist_rise_frames = 0
            if released:
                self.events["RELEASE"] = frame_idx
                self.phase = "FOLLOWTHROUGH"

    def finalize_release_fallback(self) -> None:
        if self.events["RELEASE"] is not None:
            return
        ffc = self.events["FFC"]
        if ffc is None:
            return
        margin = int(self.fps * 0.08)
        h = [(f, y) for f, y in self.wrist_y_hist if f > ffc + margin]
        if len(h) < 8:
            return
        ymin = min(y for _, y in h)
        i0 = next((i for i, (_, y) in enumerate(h) if y <= ymin + 1e-6), None)
        if i0 is None:
            return
        rise = 0
        for i in range(i0 + 1, len(h)):
            if h[i][1] >= ymin + 0.02:
                rise += 1
                if rise >= 4:
                    self.events["RELEASE"] = h[i][0]
                    self.phase = "FOLLOWTHROUGH"
                    return
            else:
                rise = 0

    def events_audit(self) -> None:
        if self.pinned_events is not None:
            return
        evb = self.events.get("BFC")
        evf = self.events.get("FFC")
        evr = self.events.get("RELEASE")
        evi = self.events.get("IMPULSE")

        if evi is not None and evb is not None and evi >= evb:
            self.events["IMPULSE"] = None
            evi = None

        if evb is not None and evf is not None and evf - evb < 2:
            self.events["FFC"] = None
            self.events["RELEASE"] = None
            self.wrist_min_y = None
            self.wrist_rise_frames = 0
            self.phase = "DELIVERY"
            evf = None
            evr = None

        if evf is not None and evr is not None and evr <= evf:
            self.events["RELEASE"] = None
            self.wrist_min_y = None
            self.wrist_rise_frames = 0
            self.phase = "RELEASE"
            evr = None

        if evr is None:
            self.finalize_release_fallback()


BALL_MAX_LOST = 8
# Post-release the ball follows a smooth ballistic path, so we let the Kalman
# filter coast through longer occlusions (behind the batter/stumps) before
# dropping the track — this keeps the flight trail continuous to screen-exit.
BALL_MAX_LOST_FLIGHT = 14
BALL_MAX_JUMP = 72
BALL_GATE_SOFT = 95
BALL_DISP_SMOOTH = 0.2


def _kalman_init(kf: cv2.KalmanFilter, cx: int, cy: int) -> None:
    n = kf.transitionMatrix.shape[0]
    s = np.zeros((n, 1), dtype=np.float32)
    s[0, 0] = float(cx)
    s[1, 0] = float(cy)
    kf.statePre = s.copy()
    kf.statePost = s.copy()


def _kalman_update(kf: cv2.KalmanFilter, cx: int, cy: int) -> tuple[int, int]:
    kf.predict()
    c = kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
    return int(c[0]), int(c[1])


def _kalman_predict(kf: cv2.KalmanFilter) -> tuple[int, int]:
    p = kf.predict()
    return int(p[0]), int(p[1])


def _ball_max_jump(fw: int) -> int:
    """Scale max per-frame jump with frame width (HD/4K need a larger gate)."""
    return max(BALL_MAX_JUMP, int(0.06 * fw))


def _ball_plausible(
    prev: tuple[int, int] | None, nxt: tuple[int, int], max_jump: int
) -> bool:
    if prev is None:
        return True
    return math.hypot(nxt[0] - prev[0], nxt[1] - prev[1]) < max_jump


def _reject_static_ground_det(
    cx: int,
    cy: int,
    fw: int,
    fh: int,
    wrist_xy: tuple[int, int] | None,
) -> bool:
    """True if detection should be discarded (crease line, sightscreen edge)."""
    if wrist_xy is not None:
        near_wrist = (
            math.hypot(cx - wrist_xy[0], cy - wrist_xy[1])
            < BALL_CREASE_WRIST_EXEMPT_PX
        )
    else:
        near_wrist = False
    if cy > int(BALL_CREASE_Y_FRAC * fh) and not near_wrist:
        return True
    edge = int(BALL_EDGE_X_FRAC * fw)
    if (cx < edge or cx > fw - edge) and not near_wrist:
        return True
    return False


def _filter_ball_dets(
    dets: list[tuple[int, int, float]],
    fw: int,
    fh: int,
    wrist_xy: tuple[int, int] | None,
) -> list[tuple[int, int, float]]:
    kept, _ = _filter_ball_dets_stats(dets, fw, fh, wrist_xy)
    return kept


def _filter_ball_dets_stats(
    dets: list[tuple[int, int, float]],
    fw: int,
    fh: int,
    wrist_xy: tuple[int, int] | None,
) -> tuple[list[tuple[int, int, float]], int]:
    kept: list[tuple[int, int, float]] = []
    rejected = 0
    for cx, cy, cf in dets:
        if _reject_static_ground_det(cx, cy, fw, fh, wrist_xy):
            rejected += 1
        else:
            kept.append((cx, cy, cf))
    return kept, rejected


def _wrist_ball_roi(
    wx: int,
    wy: int,
    fw: int,
    fh: int,
    *,
    scale: float = 0.24,
) -> tuple[int, int, int, int]:
    """Tight crop around the bowling wrist for post-release ball search."""
    side = int(max(BALL_ROI_MIN_PX, scale * max(fw, fh)))
    half = side // 2
    return clamp_box((wx - half, wy - half, wx + half, wy + half), fw, fh)


def _flight_parabola_polyline(
    flight_pts: list[tuple[int, int, int]],
    inlier_frames: set[int] | None = None,
    n_samples: int = 40,
) -> list[tuple[int, int]]:
    """Fit x linear + y quadratic on flight inliers; return smooth screen points."""
    pts = flight_pts
    if inlier_frames is not None:
        pts = [p for p in flight_pts if p[0] in inlier_frames]
    if len(pts) < 4:
        return [(p[1], p[2]) for p in flight_pts]
    f = np.array([p[0] for p in pts], dtype=np.float64)
    cx = np.array([p[1] for p in pts], dtype=np.float64)
    cy = np.array([p[2] for p in pts], dtype=np.float64)
    try:
        px = np.polyfit(f, cx, 1)
        py = np.polyfit(f, cy, 2)
    except Exception:
        return [(p[1], p[2]) for p in flight_pts]
    f0, f1 = float(f.min()), float(f.max())
    if f1 <= f0:
        return [(int(cx[0]), int(cy[0]))]
    out: list[tuple[int, int]] = []
    for i in range(n_samples):
        t = f0 + (f1 - f0) * (i / max(1, n_samples - 1))
        x = int(round(px[0] * t + px[1]))
        y = int(round(py[0] * t * t + py[1] * t + py[2]))
        out.append((x, y))
    return out


def _pick_ball_measurement(
    dets: list[tuple[int, int, float]],
    anchor: tuple[int, int] | None,
    max_jump: int = BALL_MAX_JUMP,
    vel_hint: tuple[float, float] | None = None,
) -> tuple[int, int, float] | None:
    if not dets:
        return None
    if anchor is None:
        dets.sort(key=lambda t: t[2], reverse=True)
        return dets[0]
    ax, ay = anchor
    soft_gate = max(BALL_GATE_SOFT, max_jump)
    scored: list[tuple[float, tuple[int, int, float]]] = []
    for cx, cy, cf in dets:
        dist = math.hypot(cx - ax, cy - ay)
        s = cf * 1.65 - dist / 85.0 - max(0.0, dist - soft_gate) / 60.0
        if vel_hint is not None:
            vx, vy = vel_hint
            vn = math.hypot(vx, vy)
            if vn >= 1.2:
                dx, dy = float(cx - ax), float(cy - ay)
                dm = math.hypot(dx, dy) + 1e-6
                cos_al = (dx * vx + dy * vy) / (dm * vn)
                s += max(0.0, cos_al) * 0.62
        if dist > max_jump * 2.5 and cf < 0.45:
            continue
        scored.append((s, (cx, cy, cf)))
    if not scored:
        nearest = min(dets, key=lambda t: math.hypot(t[0] - ax, t[1] - ay))
        dmin = math.hypot(nearest[0] - ax, nearest[1] - ay)
        if dmin < max_jump * 2.5 and nearest[2] >= 0.30:
            return nearest
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score < -0.8:
        return None
    return best


@dataclass
class BallTrackState:
    model: YOLO
    kf: cv2.KalmanFilter = field(default_factory=create_ball_kalman)
    ready: bool = False
    lost: int = 0
    last: tuple[int, int] | None = None
    conf: float = 0.0
    predicted: bool = False
    raw_pts: list[tuple[int, int, int]] = field(default_factory=list)
    max_conf: float = 0.0
    disp_xy: tuple[float, float] | None = None
    flight_pts: list[tuple[int, int, int]] = field(default_factory=list)
    release_locked: bool = False
    debug: bool = False
    prev_gray: np.ndarray | None = field(default=None, repr=False)
    roi_fail_streak: int = 0
    last_kf_pred: tuple[int, int] | None = None
    last_roi: tuple[int, int, int, int] | None = None
    last_yolo_dets: list[tuple[int, int, float]] = field(default_factory=list)
    # Rolling log of recent raw YOLO detection centers (frame, cx, cy) used to
    # detect & reject persistent static false positives (sightscreen logos,
    # painted lines) that this ball model fires on — the real flying ball moves
    # fast and never accumulates a same-pixel cluster.
    static_hist: deque[tuple[int, int, int]] = field(
        default_factory=lambda: deque(maxlen=BALL_STATIC_WINDOW)
    )
    static_reject_frames: int = 0
    flight_inlier_frames: set[int] = field(default_factory=set)
    yolo_meas_frames: int = 0
    hsv_meas_frames: int = 0
    motion_meas_frames: int = 0
    yolo_raw_det_frames: int = 0
    rejected_geom_frames: int = 0
    flight_yolo_meas: int = 0
    flight_hsv_meas: int = 0
    flight_motion_meas: int = 0
    last_meas_source: str = ""

    def reset(self) -> None:
        """Clear tracker state when ball detection starts at LANDING."""
        self.kf = create_ball_kalman()
        self.ready = False
        self.lost = 0
        self.last = None
        self.conf = 0.0
        self.predicted = False
        self.raw_pts = []
        self.max_conf = 0.0
        self.disp_xy = None
        self.flight_pts = []
        self.release_locked = False
        self.prev_gray = None
        self.roi_fail_streak = 0
        self.last_kf_pred = None
        self.last_roi = None
        self.last_yolo_dets = []
        self.static_hist.clear()
        self.static_reject_frames = 0
        self.flight_inlier_frames = set()
        self.yolo_meas_frames = 0
        self.hsv_meas_frames = 0
        self.motion_meas_frames = 0
        self.yolo_raw_det_frames = 0
        self.rejected_geom_frames = 0
        self.flight_yolo_meas = 0
        self.flight_hsv_meas = 0
        self.flight_motion_meas = 0
        self.last_meas_source = ""

    def diagnostics_dict(self, release_frame: int | None) -> dict[str, int]:
        return {
            "yolo_meas_frames": self.yolo_meas_frames,
            "hsv_meas_frames": self.hsv_meas_frames,
            "motion_meas_frames": self.motion_meas_frames,
            "yolo_raw_det_frames": self.yolo_raw_det_frames,
            "rejected_geom_frames": self.rejected_geom_frames,
            "static_reject_frames": self.static_reject_frames,
            "flight_yolo_meas": self.flight_yolo_meas,
            "flight_hsv_meas": self.flight_hsv_meas,
            "flight_motion_meas": self.flight_motion_meas,
        }

    def _parabola_residual(self, fi: int, mx: int, my: int) -> float:
        if len(self.flight_pts) < 5:
            return 0.0
        recent = self.flight_pts[-12:]
        f = np.array([p[0] for p in recent], dtype=np.float64)
        cy = np.array([p[2] for p in recent], dtype=np.float64)
        cx = np.array([p[1] for p in recent], dtype=np.float64)
        try:
            py = np.polyfit(f, cy, 2)
            px_lin = np.polyfit(f, cx, 1)
            py_pred = py[0] * fi * fi + py[1] * fi + py[2]
            px_pred = px_lin[0] * fi + px_lin[1]
            return float(math.hypot(mx - px_pred, my - py_pred))
        except Exception:
            return 0.0

    def _is_static(self, cx: int, cy: int, frame_idx: int) -> bool:
        """True if a candidate sits on a persistent fixed-pixel cluster.

        Counts distinct *earlier* frames in the rolling window that produced a
        detection within BALL_STATIC_RADIUS_PX of (cx, cy). A genuine moving
        ball won't accumulate such a cluster; sightscreen/line false positives
        will. We never let the currently-tracked ball be flagged static so an
        actually-stationary lock can still recover.
        """
        if self.last is not None and math.hypot(
            cx - self.last[0], cy - self.last[1]
        ) <= BALL_STATIC_RADIUS_PX and not self.predicted:
            # Already the active, moving lock — don't second-guess it here.
            pass
        seen_frames: set[int] = set()
        for fr, hx, hy in self.static_hist:
            if fr == frame_idx:
                continue
            if math.hypot(cx - hx, cy - hy) <= BALL_STATIC_RADIUS_PX:
                seen_frames.add(fr)
        return len(seen_frames) >= BALL_STATIC_MIN_FRAMES

    def _switch_to_ca_kalman(self) -> None:
        """Replace the constant-velocity KF with a constant-acceleration KF
        once the ball is in flight, preserving the current position/velocity.
        """
        if self.kf.transitionMatrix.shape[0] >= 6:
            return
        old_state = self.kf.statePost.flatten()
        ox = float(old_state[0]) if len(old_state) > 0 else 0.0
        oy = float(old_state[1]) if len(old_state) > 1 else 0.0
        ovx = float(old_state[2]) if len(old_state) > 2 else 0.0
        ovy = float(old_state[3]) if len(old_state) > 3 else 0.0
        new_kf = create_ball_kalman_ca()
        s = np.array(
            [[ox], [oy], [ovx], [ovy], [0.0], [0.0]], dtype=np.float32
        )
        new_kf.statePre = s.copy()
        new_kf.statePost = s.copy()
        self.kf = new_kf

    def _run_yolo(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None,
        min_conf: float = BALL_YOLO_CONF_LOW,
    ) -> list[tuple[int, int, float]]:
        if roi is not None:
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return []
            res = self.model(
                crop,
                conf=min_conf,
                iou=BALL_YOLO_IOU,
                imgsz=BALL_YOLO_IMGSZ_ROI,
                verbose=False,
            )[0]
        else:
            res = self.model(
                frame,
                conf=min_conf,
                iou=BALL_YOLO_IOU,
                imgsz=BALL_YOLO_IMGSZ_FULL,
                verbose=False,
            )[0]
        out: list[tuple[int, int, float]] = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                cls_id = int(b.cls.view(-1)[0].item())
                if cls_id != 0:
                    continue
                cf = float(b.conf.cpu().numpy())
                if cf < min_conf:
                    continue
                x1b, y1b, x2b, y2b = map(int, b.xyxy.cpu().numpy().ravel())
                cx = (x1b + x2b) // 2
                cy = (y1b + y2b) // 2
                if roi is not None:
                    cx += roi[0]
                    cy += roi[1]
                out.append((cx, cy, cf))
        return out

    def step(
        self,
        frame_idx: int,
        frame: np.ndarray,
        wrist_xy: tuple[int, int] | None = None,
        release_frame: int | None = None,
        bowler_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[tuple[int, int] | None, float, float, bool]:
        fh, fw = frame.shape[:2]
        max_jump = _ball_max_jump(fw)
        pre_release = release_frame is None or frame_idx < release_frame
        post_release = not pre_release
        lost_cap = BALL_MAX_LOST_FLIGHT if post_release else BALL_MAX_LOST
        bootstrap = not self.ready
        if bootstrap:
            yolo_min_conf = BALL_YOLO_CONF_PRIMARY
        elif post_release:
            yolo_min_conf = BALL_YOLO_CONF_FLIGHT
        else:
            yolo_min_conf = BALL_YOLO_CONF_LOW
        wrist_gate = (
            BALL_BOOTSTRAP_WRIST_PX
            if (pre_release and bootstrap)
            else (BALL_PRE_RELEASE_WRIST_PX if pre_release else None)
        )

        # Higher cadence post-release or whenever we've lost the track.
        cadence = 1 if (post_release or self.lost > 0 or not self.ready) else YOLO_BALL_EVERY
        run_yolo = frame_idx % cadence == 0 or not self.ready

        # Predict first so we know where to crop the ROI.
        kf_pred: tuple[int, int] | None = None
        if self.ready:
            sp = self.kf.predict()
            kf_pred = (int(sp[0]), int(sp[1]))
            self.last_kf_pred = kf_pred

        yolo_dets: list[tuple[int, int, float]] = []
        roi_used: tuple[int, int, int, int] | None = None
        if run_yolo:
            anchor_for_roi = kf_pred if kf_pred is not None else self.last
            use_track_roi = (
                self.ready
                and anchor_for_roi is not None
                and self.roi_fail_streak < BALL_ROI_FAIL_LIMIT
                and not _ball_roi_near_edge(
                    anchor_for_roi[0], anchor_for_roi[1], fw, fh
                )
            )
            use_wrist_roi = (
                post_release
                and wrist_xy is not None
                and not use_track_roi
            )
            if use_wrist_roi:
                roi_used = _wrist_ball_roi(
                    wrist_xy[0], wrist_xy[1], fw, fh
                )
                yolo_dets = self._run_yolo(
                    frame, roi_used, min_conf=yolo_min_conf
                )
                if not yolo_dets:
                    yolo_dets = self._run_yolo(
                        frame, None, min_conf=yolo_min_conf
                    )
                    roi_used = None
            elif pre_release and bowler_box is not None and not use_track_roi:
                roi_used = bowler_box
                yolo_dets = self._run_yolo(
                    frame, roi_used, min_conf=yolo_min_conf
                )
                if not yolo_dets:
                    yolo_dets = self._run_yolo(
                        frame, None, min_conf=yolo_min_conf
                    )
                    roi_used = None
            elif use_track_roi:
                roi_used = _ball_roi_box(
                    anchor_for_roi[0], anchor_for_roi[1], fw, fh
                )
                yolo_dets = self._run_yolo(
                    frame, roi_used, min_conf=yolo_min_conf
                )
                if not yolo_dets:
                    self.roi_fail_streak += 1
                    yolo_dets = self._run_yolo(
                        frame, None, min_conf=yolo_min_conf
                    )
                    roi_used = None
                else:
                    self.roi_fail_streak = 0
            else:
                self.roi_fail_streak = 0
                yolo_dets = self._run_yolo(
                    frame, None, min_conf=yolo_min_conf
                )
        self.last_roi = roi_used
        self.last_yolo_dets = list(yolo_dets) if self.debug else []

        if run_yolo and yolo_dets:
            self.yolo_raw_det_frames += 1

        # Log raw detection centers, then (in flight) drop persistent static
        # false positives that this ball model fires on. The real in-flight
        # ball never lingers at one pixel, so this only removes background blobs.
        if run_yolo:
            for _cx, _cy, _cf in yolo_dets:
                self.static_hist.append((frame_idx, _cx, _cy))
        if post_release and yolo_dets:
            kept_dets = [
                d for d in yolo_dets
                if not self._is_static(d[0], d[1], frame_idx)
            ]
            if len(kept_dets) < len(yolo_dets):
                self.static_reject_frames += 1
            yolo_dets = kept_dets

        yolo_dets, n_rej = _filter_ball_dets_stats(yolo_dets, fw, fh, wrist_xy)
        if n_rej:
            self.rejected_geom_frames += 1

        if wrist_gate is not None and wrist_xy is not None and yolo_dets:
            wx, wy = wrist_xy
            yolo_dets = [
                d
                for d in yolo_dets
                if math.hypot(d[0] - wx, d[1] - wy) <= wrist_gate
            ]

        anchor = self.last if self.last is not None else wrist_xy
        vel_hint: tuple[float, float] | None = None
        if post_release and len(self.flight_pts) >= 2:
            f2, x2, y2 = self.flight_pts[-1]
            f1, x1, y1 = self.flight_pts[-2]
            df = max(1, int(f2 - f1))
            vel_hint = ((x2 - x1) / df, (y2 - y1) / df)
        meas = _pick_ball_measurement(
            yolo_dets, anchor, max_jump=max_jump, vel_hint=vel_hint
        )
        meas_source = "yolo" if meas is not None else ""

        # No motion/HSV until first YOLO lock (HSV picks up white crease).
        allow_fallback = False
        if not bootstrap:
            allow_fallback = run_yolo and (
                self.lost >= 2 or not self.ready or post_release
            )

        strong_yolo_flight = (
            post_release
            and len(self.flight_pts) >= 8
            and self.flight_yolo_meas >= 10
            and self.lost < 4
        )

        if meas is None and allow_fallback:
            search_roi = roi_used
            if search_roi is None and anchor is not None and not _ball_roi_near_edge(
                anchor[0], anchor[1], fw, fh
            ):
                search_roi = _ball_roi_box(anchor[0], anchor[1], fw, fh)
            motion_blobs = motion_ball_candidates(
                frame, self.prev_gray, search_roi
            )
            motion_blobs, m_rej = _filter_ball_dets_stats(
                [(cx, cy, sc) for cx, cy, sc in motion_blobs],
                fw,
                fh,
                wrist_xy,
            )
            if m_rej:
                self.rejected_geom_frames += 1
            if wrist_gate is not None and wrist_xy is not None:
                motion_blobs = [
                    (cx, cy, sc)
                    for (cx, cy, sc) in motion_blobs
                    if math.hypot(cx - wrist_xy[0], cy - wrist_xy[1]) <= wrist_gate
                ]
            if motion_blobs and anchor is not None:
                motion_blobs.sort(
                    key=lambda t: (t[0] - anchor[0]) ** 2 + (t[1] - anchor[1]) ** 2
                )
                cx, cy, sc = motion_blobs[0]
                if math.hypot(cx - anchor[0], cy - anchor[1]) < max_jump * 1.4:
                    meas = (cx, cy, max(0.22, sc))
                    meas_source = "motion"

            if (
                meas is None
                and not bootstrap
                and not strong_yolo_flight
                and (post_release or self.lost >= 3)
            ):
                blobs = hsv_ball_candidates(frame)
                blobs, h_rej = _filter_ball_dets_stats(
                    [(cx, cy, sc) for cx, cy, sc in blobs], fw, fh, wrist_xy
                )
                if h_rej:
                    self.rejected_geom_frames += 1
                if wrist_gate is not None and wrist_xy is not None:
                    blobs = [
                        (cx, cy, sc)
                        for (cx, cy, sc) in blobs
                        if math.hypot(cx - wrist_xy[0], cy - wrist_xy[1]) <= wrist_gate
                    ]
                if blobs and anchor is not None:
                    blobs.sort(
                        key=lambda t: (t[0] - anchor[0]) ** 2
                        + (t[1] - anchor[1]) ** 2
                    )
                    cx, cy, sc = blobs[0]
                    if math.hypot(cx - anchor[0], cy - anchor[1]) < 88:
                        meas = (cx, cy, max(0.22, sc))
                        meas_source = "hsv"
                elif blobs and anchor is None and not pre_release:
                    blobs.sort(key=lambda t: t[2], reverse=True)
                    cx, cy, sc = blobs[0]
                    meas = (cx, cy, max(0.22, sc))
                    meas_source = "hsv"

        if bootstrap and meas is not None:
            mx, my, mcf = meas
            if mcf < BALL_YOLO_CONF_PRIMARY:
                meas = None
                meas_source = ""
            elif wrist_xy is not None and math.hypot(
                mx - wrist_xy[0], my - wrist_xy[1]
            ) > BALL_BOOTSTRAP_WRIST_PX:
                meas = None
                meas_source = ""

        if (
            meas is not None
            and post_release
            and len(self.flight_pts) >= 5
        ):
            mx, my, mcf = meas
            resid = self._parabola_residual(frame_idx, mx, my)
            if resid > 32.0 and mcf < 0.65:
                meas = None

        spd = 0.0
        self.predicted = False
        if meas is not None:
            mx, my, mcf = meas
            plausible = _ball_plausible(self.last, (mx, my), max_jump)
            if not plausible and mcf < 0.62:
                meas = None

        if meas is not None:
            mx, my, mcf = meas
            plausible = _ball_plausible(self.last, (mx, my), max_jump)
            if not self.ready:
                _kalman_init(self.kf, mx, my)
                self.ready = True
                self.last = (mx, my)
                self.lost = 0
                self.conf = mcf
                self.max_conf = max(self.max_conf, mcf)
                if meas_source:
                    self.last_meas_source = meas_source
                    if meas_source == "yolo":
                        self.yolo_meas_frames += 1
                    elif meas_source == "motion":
                        self.motion_meas_frames += 1
                    elif meas_source == "hsv":
                        self.hsv_meas_frames += 1
            elif plausible or mcf >= 0.68:
                # We already called predict() above; correct() consumes that
                # prediction without advancing state again.
                c = self.kf.correct(
                    np.array([[np.float32(mx)], [np.float32(my)]])
                )
                self.last = (int(c[0]), int(c[1]))
                self.lost = 0
                self.conf = mcf
                self.max_conf = max(self.max_conf, mcf)
                if meas_source:
                    self.last_meas_source = meas_source
                    if meas_source == "yolo":
                        self.yolo_meas_frames += 1
                    elif meas_source == "motion":
                        self.motion_meas_frames += 1
                    elif meas_source == "hsv":
                        self.hsv_meas_frames += 1
            else:
                if kf_pred is not None:
                    self.last = kf_pred
                self.lost += 1
                self.predicted = True
                self.conf *= 0.88
        elif self.ready and self.lost < lost_cap:
            if kf_pred is not None:
                self.last = kf_pred
            self.lost += 1
            self.predicted = True
            self.conf *= 0.92
        else:
            if self.lost >= lost_cap:
                self.ready = False
                self.lost = 0
                self.last = None
                self.disp_xy = None
            self.conf = 0.0

        if (
            release_frame is not None
            and frame_idx >= release_frame
            and not self.release_locked
        ):
            self.release_locked = True
            self.flight_pts = []
            self._switch_to_ca_kalman()

        trail_xy: tuple[int, int] | None = None
        if self.last is not None:
            lx, ly = float(self.last[0]), float(self.last[1])
            if self.disp_xy is None:
                self.disp_xy = (lx, ly)
            else:
                a = BALL_DISP_SMOOTH
                self.disp_xy = (
                    a * lx + (1.0 - a) * self.disp_xy[0],
                    a * ly + (1.0 - a) * self.disp_xy[1],
                )
            trail_xy = (
                int(round(self.disp_xy[0])),
                int(round(self.disp_xy[1])),
            )
            record_flight = (
                release_frame is not None
                and frame_idx >= release_frame
                and not self.predicted
            )
            if record_flight:
                self.raw_pts.append((frame_idx, trail_xy[0], trail_xy[1]))
                self.flight_pts.append(
                    (frame_idx, trail_xy[0], trail_xy[1])
                )
                src = self.last_meas_source
                if src == "yolo":
                    self.flight_yolo_meas += 1
                elif src == "motion":
                    self.flight_motion_meas += 1
                elif src == "hsv":
                    self.flight_hsv_meas += 1

        if len(self.raw_pts) >= 2 and self.raw_pts[-1][0] == frame_idx:
            a, b = self.raw_pts[-2], self.raw_pts[-1]
            df = max(1, b[0] - a[0])
            spd = math.hypot(b[1] - a[1], b[2] - a[2]) / df

        # Cache grayscale for the next frame's motion fallback.
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return trail_xy or self.last, self.conf, spd, self.predicted


def spine_tilt_deg_at(lms: list[Any]) -> float | None:
    ls, rs = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
    lh, rh = lms[LEFT_HIP], lms[RIGHT_HIP]
    if min(lm_vis(ls), lm_vis(rs), lm_vis(lh), lm_vis(rh)) < 0.5:
        return None
    ms = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2], dtype=np.float64)
    mh = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2], dtype=np.float64)
    v = ms - mh
    ref = np.array([0.0, -1.0], dtype=np.float64)
    return angle_between_deg(v, ref)


def text_pill(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    org: tuple[int, int],
) -> None:
    x0, y0 = org
    fs = 0.55
    th = int(22 * fs * 2)
    tw = max(int(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]) for t, _ in lines)
    pad = 6
    cv2.rectangle(
        img,
        (x0, y0),
        (x0 + tw + 2 * pad, y0 + len(lines) * th + 2 * pad),
        (0, 0, 0),
        -1,
    )
    for i, (txt, col) in enumerate(lines):
        y = y0 + pad + (i + 1) * th - 6
        cv2.putText(img, txt, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)


def _estimate_mid_ankle_stride_amplitude(
    sm_mid: list[tuple[int, float]],
    end_frame_exclusive: int,
    fps: float,
) -> float:
    """Typical ankle-midpoint 'lift' amplitude from run-up gait (not delivery).

    Uses per-frame deltas of smoothed midpoint y during the portion of the
    clip BEFORE `end_frame_exclusive` (usually the estimated gather/start of
    the delivery stride). Stride hops show up as small oscillations — we use
    a high percentile of |dy| as the noise floor to reject gait from real jump.
    """
    ys = [(f, y) for f, y in sm_mid if f < end_frame_exclusive]
    if len(ys) < 12:
        return 0.02
    dv = []
    for j in range(1, len(ys)):
        dv.append(abs(ys[j][1] - ys[j - 1][1]))
    if not dv:
        return 0.02
    arr = np.array(dv, dtype=np.float64)
    pct = float(np.percentile(arr, 94))
    return max(pct * 4.6, 0.030)


def _detect_jump_window(
    sm_mid: list[tuple[int, float]],
    vel_mid: list[tuple[int, float]],
    baseline: float | None,
    fps: float,
    *,
    f_min: int | None = None,
    f_max_exclusive: int | None = None,
    min_amplitude_above_gait: float | None = None,
) -> tuple[int, int, int] | None:
    """Find the takeoff, apex, and landing frames of the delivery jump.

    Slides a ~0.4 s window over the smoothed ankle midpoint, fits a quadratic
    `y = a*t^2 + b*t + c`, and ranks windows by `a * amplitude_above_baseline`
    (large positive `a` = concave-up parabola in screen-y, i.e. ankles rise
    then fall — a real jump). Returns (takeoff_frame, apex_frame, landing_frame)
    or None if no convincing jump is found.

    If `f_max_exclusive` / `f_min` are set, only windows overlapping the
    [f_min, f_max_exclusive] frame range are considered — this prevents early
    run-up gait striding from falsely ending RUN_UP days before delivery.
    """
    n = len(sm_mid)
    if n < 6 or baseline is None:
        return None
    lo = max(0, int(f_min) if f_min is not None else 0)
    hi_excl = (
        int(f_max_exclusive) if f_max_exclusive is not None else sm_mid[-1][0] + 1
    )
    hi_excl = min(hi_excl, sm_mid[-1][0] + 1)
    # Map frame → index range for slicing y array
    fs = np.array([f for f, _ in sm_mid], dtype=np.int32)
    if lo >= hi_excl:
        return None
    win = max(6, int(round(0.4 * fps)))
    if win >= n:
        win = max(6, n // 2)
    best: tuple[float, int, int, int] | None = None
    ys_all = np.array([y for _, y in sm_mid], dtype=np.float64)
    half_amp_threshold = 0.034
    if min_amplitude_above_gait is not None:
        half_amp_threshold = max(half_amp_threshold, float(min_amplitude_above_gait))
    for i in range(0, n - win + 1):
        f_start = int(fs[i])
        f_end = int(fs[min(i + win - 1, n - 1)])
        if f_end < lo or f_start >= hi_excl:
            continue
        seg = ys_all[i: i + win]
        if seg.min() >= baseline - 0.01:
            continue
        amplitude = float(baseline - seg.min())
        if amplitude < half_amp_threshold:
            continue
        x = np.arange(win, dtype=np.float64)
        a, b, c = np.polyfit(x, seg, 2)
        if a <= 0:
            continue
        score = float(a) * amplitude
        if best is None or score > best[0]:
            apex_local = int(np.argmin(seg))
            best = (score, i, apex_local, i + win - 1)
    if best is None:
        return None
    _, win_start_idx, apex_local, win_end_idx = best
    apex_idx = win_start_idx + apex_local
    takeoff_idx = win_start_idx
    for k in range(apex_idx - 1, -1, -1):
        if sm_mid[k][1] >= baseline - 0.01:
            takeoff_idx = k + 1
            break
        if k == 0:
            takeoff_idx = 0
    landing_idx = win_end_idx
    for k in range(apex_idx + 1, n):
        if sm_mid[k][1] >= baseline - 0.01:
            landing_idx = k
            break
    if takeoff_idx >= apex_idx or apex_idx >= landing_idx:
        return None
    takeoff_f = sm_mid[takeoff_idx][0]
    landing_f = sm_mid[landing_idx][0]
    # Run-up steps are a few frames; a delivery hop spans noticeably longer.
    if landing_f - takeoff_f < max(9, int(0.11 * fps)):
        return None
    return (takeoff_f, sm_mid[apex_idx][0], landing_f)


def _detect_foot_strike(
    sm: list[tuple[int, float]],
    vel: list[tuple[int, float]],
    baseline: float | None,
    landing_frame: int,
    fps: float,
    f_min: int | None = None,
    f_max: int | None = None,
) -> int | None:
    """Find a foot-strike frame: the foot has just come down (velocity was
    strongly positive then drops to ~0) AND the ankle is near the floor.

    Returns the frame closest to `landing_frame` that satisfies both
    conditions inside the [f_min, f_max] window.
    """
    if not sm or not vel or baseline is None:
        return None
    if f_min is None:
        f_min = landing_frame - int(fps * 0.5)
    if f_max is None:
        f_max = landing_frame + int(fps * 0.5)
    cand: list[tuple[int, float]] = []
    fs = [f for f, _ in sm]
    ys = {f: y for f, y in sm}
    vs = {f: v for f, v in vel}
    for k in range(2, len(fs) - 1):
        f = fs[k]
        if f < f_min or f > f_max:
            continue
        if f not in vs or f not in ys:
            continue
        v_now = vs[f]
        v_prev = vs.get(fs[k - 1], 0.0)
        v_prev2 = vs.get(fs[k - 2], 0.0)
        peak_descent = max(v_prev, v_prev2)
        if peak_descent < 0.004:
            continue
        if abs(v_now) > 0.005:
            continue
        if ys[f] < baseline - 0.04:
            continue
        score = abs(f - landing_frame)
        cand.append((f, score))
    if not cand:
        return None
    cand.sort(key=lambda t: t[1])
    return cand[0][0]


def _detect_release_by_arm_angle(
    wrist_y: list[tuple[int, float]],
    wrist_x: list[tuple[int, float]],
    shoulder_x: list[tuple[int, float]],
    shoulder_y: list[tuple[int, float]],
    ffc: int,
    fps: float,
) -> int | None:
    """Detect RELEASE as the frame the wrist-shoulder vector passes vertical.

    Operates within (ffc, ffc + 0.4s). The angle goes from positive (arm
    behind/above) to negative (arm in front/coming down) as the bowler
    completes the over-the-top action; the zero-crossing is the moment the
    arm is straight up — and the ball leaves the hand within ±1 frame.
    """
    if not wrist_y or not wrist_x or not shoulder_x or not shoulder_y or ffc is None:
        return None
    wy = {f: y for f, y in wrist_y}
    wx = {f: x for f, x in wrist_x}
    sx = {f: x for f, x in shoulder_x}
    sy = {f: y for f, y in shoulder_y}
    common = sorted(set(wy) & set(wx) & set(sx) & set(sy))
    if not common:
        return None
    f_min = ffc
    f_max = ffc + int(fps * 0.4)
    win_frames = [f for f in common if f_min <= f <= f_max]
    if len(win_frames) < 3:
        return None
    angles: list[tuple[int, float, float]] = []
    for f in win_frames:
        theta = math.atan2(wx[f] - sx[f], sy[f] - wy[f])
        dx = wx[f] - wx.get(f - 1, wx[f])
        angles.append((f, theta, dx))
    near_vertical = sorted(angles, key=lambda t: abs(t[1]))
    for f, theta, dx in near_vertical:
        if dx >= -0.002:
            return f
    return near_vertical[0][0]


def _wrist_release_gap_threshold(
    sm_wrist_y: list[tuple[int, float]],
    sm_shoulder_y: list[tuple[int, float]],
    fps: float,
) -> float:
    """Stricter gap for flat / low cameras where run-up arm carriage mimics delivery."""
    base = 0.056
    if not sm_wrist_y or not sm_shoulder_y:
        return base
    sy_map = {f: y for f, y in sm_shoulder_y}
    f0, f1 = sm_wrist_y[0][0], sm_wrist_y[-1][0]
    mid = f0 + int(0.38 * max(1, f1 - f0))
    noisy = 0
    total = 0
    for f, wy in sm_wrist_y:
        if f > mid:
            break
        sy = sy_map.get(f)
        if sy is None:
            continue
        total += 1
        if sy - wy > 0.034:
            noisy += 1
    if total >= 20 and noisy / total > 0.14:
        return max(base, 0.072)
    return base


def _detect_release_by_wrist_apex(
    sm_wrist_y: list[tuple[int, float]],
    sm_shoulder_y: list[tuple[int, float]],
    fps: float,
    min_above_shoulder: float | None = None,
) -> int | None:
    """Detect RELEASE as the wrist's global apex while clearly above shoulder.

    The bowling-arm wrist is below or at the shoulder during the entire
    run-up. It only goes meaningfully above the shoulder during the bowling
    action. We pick the global minimum (highest screen position) of wrist y
    among frames where (shoulder_y - wrist_y) > min_above_shoulder. This is
    the most distinctive single signal in any bowling video and works even
    when the bowler doesn't jump.
    """
    if not sm_wrist_y or not sm_shoulder_y:
        return None
    if min_above_shoulder is None:
        min_above_shoulder = _wrist_release_gap_threshold(
            sm_wrist_y, sm_shoulder_y, fps
        )
    sy_map = {f: y for f, y in sm_shoulder_y}
    f_last = sm_wrist_y[-1][0]
    f_first = sm_wrist_y[0][0]
    span = max(1, f_last - f_first)
    tail = min(5.8, max(2.6, (span / max(fps, 1e-6)) * 0.48))
    f_cand_min = int(f_last - tail * fps)
    cand: list[tuple[int, float]] = []
    for f, wy in sm_wrist_y:
        if f < f_cand_min:
            continue
        sy = sy_map.get(f)
        if sy is None:
            continue
        gap = sy - wy
        if gap > min_above_shoulder:
            cand.append((f, wy))
    if not cand:
        return None
    return min(cand, key=lambda t: t[1])[0]


def _find_local_max_in_window(
    sm: list[tuple[int, float]],
    f_min: int,
    f_max: int,
    fps: float,
    *,
    baseline: float | None = None,
    near_baseline_tol: float = 0.04,
) -> int | None:
    """Find the most recent (latest) plausible foot-strike in [f_min, f_max].

    A foot-strike is a frame where the ankle y is at a local maximum AND
    near the floor (within `near_baseline_tol` of `baseline`). Prefers the
    latest qualifying frame so we pick the delivery-stride plant, not an
    earlier run-up plant.
    """
    if not sm:
        return None
    win = [(f, y) for f, y in sm if f_min <= f <= f_max]
    if not win:
        return None
    if len(win) < 3:
        return max(win, key=lambda t: t[1])[0]
    qual: list[tuple[int, float]] = []
    for i in range(1, len(win) - 1):
        f, y = win[i]
        if y < win[i - 1][1] - 1e-6 or y < win[i + 1][1] - 1e-6:
            continue
        if baseline is not None and y < baseline - near_baseline_tol:
            continue
        qual.append((f, y))
    if not qual:
        # No local max found — fall back to global argmax in the window.
        return max(win, key=lambda t: t[1])[0]
    return qual[-1][0]


def detect_events_from_series(
    b_ankle: list[tuple[int, float]],
    nb_ankle: list[tuple[int, float]],
    mid_ankle: list[tuple[int, float]],
    wrist: list[tuple[int, float]],
    fps: float,
    *,
    wrist_x: list[tuple[int, float]] | None = None,
    shoulder_x: list[tuple[int, float]] | None = None,
    shoulder_y: list[tuple[int, float]] | None = None,
) -> dict[str, int | None]:
    """Compute (IMPULSE, JUMP_END, BFC, FFC, RELEASE) frames from full-video
    series using kinematic detectors (velocity + baseline + parabolic-dip fit
    + wrist-arc release). Falls back to the legacy plateau detectors per-event
    when a sanity gate fails.
    """
    events: dict[str, int | None] = {
        "IMPULSE": None,
        "JUMP_END": None,
        "LANDING": None,
        "BFC": None,
        "FFC": None,
        "RELEASE": None,
    }

    sm_b = smooth_series(b_ankle, fps)
    sm_nb = smooth_series(nb_ankle, fps)
    sm_mid = smooth_series(mid_ankle, fps)
    sm_w = smooth_series(wrist, fps)
    sm_sh_y = smooth_series(shoulder_y, fps) if shoulder_y else []
    vel_b = smoothed_velocity(sm_b)
    vel_nb = smoothed_velocity(sm_nb)
    vel_mid = smoothed_velocity(sm_mid)

    base_b = compute_ankle_baseline(sm_b, fps)
    base_nb = compute_ankle_baseline(sm_nb, fps)
    base_mid = compute_ankle_baseline(sm_mid, fps)

    def _fmt(v: float | None) -> str:
        return "None" if v is None else f"{v:.3f}"

    print(
        f"Phase pre-pass baselines: b={_fmt(base_b)} "
        f"nb={_fmt(base_nb)} mid={_fmt(base_mid)}"
    )

    wrist_x_list = wrist_x or []
    shoulder_x_list = shoulder_x or []
    shoulder_y_list = shoulder_y or []

    def _print_release_angle(rel: int | None) -> None:
        if rel is None or not wrist_x_list or not shoulder_x_list:
            return
        wx = {f: x for f, x in wrist_x_list}
        sx = {f: x for f, x in shoulder_x_list}
        sy = {f: y for f, y in shoulder_y_list}
        wy = {f: y for f, y in wrist}
        if rel in wx and rel in sx and rel in sy and rel in wy:
            theta_deg = math.degrees(
                math.atan2(wx[rel] - sx[rel], sy[rel] - wy[rel])
            )
            print(
                f"Phase pre-pass: angle at RELEASE frame {rel} = "
                f"{theta_deg:.1f}° (0° = vertical)"
            )

    # ------------------------------------------------------------------
    # Anchor RELEASE (wrist apex), then derive FFC/BFC backwards — stable
    # on flat-angle clips. Try wrist-shoulder angle first when FFC known.
    # ------------------------------------------------------------------
    release: int | None = None
    if sm_w and sm_sh_y:
        release = _detect_release_by_wrist_apex(sm_w, sm_sh_y, fps)
    if release is not None:
        print(
            f"Phase pre-pass: RELEASE anchored at wrist apex frame {release}"
        )
    elif sm_w:
        f_last = sm_w[-1][0]
        f_first = sm_w[0][0]
        span = max(1, f_last - f_first)
        tail = min(5.8, max(2.6, (span / max(fps, 1e-6)) * 0.48))
        f0 = int(f_last - tail * fps)
        pool = [(f, y) for f, y in sm_w if f >= f0] or sm_w
        release = min(pool, key=lambda t: t[1])[0]
        print(
            f"Phase pre-pass: RELEASE fell back to global wrist min "
            f"(frame {release})"
        )
    events["RELEASE"] = release

    ffc: int | None = None
    if release is not None:
        ff_min = release - int(fps * 0.6)
        ff_max = release
        ffc = _detect_foot_strike(
            sm_nb, vel_nb, base_nb, release, fps,
            f_min=ff_min, f_max=ff_max,
        )
        if ffc is None:
            ffc = _find_local_max_in_window(
                sm_nb, ff_min, ff_max, fps, baseline=base_nb
            )
            if ffc is not None:
                print(
                    f"Phase pre-pass: FFC fell back to local-max in window "
                    f"(frame {ffc})"
                )
        if ffc is None:
            ffc = try_detect_ffc(nb_ankle, release, fps)
            if ffc is not None:
                print(
                    f"Phase pre-pass: FFC fell back to legacy plateau detector "
                    f"(frame {ffc})"
                )
    events["FFC"] = ffc

    if (
        ffc is not None
        and wrist_x_list
        and shoulder_x_list
        and shoulder_y_list
    ):
        angle_release = _detect_release_by_arm_angle(
            wrist,
            wrist_x_list,
            shoulder_x_list,
            shoulder_y_list,
            ffc,
            fps,
        )
        if (
            angle_release is not None
            and ffc < angle_release <= ffc + int(fps * 0.4)
        ):
            events["RELEASE"] = angle_release
            release = angle_release
            print(
                f"Phase pre-pass: RELEASE refined via wrist-shoulder angle "
                f"(frame {release})"
            )

    bfc: int | None = None
    anchor = ffc if ffc is not None else release
    if anchor is not None:
        bf_min = anchor - int(fps * 0.6)
        bf_max = anchor - 1 if ffc is not None else anchor
        bfc = _detect_foot_strike(
            sm_b, vel_b, base_b, anchor, fps,
            f_min=bf_min, f_max=bf_max,
        )
        if bfc is None:
            bfc = _find_local_max_in_window(
                sm_b, bf_min, bf_max, fps, baseline=base_b
            )
            if bfc is not None:
                print(
                    f"Phase pre-pass: BFC fell back to local-max in window "
                    f"(frame {bfc})"
                )
        if bfc is None:
            bfc = try_detect_bfc(b_ankle, fps, min_frame=max(0, bf_min))
            if bfc is not None and bf_min <= bfc <= bf_max:
                print(
                    f"Phase pre-pass: BFC fell back to legacy plateau detector "
                    f"(frame {bfc})"
                )
            elif bfc is not None:
                bfc = None
    events["BFC"] = bfc

    gather_lookback = max(int(fps * 0.55), 18)
    warmup_frames = max(int(fps * 0.35), 25)
    jump = None
    if bfc is not None and release is not None:
        gather_start = max(0, min(bfc, release) - gather_lookback)
        f_max_jump = bfc
        jump_strip_frames = max(int(fps * 0.40), 14)
        f_min_jump = max(
            warmup_frames,
            gather_start - int(fps * 0.08),
            bfc - jump_strip_frames,
        )
        stride_noise = _estimate_mid_ankle_stride_amplitude(
            sm_mid, gather_start, fps
        )
        print(
            f"Phase pre-pass: jump search f∈[{f_min_jump},{f_max_jump}), "
            f"gait stride noise floor (mid-ankle Δ)={stride_noise:.4f}"
        )
        jump = _detect_jump_window(
            sm_mid,
            vel_mid,
            base_mid,
            fps,
            f_min=f_min_jump,
            f_max_exclusive=f_max_jump,
            min_amplitude_above_gait=stride_noise,
        )

    if jump is not None and bfc is not None:
        ji, ja, jl = jump
        if (
            jl - ji < max(9, int(0.10 * fps))
            or ja <= ji
            or ja >= jl
            or jl > bfc
            or bfc - ji > int(0.50 * fps)
        ):
            print(
                "Phase pre-pass: delivery jump failed plausibility vs BFC; "
                "discarding"
            )
            jump = None

    if jump is not None:
        impulse, _apex, landing = jump
        events["IMPULSE"] = impulse
        events["JUMP_END"] = landing
        events["LANDING"] = landing
        print(
            f"Phase pre-pass: delivery jump IMPULSE={impulse} "
            f"LANDING/JUMP_END={landing}"
        )
    elif bfc is not None:
        gather_dur = max(2, int(round(0.22 * fps)))
        events["IMPULSE"] = max(0, bfc - gather_dur)
        events["JUMP_END"] = bfc
        events["LANDING"] = bfc
        print(
            f"Phase pre-pass: no delivery jump in gather window; synthesized "
            f"gather IMPULSE={events['IMPULSE']} LANDING={bfc}"
        )
    else:
        print("Phase pre-pass: jump and BFC both missing; phases will be empty")

    _print_release_angle(events.get("RELEASE"))

    impulse = events.get("IMPULSE")
    landing = events.get("LANDING")
    bfc = events.get("BFC")
    ffc = events.get("FFC")
    release = events.get("RELEASE")

    if impulse is not None and landing is not None and impulse >= landing:
        print("Phase pre-pass WARN: IMPULSE >= LANDING; clearing IMPULSE")
        events["IMPULSE"] = None

    if bfc is not None and ffc is not None and ffc <= bfc:
        print(
            f"Phase pre-pass WARN: FFC ({ffc}) not after BFC ({bfc}); "
            f"clearing FFC"
        )
        events["FFC"] = try_detect_ffc(nb_ankle, bfc, fps)
    elif bfc is not None and ffc is not None and ffc - bfc > int(fps * 0.6):
        print(
            f"Phase pre-pass WARN: BFC->FFC gap > 0.6s ({ffc - bfc} frames)"
        )

    if (
        ffc is not None
        and release is not None
        and (release <= ffc or release - ffc > int(fps * 0.6))
    ):
        print(
            f"Phase pre-pass WARN: RELEASE ({release}) outside (FFC, FFC+0.6s)"
        )

    return events


def precompute_phase_events(
    cfr_path: str,
    fw: int,
    fh: int,
    fps: float,
    bowling_arm: str,
    entry_side: str,
    person_model: YOLO,
    bowler_calibration_path: str | None = None,
) -> dict[str, int | None]:
    """First pass over the entire video to build deterministic event timings.

    Runs person YOLO + pose to extract bowling-arm ankle, non-bowling ankle,
    ankle-midpoint, and bowling-hand wrist y-series, then derives events.
    """
    cap = cv2.VideoCapture(cfr_path)
    if not cap.isOpened():
        return {"IMPULSE": None, "BFC": None, "FFC": None, "RELEASE": None}

    pose_est = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    bt_pre = BowlerTracker(
        person_model,
        bowling_arm,
        entry_side,
        fw,
        fh,
        exclusion_zones=load_bowler_exclusion_zones(bowler_calibration_path),
    )

    b_idx = RIGHT_ANKLE if bowling_arm == "right" else LEFT_ANKLE
    nb_idx = LEFT_ANKLE if bowling_arm == "right" else RIGHT_ANKLE
    w_idx = RIGHT_WRIST if bowling_arm == "right" else LEFT_WRIST
    sh_idx = RIGHT_SHOULDER if bowling_arm == "right" else LEFT_SHOULDER

    b_ankle: list[tuple[int, float]] = []
    nb_ankle: list[tuple[int, float]] = []
    mid_ankle: list[tuple[int, float]] = []
    wrist_y: list[tuple[int, float]] = []
    wrist_x: list[tuple[int, float]] = []
    shoulder_y: list[tuple[int, float]] = []
    shoulder_x: list[tuple[int, float]] = []

    empty_events = {"IMPULSE": None, "BFC": None, "FFC": None, "RELEASE": None}
    fi = 0
    last_lms: list[Any] | None = None
    prev_pose_box: tuple[int, int, int, int] | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = bt_pre.update(fi, frame, "RUN_UP", empty_events, ball_pos=None)
        lms = extract_pose_landmarks(
            frame, box, pose_est, fw, fh, entry_side=entry_side
        )
        if lms is None and last_lms is not None:
            if _pose_reuse_ok(box, prev_pose_box, fw, fh):
                lms = last_lms
        elif lms is not None:
            last_lms = lms
        prev_pose_box = box
        if lms is not None:
            la, ra = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]
            ba, nba = lms[b_idx], lms[nb_idx]
            wlm = lms[w_idx]
            shlm = lms[sh_idx]
            if lm_vis(ba) > 0.5:
                b_ankle.append((fi, float(ba.y)))
            if lm_vis(nba) > 0.5:
                nb_ankle.append((fi, float(nba.y)))
            if lm_vis(la) > 0.5 and lm_vis(ra) > 0.5:
                mid_ankle.append((fi, float((la.y + ra.y) / 2)))
            if lm_vis(wlm) > 0.4:
                wrist_y.append((fi, float(wlm.y)))
                wrist_x.append((fi, float(wlm.x)))
            if lm_vis(shlm) > 0.4:
                shoulder_y.append((fi, float(shlm.y)))
                shoulder_x.append((fi, float(shlm.x)))
        fi += 1

    cap.release()
    pose_est.close()

    return detect_events_from_series(
        b_ankle,
        nb_ankle,
        mid_ankle,
        wrist_y,
        fps,
        wrist_x=wrist_x,
        shoulder_x=shoulder_x,
        shoulder_y=shoulder_y,
    )


def analyse_video(
    video_path: str,
    out_video: str,
    out_json: str,
    bowling_arm: str,
    entry_side: str,
    ball_model_path: str,
    debug_ball: bool = False,
    bowler_calibration_json: str | None = None,
) -> dict[str, Any]:
    bowling_arm = bowling_arm.strip().lower()
    entry_side = entry_side.strip().lower()
    cfr_path, fps, tmpdir = ensure_cfr_input(video_path)
    cap = cv2.VideoCapture(cfr_path)
    if not cap.isOpened():
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("Could not open video")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    person_model = YOLO("yolov8n.pt")
    # Ball tracking removed: the ball detector/trajectory was producing
    # unreliable lines on the output video, so it is no longer loaded or run.
    pose_est = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pinned = precompute_phase_events(
        cfr_path,
        fw,
        fh,
        fps,
        bowling_arm,
        entry_side,
        person_model,
        bowler_calibration_path=bowler_calibration_json,
    )
    print(
        "Pre-pass events: "
        f"IMPULSE={pinned.get('IMPULSE')} LANDING={pinned.get('LANDING')} "
        f"JUMP_END={pinned.get('JUMP_END')} BFC={pinned.get('BFC')} "
        f"FFC={pinned.get('FFC')} RELEASE={pinned.get('RELEASE')}"
    )
    pinned_for_phase = pinned if pinned.get("BFC") is not None else None
    bt = BowlerTracker(
        person_model,
        bowling_arm,
        entry_side,
        fw,
        fh,
        exclusion_zones=load_bowler_exclusion_zones(bowler_calibration_json),
    )
    ph = PhaseDetector(
        fps, bowling_arm, fw, fh, pinned_events=pinned_for_phase
    )
    writer = make_writer(out_video, fps, fw, fh)

    run_trail: deque[tuple[int, int]] = deque(maxlen=max(3, int(fps * 3)))
    frozen_trail: list[tuple[int, int]] = []
    mid_hip_hist: list[tuple[int, float, float]] = []
    shoulder_w_sum = 0.0
    shoulder_w_n = 0
    strides: list[dict[str, Any]] = []
    last_stride_f = -999999
    elbow_peak = 0.0
    wrist_flare_sum = 0.0
    wrist_flare_n = 0
    worst_elbow_f: int | None = None
    spine_buf: deque[float] = deque(maxlen=7)
    spine_series: list[tuple[int, float]] = []
    # Back-view-reliable orientation/displacement series (fi -> value).
    shoulder_line_series: list[tuple[int, float]] = []  # signed shoulder-line angle vs horizontal (deg)
    hip_line_series: list[tuple[int, float]] = []        # signed hip-line angle vs horizontal (deg)
    sep_series: list[tuple[int, float]] = []             # |shoulder - hip| reduced to [0,90] (deg)
    head_x_series: list[tuple[int, float]] = []          # nose x in px (lateral position)
    midhip_x_series: list[tuple[int, float]] = []        # mid-hip x in px (lateral position)
    loadup_done = False
    loadup_report: dict[str, Any] = {}
    bfc_overlay_until: int | None = None
    last_lms: list[Any] | None = None
    prev_pose_box: tuple[int, int, int, int] | None = None
    frozen_arrow: tuple[tuple[int, int], tuple[int, int]] | None = None
    ay_buf: deque[tuple[int, float, str]] = deque(maxlen=5)
    ankle_trace: list[tuple[int, int, int, int, int]] = []
    frozen_ankle_trace: list[tuple[int, int, int, int, int]] = []
    prev_ball_px: tuple[int, int] | None = None
    ft_release_anchor_px: tuple[int, int] | None = None
    ft_ideal_dir: tuple[float, float] | None = None
    ft_actual_dir: tuple[float, float] | None = None
    ft_label: str = ""
    ft_label_col: tuple[int, int, int] = (255, 255, 255)
    ft_computed: bool = False
    live_straight_label: str = ""
    live_straight_col: tuple[int, int, int] = (255, 255, 255)
    live_arm_label: str = ""
    live_arm_col: tuple[int, int, int] = (255, 255, 255)
    live_final_stride_warn: bool = False
    live_overstride: bool = False
    live_understride: bool = False
    live_coach_feedback: list[tuple[str, tuple[int, int, int]]] = []
    coach_seen: set[str] = set()

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if ph.pinned_events is not None:
            phase = ph._pinned_phase_for(fi)
            for k in ("IMPULSE", "JUMP_END", "LANDING", "BFC", "FFC", "RELEASE"):
                f_pin = ph.pinned_events.get(k)
                if f_pin is not None and fi >= f_pin and ph.events.get(k) is None:
                    ph.events[k] = f_pin
        else:
            phase = ph.phase
        box = bt.update(fi, frame, phase, ph.events, ball_pos=prev_ball_px)
        lms = extract_pose_landmarks(
            frame, box, pose_est, fw, fh, entry_side=entry_side
        )
        if lms is None and last_lms is not None:
            if _pose_reuse_ok(box, prev_pose_box, fw, fh):
                lms = last_lms
        elif lms is not None:
            last_lms = lms
        prev_pose_box = box

        # Ball tracking removed: no ball detection, trajectory, or trail.
        ph.update(fi, lms, None, 0.0, 0.0)

        if lms is not None and shoulder_w_n < 10:
            ls, rs = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
            if lm_vis(ls) > 0.5 and lm_vis(rs) > 0.5:
                d = math.hypot((rs.x - ls.x) * fw, (rs.y - ls.y) * fh)
                shoulder_w_sum += d
                shoulder_w_n += 1
        shoulder_w_px = (
            shoulder_w_sum / max(1, shoulder_w_n) if shoulder_w_n else float(fw) * 0.2
        )

        if lms is not None:
            lh, rh = lms[LEFT_HIP], lms[RIGHT_HIP]
            if lm_vis(lh) > 0.5 and lm_vis(rh) > 0.5:
                mx = (lh.x + rh.x) / 2 * fw
                my = (lh.y + rh.y) / 2 * fh
                mid_hip_hist.append((fi, mx, my))
                midhip_x_series.append((fi, mx))
                if ph.events["BFC"] is None:
                    run_trail.append((int(mx), int(my)))
                elif not frozen_trail and ph.events["BFC"] is not None:
                    frozen_trail = list(run_trail)

            # Back-view orientation/lateral series (transverse-plane proxies).
            lsh, rsh = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
            if lm_vis(lsh) > 0.5 and lm_vis(rsh) > 0.5:
                sh_ang = math.degrees(
                    math.atan2((rsh.y - lsh.y) * fh, (rsh.x - lsh.x) * fw)
                )
                if sh_ang > 90.0:
                    sh_ang -= 180.0
                elif sh_ang < -90.0:
                    sh_ang += 180.0
                shoulder_line_series.append((fi, sh_ang))
                if lm_vis(lh) > 0.5 and lm_vis(rh) > 0.5:
                    hip_ang = math.degrees(
                        math.atan2((rh.y - lh.y) * fh, (rh.x - lh.x) * fw)
                    )
                    if hip_ang > 90.0:
                        hip_ang -= 180.0
                    elif hip_ang < -90.0:
                        hip_ang += 180.0
                    hip_line_series.append((fi, hip_ang))
                    sep = abs(sh_ang - hip_ang)
                    if sep > 90.0:
                        sep = 180.0 - sep
                    sep_series.append((fi, sep))
            nose = lms[NOSE]
            if lm_vis(nose) > 0.5:
                head_x_series.append((fi, nose.x * fw))

        if ph.events["BFC"] is None and len(mid_hip_hist) >= 8:
            xs_live = [p[1] for p in mid_hip_hist]
            ys_live = [p[2] for p in mid_hip_hist]
            try:
                coef_live = np.polyfit(xs_live, ys_live, 1)
                devs_live = [
                    abs(y - (coef_live[0] * x + coef_live[1])) / fh
                    for x, y in zip(xs_live, ys_live)
                ]
                live_straight = 1.0 - min(1.0, float(np.mean(devs_live)) / 0.05)
            except (np.linalg.LinAlgError, ValueError):
                live_straight = 1.0
            if live_straight >= 0.85:
                live_straight_label = "Run-up line: Straight"
                live_straight_col = (0, 255, 0)
            elif live_straight >= 0.65:
                live_straight_label = "Run-up line: Slight drift"
                live_straight_col = (0, 165, 255)
            else:
                live_straight_label = "Run-up line: Drifting"
                live_straight_col = (0, 0, 255)

        if lms is not None and ph.phase == "RUN_UP":
            la, ra = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]
            la_vis = lm_vis(la) > 0.5
            ra_vis = lm_vis(ra) > 0.5
            if la_vis or ra_vis:
                lx_px = int(la.x * fw) if la_vis else -1
                ly_px = int(la.y * fh) if la_vis else -1
                rx_px = int(ra.x * fw) if ra_vis else -1
                ry_px = int(ra.y * fh) if ra_vis else -1
                ankle_trace.append((fi, lx_px, ly_px, rx_px, ry_px))
        elif (
            ph.events["BFC"] is not None
            and not frozen_ankle_trace
            and ankle_trace
        ):
            frozen_ankle_trace = list(ankle_trace)

        if lms is not None and ph.phase == "RUN_UP":
            la_s, ra_s = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]
            if lm_vis(la_s) > 0.5 and lm_vis(ra_s) > 0.5:
                ay = (la_s.y + ra_s.y) / 2
                ay_buf.append((fi, ay, "L" if la_s.y >= ra_s.y else "R"))
                if len(ay_buf) >= 3:
                    f0, y0, _ = ay_buf[-3]
                    f1, y1, _ = ay_buf[-2]
                    f2, y2, s2 = ay_buf[-1]
                    if y2 >= y1 and y2 >= y0 and y2 - min(y0, y1) > 0.003:
                        if fi - last_stride_f >= int(fps * 0.10):
                            ax = (la_s.x + ra_s.x) / 2
                            ay_mid = (la_s.y + ra_s.y) / 2
                            strides.append(
                                {
                                    "frame": f2,
                                    "x": ax,
                                    "y": ay_mid,
                                    "side": s2,
                                }
                            )
                            last_stride_f = f2

            if shoulder_w_px > 1e-6 and len(strides) >= 4:
                live_lens: list[float] = []
                for k in range(1, len(strides)):
                    p0 = strides[k - 1]
                    p1 = strides[k]
                    d_live = math.hypot(
                        (p1["x"] - p0["x"]) * fw,
                        (p1["y"] - p0["y"]) * fh,
                    )
                    live_lens.append(d_live / shoulder_w_px)
                if len(live_lens) >= 3:
                    usable = live_lens[1: min(len(live_lens), 5)]
                    ref_live = (
                        float(np.median(usable))
                        if usable else 1.0
                    )
                    ratios_live = (
                        [s / ref_live for s in live_lens]
                        if ref_live > 1e-6 else []
                    )
                    if len(ratios_live) >= 3:
                        live_final_stride_warn = any(
                            abs(ratios_live[k] - 1.0) > 0.25
                            for k in range(
                                max(0, len(ratios_live) - 3),
                                len(ratios_live),
                            )
                        )
                    last_len = live_lens[-1]
                    live_overstride = last_len > ref_live * 1.30
                    live_understride = last_len < ref_live * 0.70

        if lms is not None and ph.events["BFC"] is None:
            le, re = lms[LEFT_ELBOW], lms[RIGHT_ELBOW]
            ls, rs = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
            if min(lm_vis(le), lm_vis(re), lm_vis(ls), lm_vis(rs)) > 0.5:
                ew = math.hypot((re.x - le.x) * fw, (re.y - le.y) * fh)
                sw = math.hypot((rs.x - ls.x) * fw, (rs.y - ls.y) * fh)
                if sw > 1e-6:
                    ratio = ew / sw
                    if ratio > elbow_peak:
                        elbow_peak = ratio
                        worst_elbow_f = fi
                    if elbow_peak < 1.1:
                        live_arm_label = "Arm Alignment: Good"
                        live_arm_col = (0, 255, 0)
                    elif elbow_peak < 1.4:
                        live_arm_label = "Arm Alignment: Moderate"
                        live_arm_col = (0, 165, 255)
                    else:
                        live_arm_label = "Arm Alignment: Excessive"
                        live_arm_col = (0, 0, 255)
                    lw, rw = lms[LEFT_WRIST], lms[RIGHT_WRIST]
                    lhip, rhip = lms[LEFT_HIP], lms[RIGHT_HIP]
                    if min(lm_vis(lw), lm_vis(rw), lm_vis(lhip), lm_vis(rhip)) > 0.5:
                        wf = max(
                            abs(lw.x - lhip.x),
                            abs(rw.x - rhip.x),
                        ) * fw / sw
                        wrist_flare_sum += wf
                        wrist_flare_n += 1

        st_deg = None
        if lms is not None and ph.phase in ("DELIVERY", "RELEASE"):
            sd = spine_tilt_deg_at(lms)
            if sd is not None:
                spine_buf.append(sd)
                if len(spine_buf):
                    st_deg = sum(spine_buf) / len(spine_buf)
                    spine_series.append((fi, st_deg))

        evb, evf, evr = ph.events["BFC"], ph.events["FFC"], ph.events["RELEASE"]

        if (
            evr is not None
            and not ft_computed
            and fi >= evr
            and lms is not None
        ):
            la_lm_ft = lms[LEFT_ANKLE]
            ra_lm_ft = lms[RIGHT_ANKLE]
            if lm_vis(la_lm_ft) > 0.4 and lm_vis(ra_lm_ft) > 0.4:
                ft_release_anchor_px = (
                    int((la_lm_ft.x + ra_lm_ft.x) / 2 * fw),
                    int((la_lm_ft.y + ra_lm_ft.y) / 2 * fh),
                )
                ft_ideal_dir = _ideal_pitch_forward_dir(
                    ft_release_anchor_px[0],
                    ft_release_anchor_px[1],
                    fw,
                    fh,
                )
            ft_computed = True

        if (
            evr is not None
            and ft_release_anchor_px is not None
            and ft_ideal_dir is not None
            and fi > evr
            and lms is not None
            and ph.phase == "FOLLOWTHROUGH"
        ):
            la_lm_ft = lms[LEFT_ANKLE]
            ra_lm_ft = lms[RIGHT_ANKLE]
            if lm_vis(la_lm_ft) > 0.4 and lm_vis(ra_lm_ft) > 0.4:
                curr_ankle_px = (
                    int((la_lm_ft.x + ra_lm_ft.x) / 2 * fw),
                    int((la_lm_ft.y + ra_lm_ft.y) / 2 * fh),
                )
                move_dx = float(curr_ankle_px[0] - ft_release_anchor_px[0])
                move_dy = float(curr_ankle_px[1] - ft_release_anchor_px[1])
                if math.hypot(move_dx, move_dy) > 6:
                    ft_actual_dir = norm_vec(move_dx, move_dy)
                    al = float(
                        ft_actual_dir[0] * ft_ideal_dir[0]
                        + ft_actual_dir[1] * ft_ideal_dir[1]
                    )
                    se = abs(cross_z(ft_actual_dir, ft_ideal_dir))
                    if al >= 0.85 and se <= 0.25:
                        ft_label = "Follow-through: Good"
                        ft_label_col = (0, 255, 0)
                    elif al >= 0.60 and se <= 0.50:
                        ft_label = "Follow-through: Slightly off line"
                        ft_label_col = (0, 165, 255)
                    else:
                        ft_label = "Follow-through: Falling away"
                        ft_label_col = (0, 0, 255)

        if (
            lms is not None
            and evb is not None
            and not loadup_done
            and abs(fi - evb) <= 3
        ):
            nbw = lms[LEFT_WRIST] if bowling_arm == "right" else lms[RIGHT_WRIST]
            nbe = lms[LEFT_ELBOW] if bowling_arm == "right" else lms[RIGHT_ELBOW]
            nose = lms[NOSE]
            if lm_vis(nbw) < 0.6:
                loadup_report = {
                    "at_bfc_frame": evb,
                    "wrist_above_nose_norm": None,
                    "forearm_angle_deg": None,
                    "height_rating": "visibility_low",
                    "alignment_rating": "unknown",
                    "visibility": lm_vis(nbw),
                    "note": "occluded",
                }
                loadup_done = True
            elif lm_vis(nose) > 0.5 and lm_vis(nbe) > 0.5:
                vo = float(nose.y - nbw.y)
                fv = np.array([nbw.x - nbe.x, nbw.y - nbe.y], dtype=np.float64)
                fang = angle_between_deg(fv, np.array([0.0, -1.0], dtype=np.float64))
                if vo > 0.30:
                    hr = "too_high"
                elif vo > 0.12:
                    hr = "good"
                elif vo >= 0.0:
                    hr = "acceptable"
                else:
                    hr = "too_low"
                ar = (
                    "good"
                    if fang < 20
                    else ("acceptable" if fang < 45 else "off_axis")
                )
                loadup_report = {
                    "at_bfc_frame": evb,
                    "wrist_above_nose_norm": vo,
                    "forearm_angle_deg": fang,
                    "height_rating": hr,
                    "alignment_rating": ar,
                    "visibility": lm_vis(nbw),
                    "note": "",
                }
                loadup_done = True
                bfc_overlay_until = fi + int(fps * 1.0)

        spine_peak_running = max(
            (d for _, d in spine_series), default=0.0
        )

        def _add_coach(key: str, txt: str, col: tuple[int, int, int]) -> None:
            if key in coach_seen:
                return
            coach_seen.add(key)
            live_coach_feedback.append((txt, col))

        if spine_peak_running >= 40.0:
            _add_coach(
                "spine_excess",
                "Excessive lateral flexion at release - reduce side-bend.",
                (0, 0, 255),
            )
        elif spine_peak_running >= 25.0:
            _add_coach(
                "spine_moderate",
                "Moderate lateral flexion - work on upright posture.",
                (0, 165, 255),
            )
        if elbow_peak > 1.4:
            _add_coach(
                "arm_excess",
                "Arms flaring too wide - keep elbows closer to body.",
                (0, 0, 255),
            )
        if loadup_report.get("height_rating") == "too_low":
            _add_coach(
                "loadup_low",
                "Front arm not raised high enough at BFC - load-up in front of face.",
                (0, 165, 255),
            )
        if loadup_report.get("height_rating") == "too_high":
            _add_coach(
                "loadup_high",
                "Front arm too high - bring it down to face height.",
                (0, 165, 255),
            )
        if live_final_stride_warn:
            _add_coach(
                "final_stride",
                "Final strides inconsistent - groove your last 3 steps.",
                (0, 165, 255),
            )
        if live_overstride:
            _add_coach(
                "overstride",
                "Over-striding at the crease - shorten your final stride.",
                (0, 165, 255),
            )
        if ft_label == "Follow-through: Falling away":
            _add_coach(
                "ft_poor",
                "Body weight falls away - drive through the line of the pitch.",
                (0, 0, 255),
            )
        elif ft_label == "Follow-through: Slightly off line":
            _add_coach(
                "ft_moderate",
                "Follow-through slightly off line - exit toward pitch line.",
                (0, 165, 255),
            )

        # --- draw ---
        disp = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(disp, (x1, y1), (x2, y2), (140, 140, 140), 2)
        if lms is not None:
            for a, b in mp.solutions.pose.POSE_CONNECTIONS:
                if a < 33 and b < 33:
                    pa = lms[a]
                    pb = lms[b]
                    if lm_vis(pa) < 0.4 or lm_vis(pb) < 0.4:
                        continue
                    cv2.line(
                        disp,
                        (int(pa.x * fw), int(pa.y * fh)),
                        (int(pb.x * fw), int(pb.y * fh)),
                        (255, 0, 255),
                        2,
                    )
            for i in range(33):
                if lm_vis(lms[i]) < 0.4:
                    continue
                cv2.circle(
                    disp,
                    (int(lms[i].x * fw), int(lms[i].y * fh)),
                    4,
                    (255, 255, 0),
                    -1,
                )

        trail_draw = frozen_trail if frozen_trail else list(run_trail)
        if len(trail_draw) >= 2:
            pts = np.array(trail_draw, np.int32)
            n_pts = len(pts)
            for i in range(1, n_pts):
                seg_overlay = disp.copy()
                cv2.line(
                    seg_overlay,
                    tuple(pts[i - 1]),
                    tuple(pts[i]),
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                alpha = (i / max(1, n_pts - 1)) * 0.85 + 0.05
                cv2.addWeighted(
                    seg_overlay, alpha, disp, 1.0 - alpha, 0, disp
                )

        if frozen_arrow is not None:
            cv2.arrowedLine(disp, frozen_arrow[0], frozen_arrow[1], (0, 255, 255), 2)
        elif len(mid_hip_hist) >= 8 and evb is None:
            recent = mid_hip_hist[-8:]
            xs = [p[1] for p in recent]
            ys = [p[2] for p in recent]
            coef = np.polyfit(xs, ys, 1)
            dx, dy = 1.0, float(coef[0])
            n = math.hypot(dx, dy) or 1.0
            mx, my = recent[-1][1], recent[-1][2]
            tip = (int(mx + 40 * dx / n), int(my + 40 * dy / n))
            cv2.arrowedLine(disp, (int(mx), int(my)), tip, (0, 255, 255), 2)
        if evb is not None and frozen_arrow is None and mid_hip_hist:
            p = next((h for h in mid_hip_hist if h[0] == evb), mid_hip_hist[-1])
            recent = [h for h in mid_hip_hist if h[0] <= evb][-8:]
            if len(recent) >= 2:
                xs = [q[1] for q in recent]
                ys = [q[2] for q in recent]
                coef = np.polyfit(xs, ys, 1)
                dx, dy = 1.0, float(coef[0])
                n = math.hypot(dx, dy) or 1.0
                mx, my = p[1], p[2]
                tip = (int(mx + 40 * dx / n), int(my + 40 * dy / n))
                frozen_arrow = ((int(mx), int(my)), tip)

        trace_draw = (
            frozen_ankle_trace if frozen_ankle_trace else ankle_trace
        )
        if trace_draw:
            # Subsample to ~8 dots per second so the run-up cadence (and the
            # jump just before BFC) is visible without flooding the frame.
            sample_stride = max(1, int(round(fps / 8.0)))
            last_drawn_fi: int | None = None
            for tfi, lx_px, ly_px, rx_px, ry_px in trace_draw:
                if (
                    last_drawn_fi is not None
                    and tfi - last_drawn_fi < sample_stride
                ):
                    continue
                left_ok = lx_px >= 0 and ly_px >= 0
                right_ok = rx_px >= 0 and ry_px >= 0
                if left_ok and right_ok:
                    mx_px = (lx_px + rx_px) // 2
                    my_px = (ly_px + ry_px) // 2
                elif left_ok:
                    mx_px, my_px = lx_px, ly_px
                elif right_ok:
                    mx_px, my_px = rx_px, ry_px
                else:
                    continue
                cv2.circle(disp, (mx_px, my_px), 3, (0, 255, 255), -1)
                cv2.circle(disp, (mx_px, my_px), 4, (0, 0, 0), 1)
                last_drawn_fi = tfi

        any_stride_warn = (
            live_final_stride_warn
            or live_overstride
            or live_understride
        )
        for si, st in enumerate(strides):
            px, py = int(st["x"] * fw), int(st["y"] * fh)
            is_last = si == len(strides) - 1
            stride_col = (
                (0, 165, 255)
                if (is_last and any_stride_warn)
                else (0, 255, 255)
            )
            cv2.circle(disp, (px, py), 6, stride_col, -1)
            cv2.circle(disp, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(
                disp,
                str(si + 1),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                stride_col,
                1,
            )

        if lms is not None and ph.events["BFC"] is None:
            le, re = lms[LEFT_ELBOW], lms[RIGHT_ELBOW]
            ls, rs = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
            if min(lm_vis(le), lm_vis(re), lm_vis(ls), lm_vis(rs)) > 0.5:
                er = (
                    math.hypot((re.x - le.x) * fw, (re.y - le.y) * fh)
                    / max(
                        1e-6,
                        math.hypot((rs.x - ls.x) * fw, (rs.y - ls.y) * fh),
                    )
                )
                col = (0, 255, 0) if er < 1.1 else ((0, 165, 255) if er < 1.4 else (0, 0, 255))
                cv2.line(
                    disp,
                    (int(ls.x * fw), int(ls.y * fh)),
                    (int(le.x * fw), int(le.y * fh)),
                    col,
                    2,
                )
                cv2.line(
                    disp,
                    (int(rs.x * fw), int(rs.y * fh)),
                    (int(re.x * fw), int(re.y * fh)),
                    col,
                    2,
                )

        if st_deg is not None and lms is not None:
            ls_lm = lms[LEFT_SHOULDER]
            rs_lm = lms[RIGHT_SHOULDER]
            lh_lm = lms[LEFT_HIP]
            rh_lm = lms[RIGHT_HIP]

            if min(
                lm_vis(ls_lm), lm_vis(rs_lm),
                lm_vis(lh_lm), lm_vis(rh_lm),
            ) > 0.45:
                mh_px = (
                    int((lh_lm.x + rh_lm.x) / 2 * fw),
                    int((lh_lm.y + rh_lm.y) / 2 * fh),
                )
                ms_px = (
                    int((ls_lm.x + rs_lm.x) / 2 * fw),
                    int((ls_lm.y + rs_lm.y) / 2 * fh),
                )

                if st_deg < 25:
                    col = (0, 255, 0)
                    label = "Good alignment"
                elif st_deg < 40:
                    col = (0, 165, 255)
                    label = "Moderate flexion"
                else:
                    col = (0, 0, 255)
                    label = "Excessive flexion - Injury Risk"

                cv2.line(disp, mh_px, ms_px, col, 3, cv2.LINE_AA)

                ref_tip = (mh_px[0], mh_px[1] - 90)
                cv2.line(
                    disp, mh_px, ref_tip,
                    (180, 180, 180), 1, cv2.LINE_AA,
                )

                cv2.ellipse(
                    disp,
                    mh_px,
                    (30, 30),
                    0,
                    -90,
                    -90 + int(st_deg),
                    col,
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    disp,
                    f"Spine: {st_deg:.1f} deg",
                    (mh_px[0] + 10, mh_px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    disp,
                    label,
                    (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    col,
                    2,
                    cv2.LINE_AA,
                )

        if bfc_overlay_until is not None and fi <= bfc_overlay_until and loadup_report:
            label = loadup_report.get("height_rating", "")
            cv2.putText(
                disp,
                f"Load-up: {label}",
                (20, fh - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
            )

        if (
            ph.phase == "FOLLOWTHROUGH"
            and ft_release_anchor_px is not None
            and ft_ideal_dir is not None
        ):
            rx, ry = ft_release_anchor_px
            curr_disp_px = 0.0
            if (
                lms is not None
                and lm_vis(lms[LEFT_ANKLE]) > 0.4
                and lm_vis(lms[RIGHT_ANKLE]) > 0.4
            ):
                cx_now = int(
                    (lms[LEFT_ANKLE].x + lms[RIGHT_ANKLE].x) / 2 * fw
                )
                cy_now = int(
                    (lms[LEFT_ANKLE].y + lms[RIGHT_ANKLE].y) / 2 * fh
                )
                curr_disp_px = math.hypot(cx_now - rx, cy_now - ry)
            green_len = int(max(fw, fh) * 0.14)
            green_thick = 3
            tip_ideal = _ideal_arrow_tip(rx, ry, green_len, flatness=0.34)
            cv2.arrowedLine(
                disp, (rx, ry), tip_ideal,
                (0, 255, 0), green_thick, cv2.LINE_AA, tipLength=0.18,
            )
            cv2.putText(
                disp, "Ideal",
                (tip_ideal[0] + 6, tip_ideal[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

            if ft_actual_dir is not None and curr_disp_px > 6.0:
                red_len = int(min(curr_disp_px, float(green_len)))
                red_thick = max(2, min(5, int(2 + curr_disp_px / 150.0)))
                tip_actual = (
                    rx + int(ft_actual_dir[0] * red_len),
                    ry + int(ft_actual_dir[1] * red_len),
                )
                cv2.arrowedLine(
                    disp, (rx, ry), tip_actual,
                    (0, 0, 255), red_thick, cv2.LINE_AA, tipLength=0.18,
                )
                cv2.putText(
                    disp, "Actual",
                    (tip_actual[0] + 6, tip_actual[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )

            cv2.circle(disp, (rx, ry), 5, (255, 255, 255), -1)
            cv2.circle(disp, (rx, ry), 6, (0, 0, 0), 1)

            if ft_label:
                cv2.putText(
                    disp, ft_label,
                    (12, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    ft_label_col, 2, cv2.LINE_AA,
                )

        # Ball trail / live-ball overlay removed along with ball tracking.

        evi = ph.events.get("IMPULSE")
        landing_anchor: int | None = None
        landing_label: str | None = None
        if evb is not None and evf is not None:
            landing_anchor = int(min(evb, evf))
            landing_label = f"LANDING @ {evb}/{evf}"
        elif evb is not None:
            landing_anchor = int(evb)
            landing_label = f"LANDING (BFC) @ {evb}"
        elif evf is not None:
            landing_anchor = int(evf)
            landing_label = f"LANDING (FFC) @ {evf}"

        landing_end = (
            int(max(evb or 0, evf or 0) + int(0.5 * fps))
            if landing_anchor is not None
            else None
        )

        ev_rows: list[tuple[str, int | None, int | None, int]] = [
            ("IMPULSE", evi, evi + int(0.5 * fps) if evi is not None else None, 0),
            ("LANDING", landing_anchor, landing_end, 1),
            ("RELEASE", evr, evr + int(0.5 * fps) if evr is not None else None, 2),
        ]
        for name, ev_start, ev_stop, row in ev_rows:
            if ev_start is None or ev_stop is None:
                continue
            if not (fi >= ev_start and fi < ev_stop):
                continue
            txt = (
                landing_label
                if name == "LANDING" and landing_label is not None
                else f"{name} @ {ev_start}"
            )
            cv2.putText(
                disp,
                txt,
                (20, 100 + 28 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        if ph.phase == "RUN_UP" and live_arm_label:
            cv2.putText(
                disp,
                live_arm_label,
                (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                live_arm_col,
                2,
                cv2.LINE_AA,
            )
        if ph.phase == "RUN_UP" and live_straight_label:
            cv2.putText(
                disp,
                live_straight_label,
                (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                live_straight_col,
                2,
                cv2.LINE_AA,
            )

        if ph.phase == "FOLLOWTHROUGH" and live_coach_feedback:
            base_y = fh - 20
            for offset, (txt, col) in enumerate(
                reversed(live_coach_feedback[:4])
            ):
                y_pos = base_y - offset * 26
                cv2.rectangle(
                    disp,
                    (8, y_pos - 18),
                    (8 + min(900, 14 + len(txt) * 9), y_pos + 6),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    disp,
                    txt,
                    (14, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    col,
                    1,
                    cv2.LINE_AA,
                )

        phase_display = {
            "RUN_UP": "RUN_UP",
            "JUMP": "JUMP",
            "DELIVERY": "LANDING",
            "RELEASE": "LANDING",
            "FOLLOWTHROUGH": "FOLLOWTHROUGH",
        }.get(ph.phase, ph.phase)
        phase_text = f"Phase: {phase_display}"
        (pt_w, _), _ = cv2.getTextSize(
            phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_pill(
            disp,
            [(phase_text, (200, 255, 200))],
            (max(0, fw - pt_w - 24), 12),
        )

        writer.write(disp)
        fi += 1

    ph.events_audit()
    pose_est.close()
    cap.release()
    writer.release()

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Post metrics
    evb, evf, evr = ph.events["BFC"], ph.events["FFC"], ph.events["RELEASE"]
    evi = ph.events.get("IMPULSE")

    def spine_at(f: int | None) -> float | None:
        if f is None:
            return None
        best = None
        for ff, deg in spine_series:
            if abs(ff - f) <= 2:
                if best is None or abs(ff - f) < abs(best[0] - f):
                    best = (ff, deg)
        return best[1] if best else None

    peak_spine = max((d for _, d in spine_series), default=None)
    peak_fr = next((ff for ff, d in spine_series if d == peak_spine), None)

    # run-up straightness
    straightness = 0.9
    if evb is not None:
        pre = [(mx, my) for f, mx, my in mid_hip_hist if f < evb]
        if len(pre) >= 5:
            xs = [p[0] for p in pre]
            ys = [p[1] for p in pre]
            coef = np.polyfit(xs, ys, 1)
            devs = []
            for x, y in zip(xs, ys):
                pred = coef[0] * x + coef[1]
                devs.append(abs(y - pred) / fh)
            mean_dev = float(np.mean(devs)) if devs else 0.0
            straightness = 1.0 - min(1.0, mean_dev / 0.05)

    # stride metrics
    ref_stride = 1.0
    stride_lens: list[float] = []
    for i in range(1, len(strides)):
        p0, p1 = strides[i - 1], strides[i]
        d = math.hypot((p1["x"] - p0["x"]) * fw, (p1["y"] - p0["y"]) * fh)
        stride_lens.append(d / shoulder_w_px)
    if len(stride_lens) >= 3:
        usable = stride_lens[1: min(len(stride_lens), 5)]
        ref_stride = float(np.median(usable)) if usable else 1.0
    ratios = [s / ref_stride for s in stride_lens] if ref_stride > 1e-6 else []
    std_r = float(np.std(ratios)) if len(ratios) > 1 else 0.0
    if std_r < 0.12:
        cons = "Consistent"
    elif std_r < 0.25:
        cons = "Variable"
    else:
        cons = "Erratic"
    final_warn = any(
        abs(ratios[i] - 1.0) > 0.25 for i in range(max(0, len(ratios) - 3), len(ratios))
    ) if len(ratios) >= 3 else False
    over = under = False
    if stride_lens:
        last = stride_lens[-1]
        over = last > ref_stride * 1.30
        under = last < ref_stride * 0.70

    if elbow_peak < 1.1:
        arm_rating = "good"
    elif elbow_peak < 1.4:
        arm_rating = "moderate"
    else:
        arm_rating = "excessive"
    avg_wf = wrist_flare_sum / max(1, wrist_flare_n)

    st_rel = spine_at(evr) or 0.0
    spine_rating = (
        "good"
        if st_rel < 25
        else ("moderate" if st_rel < 40 else "excessive")
    )

    # follow-through
    ft: dict[str, Any] = {}
    if evr is not None and evf is not None:
        pre = [(f, mx, my) for f, mx, my in mid_hip_hist if evf - 10 <= f <= evf]
        anchor_ft = next((h for h in mid_hip_hist if h[0] == evr), None)
        if anchor_ft is not None:
            idir2 = _ideal_pitch_forward_dir(
                int(anchor_ft[1]), int(anchor_ft[2]), fw, fh
            )
        elif len(pre) >= 2:
            s = np.array([pre[0][1], pre[0][2]], dtype=np.float64)
            e = np.array([pre[-1][1], pre[-1][2]], dtype=np.float64)
            idir2 = _ideal_pitch_forward_dir(
                int(pre[0][1]), int(pre[0][2]), fw, fh
            )
        else:
            idir2 = _ideal_pitch_forward_dir(fw // 2, fh // 2, fw, fh)
        endf = min(fi - 1, evr + int(fps * 0.5))
        m0 = next((h for h in mid_hip_hist if h[0] == evr), None)
        m1 = next((h for h in mid_hip_hist if h[0] == endf), None)
        if m0 and m1:
            ad = norm_vec(m1[1] - m0[1], m1[2] - m0[2])
            al = float(ad[0] * idir2[0] + ad[1] * idir2[1])
            se = abs(cross_z(ad, idir2))
            dev_ang = math.degrees(math.acos(clamp(al, -1.0, 1.0)))
            if al >= 0.85 and se <= 0.25:
                frt = "good"
            elif al >= 0.60 and se <= 0.50:
                frt = "moderate"
            else:
                frt = "poor"
            ft = {
                "alignment_score": al,
                "sideways_error": se,
                "deviation_angle_deg": dev_ang,
                "rating": frt,
                "ideal_direction": [idir2[0], idir2[1]],
                "actual_direction": [ad[0], ad[1]],
            }
        else:
            ft = {
                "alignment_score": 0.0,
                "sideways_error": 1.0,
                "deviation_angle_deg": 90.0,
                "rating": "poor",
                "ideal_direction": [idir2[0], idir2[1]],
                "actual_direction": [0.0, 1.0],
            }

    # ----- Back-view-reliable transverse/lateral metrics -----
    def series_at(series: list[tuple[int, float]], f: int | None, tol: int = 2) -> float | None:
        if f is None:
            return None
        best: tuple[int, float] | None = None
        for ff, val in series:
            if abs(ff - f) <= tol and (best is None or abs(ff - f) < abs(best[0] - f)):
                best = (ff, val)
        return best[1] if best else None

    # Shoulder counter-rotation / mixed-action proxy (BFC -> FFC).
    sh_bfc = series_at(shoulder_line_series, evb)
    sh_ffc = series_at(shoulder_line_series, evf)
    shoulder_rotation_bfc_ffc = (
        float(sh_ffc - sh_bfc) if (sh_bfc is not None and sh_ffc is not None) else None
    )
    counter_rotation_deg: float | None = None
    if sh_bfc is not None and sh_ffc is not None and evb is not None and evf is not None:
        seg = [v for f, v in shoulder_line_series if evb <= f <= evf]
        if seg:
            net = sh_ffc - sh_bfc
            direction = 1.0 if net >= 0 else -1.0
            excursions = [-(v - sh_bfc) * direction for v in seg]
            counter_rotation_deg = max(0.0, float(max(excursions)))
    mixed_action = bool(counter_rotation_deg is not None and counter_rotation_deg > 30.0)
    if counter_rotation_deg is None:
        shoulder_rating = "unknown"
    elif counter_rotation_deg <= 20.0:
        shoulder_rating = "good"
    elif counter_rotation_deg <= 30.0:
        shoulder_rating = "moderate"
    else:
        shoulder_rating = "excessive"

    # Hip-shoulder separation proxy (informational; pace link inconsistent in literature).
    sep_bfc = series_at(sep_series, evb)
    sep_ffc = series_at(sep_series, evf)
    sep_peak: float | None = None
    if evb is not None and evf is not None:
        lo, hi = min(evb, evf), max(evb, evf)
        seg_sep = [v for f, v in sep_series if lo <= f <= hi]
        if seg_sep:
            sep_peak = float(max(seg_sep))

    # Falling-away: lateral drift of head and mid-hip from release through early follow-through.
    falling_away_score: float | None = None
    head_lateral_drift_sw: float | None = None
    midhip_lateral_drift_sw: float | None = None
    falling_rating = "unknown"
    if evr is not None and shoulder_w_px > 1e-6:
        endf_fa = min(fi - 1, evr + int(fps * 0.3))
        mx0 = series_at(midhip_x_series, evr)
        mx1 = series_at(midhip_x_series, endf_fa)
        hx0 = series_at(head_x_series, evr)
        hx1 = series_at(head_x_series, endf_fa)
        if mx0 is not None and mx1 is not None:
            midhip_lateral_drift_sw = float((mx1 - mx0) / shoulder_w_px)
        if hx0 is not None and hx1 is not None:
            head_lateral_drift_sw = float((hx1 - hx0) / shoulder_w_px)
        candidates = [abs(v) for v in (head_lateral_drift_sw, midhip_lateral_drift_sw) if v is not None]
        if candidates:
            falling_away_score = float(max(candidates))
            if falling_away_score <= 0.25:
                falling_rating = "good"
            elif falling_away_score <= 0.6:
                falling_rating = "moderate"
            else:
                falling_rating = "excessive"

    coach: list[str] = []
    if mixed_action:
        coach.append(
            "Mixed action detected — shoulders counter-rotate after back-foot contact, raising lower-back stress. Keep your alignment consistent through the crease."
        )
    if falling_rating == "excessive":
        coach.append(
            "Head and body fall away to the side at release — stay tall and drive your chest down the pitch."
        )
    if (spine_at(evr) or 0) >= 40:
        coach.append(
            "Excessive lateral flexion at release — reduce side-bend to protect your lower back."
        )
    elif 25 <= (spine_at(evr) or 0) < 40:
        coach.append(
            "Moderate lateral flexion — work on upright posture through the crease."
        )
    if elbow_peak > 1.4:
        coach.append(
            "Arms flaring too wide in your run-up — keep elbows closer to your body for better balance."
        )
    if loadup_report.get("height_rating") == "too_low":
        coach.append(
            "Front arm not raised high enough at back foot contact — load-up in front of your face."
        )
    if loadup_report.get("height_rating") == "too_high":
        coach.append(
            "Front arm too high at load-up — bring it down to face height for better counter-rotation."
        )
    if final_warn:
        coach.append(
            "Final strides are inconsistent — use stride markers in training to groove your last 3 steps."
        )
    if over:
        coach.append(
            "You are over-striding at the crease — shorten your final stride to improve balance."
        )
    if ft.get("rating") == "poor":
        coach.append(
            "Body weight falls away after release — drive through the line of the pitch."
        )
    elif ft.get("rating") == "moderate":
        coach.append(
            "Follow-through slightly off line — aim to exit on the left of the pitch."
        )

    report: dict[str, Any] = {
        "video_path": video_path,
        "fps": fps,
        "bowling_arm": bowling_arm,
        "events": {
            "IMPULSE": {"frame": evi, "time_s": evi / fps if evi else None},
            "LANDING": {
                "frame": (
                    min(evb, evf) if (evb is not None and evf is not None)
                    else (evb if evb is not None else evf)
                ),
                "time_s": (
                    (min(evb, evf) / fps)
                    if (evb is not None and evf is not None)
                    else (
                        (evb / fps) if evb is not None
                        else (evf / fps) if evf is not None
                        else None
                    )
                ),
                "bfc_frame": evb,
                "ffc_frame": evf,
            },
            "BFC": {"frame": evb, "time_s": evb / fps if evb else None},
            "FFC": {"frame": evf, "time_s": evf / fps if evf else None},
            "RELEASE": {"frame": evr, "time_s": evr / fps if evr else None},
        },
        "run_up": {
            "strides": strides,
            "reference_stride_norm": ref_stride,
            "stride_ratios": ratios,
            "consistency_label": cons,
            "final_stride_warning": final_warn,
            "overstride": over,
            "understride": under,
            "straightness_score": straightness,
        },
        "arm_alignment": {
            "peak_elbow_flare_ratio": elbow_peak,
            "avg_wrist_flare_norm": avg_wf,
            "rating": arm_rating,
            "worst_frame": worst_elbow_f,
        },
        "spine_tilt": {
            "at_bfc_deg": spine_at(evb),
            "at_ffc_deg": spine_at(evf),
            "at_release_deg": spine_at(evr),
            "peak_deg": peak_spine,
            "peak_frame": peak_fr,
            "rating": spine_rating,
        },
        "shoulder_alignment": {
            "shoulder_angle_bfc_deg": sh_bfc,
            "shoulder_angle_ffc_deg": sh_ffc,
            "shoulder_rotation_bfc_to_ffc_deg": shoulder_rotation_bfc_ffc,
            "counter_rotation_deg": counter_rotation_deg,
            "mixed_action": mixed_action,
            "rating": shoulder_rating,
        },
        "hip_shoulder_separation": {
            "at_bfc_deg": sep_bfc,
            "at_ffc_deg": sep_ffc,
            "peak_bfc_to_ffc_deg": sep_peak,
            "note": "2D back-view transverse proxy; literature link to pace is inconsistent (informational).",
        },
        "falling_away": {
            "head_lateral_drift_sw": head_lateral_drift_sw,
            "midhip_lateral_drift_sw": midhip_lateral_drift_sw,
            "score_sw": falling_away_score,
            "rating": falling_rating,
        },
        "loadup": loadup_report,
        "ball_tracking": {
            "status": "disabled",
            "frames_tracked": 0,
            "release_frame": evr,
            "speed_kmh": None,
        },
        "follow_through": {
            "alignment_score": ft.get("alignment_score"),
            "sideways_error": ft.get("sideways_error"),
            "deviation_angle_deg": ft.get("deviation_angle_deg"),
            "rating": ft.get("rating", "unknown"),
            "ideal_direction": ft.get("ideal_direction"),
            "actual_direction": ft.get("actual_direction"),
        },
        "coach_feedback": coach,
        "drills": [],
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# =========================================================================== #
# FEEDBACK ENGINE (merged from bowling_feedback.py)
#
# Consumes the report dict produced by analyse_video() above and turns raw
# biomechanical metrics into structured, research-grounded coaching feedback.
#
# Pipeline:
#     report dict  ->  extract + score each metric against references
#                  ->  gate by confidence (event availability + data presence)
#                  ->  rank into strengths / improvements / safety flags
#                  ->  deterministic coach text + drills (works with no API)
#                  ->  LLM-ready payload + guarded system prompt (API optional)
#
# Scope: BACK VIEW only. Sagittal-plane pace predictors (front-knee extension,
# forward trunk flexion, run-up speed, release speed) are intentionally marked
# "needs_side_on" and never fabricated.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Reference benchmarks (back-view metrics only).
#
# Ranges are grounded in the reviewed literature and the thresholds already
# used inside the analyser above:
#   - Lateral trunk flexion: injury-first; large side-bend at release loads the
#     lumbar spine (IJISRT review; many 3D studies).
#   - Shoulder counter-rotation > ~30 deg = "mixed action" = elevated lower-back
#     stress / stress-fracture risk (Portus/Elliott, cited in IJISRT review).
#   - Falling-away: lateral head/COM drift at release reduces momentum down the
#     pitch and degrades accuracy.
#   - Run-up straightness / stride consistency: rhythm and repeatability.
#   - Arm flare: balance proxy in the approach.
#   - Hip-shoulder separation: informational (pace link inconsistent across
#     studies per the 2024 systematic review).
# Levels tighten the "good" boundary for more advanced bowlers.
# --------------------------------------------------------------------------- #

LEVELS = ("junior", "club", "elite")


@dataclass
class Band:
    """A lower-is-better or higher-is-better banded reference."""
    good_max: float | None = None     # for lower-is-better: value <= good_max -> good
    moderate_max: float | None = None  # value <= moderate_max -> moderate, else poor
    good_min: float | None = None     # for higher-is-better: value >= good_min -> good
    moderate_min: float | None = None  # value >= moderate_min -> moderate, else poor
    higher_is_better: bool = False
    unit: str = ""


# Per-level band overrides keyed by metric.
REFERENCES: dict[str, dict[str, Band]] = {
    "lateral_flexion_release_deg": {
        "junior": Band(good_max=28, moderate_max=42, unit="deg"),
        "club": Band(good_max=25, moderate_max=40, unit="deg"),
        "elite": Band(good_max=22, moderate_max=35, unit="deg"),
    },
    "counter_rotation_deg": {
        "junior": Band(good_max=22, moderate_max=32, unit="deg"),
        "club": Band(good_max=20, moderate_max=30, unit="deg"),
        "elite": Band(good_max=15, moderate_max=25, unit="deg"),
    },
    "falling_away_score_sw": {
        "junior": Band(good_max=0.30, moderate_max=0.65, unit="shoulder-widths"),
        "club": Band(good_max=0.25, moderate_max=0.60, unit="shoulder-widths"),
        "elite": Band(good_max=0.20, moderate_max=0.50, unit="shoulder-widths"),
    },
    "runup_straightness": {
        "junior": Band(good_min=0.80, moderate_min=0.60, higher_is_better=True, unit="score"),
        "club": Band(good_min=0.85, moderate_min=0.65, higher_is_better=True, unit="score"),
        "elite": Band(good_min=0.90, moderate_min=0.72, higher_is_better=True, unit="score"),
    },
    "elbow_flare_ratio": {
        "junior": Band(good_max=1.2, moderate_max=1.5, unit="ratio"),
        "club": Band(good_max=1.1, moderate_max=1.4, unit="ratio"),
        "elite": Band(good_max=1.05, moderate_max=1.3, unit="ratio"),
    },
    "follow_through_deviation_deg": {
        "junior": Band(good_max=25, moderate_max=50, unit="deg"),
        "club": Band(good_max=20, moderate_max=45, unit="deg"),
        "elite": Band(good_max=15, moderate_max=35, unit="deg"),
    },
}

# Drills mapped per metric + rating bucket.
DRILLS: dict[str, dict[str, list[str]]] = {
    "lateral_flexion_release_deg": {
        "moderate": ["Tall-spine shadow bowling", "Side-plank holds"],
        "poor": ["Tall-spine shadow bowling", "Side-plank holds", "Alignment-stick posture drill"],
    },
    "counter_rotation_deg": {
        "moderate": ["Back-foot landing alignment drill", "Wall-line shoulder tracking"],
        "poor": ["Front-on/side-on consistency drill", "Resisted hip-shoulder timing", "Video-mirror alignment work"],
    },
    "falling_away_score_sw": {
        "moderate": ["Chest-over-front-knee drill", "Target-line follow-through walk-throughs"],
        "poor": ["Brace-and-drive bound drill", "Bowl-at-single-stump line drill", "Front-leg bracing wall drill"],
    },
    "runup_straightness": {
        "moderate": ["Line-on-the-ground run-up drill", "Cone-gate approach"],
        "poor": ["Tramline run-up drill", "Metronome rhythm runs", "Mark-and-repeat approach"],
    },
    "stride_consistency": {
        "moderate": ["Stride-marker grooving (last 3 steps)", "Metronome run-up practice"],
        "poor": ["Tape-marker stride drill", "Bound-to-mark repeats", "Rhythm-count approach"],
    },
    "elbow_flare_ratio": {
        "moderate": ["Elbows-in arm-drive drill", "Tucked-arm shadow runs"],
        "poor": ["Towel-under-arm run drill", "Relaxed-hands sprint mechanics"],
    },
    "follow_through_deviation_deg": {
        "moderate": ["Target-line follow-through drill", "Step-through bowling drill"],
        "poor": ["Straight-line follow-through walk-throughs", "Drive-through-the-gate drill"],
    },
}

# Metrics that the back view simply cannot measure honestly.
NEEDS_SIDE_ON = [
    {"key": "front_knee_extension_br", "label": "Front-knee extension at release",
     "why": "Strongest technique predictor of pace; sagittal-plane angle invisible from behind."},
    {"key": "trunk_forward_flexion_ffc_br", "label": "Upper-trunk forward flexion (FFC->BR)",
     "why": "Pace contributor; forward (sagittal) flexion cannot be separated from side-bend in back view."},
    {"key": "shoulder_flexion_ffc", "label": "Bowling-arm shoulder flexion at FFC",
     "why": "Arm-elevation range correlates with pace; needs side-on."},
    {"key": "run_up_speed_ms", "label": "Run-up speed",
     "why": "Strong pace predictor; horizontal speed is foreshortened from behind."},
    {"key": "release_speed_kmh", "label": "Ball release speed",
     "why": "Ball travels away from a back-view camera, so pixel speed is unreliable without calibration."},
]


# --------------------------------------------------------------------------- #
# Finding model
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    key: str
    label: str
    category: str            # "safety" | "efficiency" | "rhythm" | "info"
    value: float | str | None
    unit: str
    rating: str              # "good" | "moderate" | "poor" | "unknown"
    confidence: str          # "high" | "medium" | "low"
    direction: str           # human-readable improvement direction
    reference: dict[str, Any]
    view_reliability: str    # "back_view" | "needs_side_on"
    message: str
    drills: list[str] = field(default_factory=list)


def _rate_band(value: float, band: Band) -> str:
    if band.higher_is_better:
        if band.good_min is not None and value >= band.good_min:
            return "good"
        if band.moderate_min is not None and value >= band.moderate_min:
            return "moderate"
        return "poor"
    if band.good_max is not None and value <= band.good_max:
        return "good"
    if band.moderate_max is not None and value <= band.moderate_max:
        return "poor" if value > band.moderate_max else "moderate"
    return "poor"


def _band_to_dict(band: Band) -> dict[str, Any]:
    d = {k: v for k, v in asdict(band).items() if v is not None and v != ""}
    return d


# --------------------------------------------------------------------------- #
# Extraction + scoring
# --------------------------------------------------------------------------- #

def _events_ok(report: dict[str, Any], *names: str) -> bool:
    ev = report.get("events", {}) or {}
    for n in names:
        node = ev.get(n) or {}
        if node.get("frame") is None:
            return False
    return True


def _events_plausible(report: dict[str, Any]) -> bool:
    """
    Guard against degenerate upstream event detection (e.g. BFC/FFC/RELEASE all
    collapsing near frame 0). Checks inter-event timing against plausible ranges.
    """
    ev = report.get("events", {}) or {}
    fps = float(report.get("fps") or 30.0)
    bfc = (ev.get("BFC") or {}).get("frame")
    ffc = (ev.get("FFC") or {}).get("frame")
    rel = (ev.get("RELEASE") or {}).get("frame")
    if bfc is None or ffc is None or rel is None:
        return False
    bfc_ffc_s = (ffc - bfc) / fps
    ffc_rel_s = (rel - ffc) / fps
    # Delivery-stride and FFC->release windows for a real action.
    if not (0.04 <= bfc_ffc_s <= 0.45):
        return False
    if not (0.02 <= ffc_rel_s <= 0.40):
        return False
    return True


def _confidence(report: dict[str, Any], required_events: tuple[str, ...], value: Any) -> str:
    if value is None:
        return "low"
    if not _events_ok(report, *required_events):
        return "low"
    # Event-timed metrics are only trustworthy if the events are plausibly spaced.
    if any(e in required_events for e in ("BFC", "FFC", "RELEASE")):
        if not _events_plausible(report):
            return "low"
    # Ball tracking quality, if present, nudges confidence for release-timed metrics.
    if "RELEASE" in required_events:
        bt = report.get("ball_tracking", {}) or {}
        status = str(bt.get("status", "")).lower()
        if status in {"lost", "none", "failed", ""} and bt.get("frames_tracked", 0) in (0, None):
            return "medium"
    return "high"


def score_findings(report: dict[str, Any], level: str) -> list[Finding]:
    level = level if level in LEVELS else "club"
    findings: list[Finding] = []

    spine = report.get("spine_tilt", {}) or {}
    shoulder = report.get("shoulder_alignment", {}) or {}
    falling = report.get("falling_away", {}) or {}
    run_up = report.get("run_up", {}) or {}
    arm = report.get("arm_alignment", {}) or {}
    ft = report.get("follow_through", {}) or {}
    sep = report.get("hip_shoulder_separation", {}) or {}

    # 1. Lateral trunk flexion at release (safety + efficiency)
    val = spine.get("at_release_deg")
    band = REFERENCES["lateral_flexion_release_deg"][level]
    rating = _rate_band(float(val), band) if val is not None else "unknown"
    conf = _confidence(report, ("RELEASE",), val)
    findings.append(Finding(
        key="lateral_flexion_release_deg",
        label="Lateral trunk flexion at release",
        category="safety",
        value=round(float(val), 1) if val is not None else None,
        unit="deg",
        rating=rating,
        confidence=conf,
        direction="lower is safer (reduce side-bend at release)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_lateral_flexion(val, rating),
        drills=DRILLS["lateral_flexion_release_deg"].get(rating, []),
    ))

    # 2. Shoulder counter-rotation / mixed action (safety)
    val = shoulder.get("counter_rotation_deg")
    band = REFERENCES["counter_rotation_deg"][level]
    rating = _rate_band(float(val), band) if val is not None else "unknown"
    conf = _confidence(report, ("BFC", "FFC"), val)
    findings.append(Finding(
        key="counter_rotation_deg",
        label="Shoulder counter-rotation (mixed-action check)",
        category="safety",
        value=round(float(val), 1) if val is not None else None,
        unit="deg",
        rating=rating,
        confidence=conf,
        direction="lower is safer (keep shoulders aligned BFC->FFC)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_counter_rotation(val, rating, bool(shoulder.get("mixed_action"))),
        drills=DRILLS["counter_rotation_deg"].get(rating, []),
    ))

    # 3. Falling away at release (efficiency + accuracy)
    val = falling.get("score_sw")
    band = REFERENCES["falling_away_score_sw"][level]
    rating = _rate_band(float(val), band) if val is not None else "unknown"
    conf = _confidence(report, ("RELEASE",), val)
    findings.append(Finding(
        key="falling_away_score_sw",
        label="Falling away at release (lateral drift)",
        category="efficiency",
        value=round(float(val), 2) if val is not None else None,
        unit="shoulder-widths",
        rating=rating,
        confidence=conf,
        direction="lower is better (drive tall down the pitch)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_falling(val, rating),
        drills=DRILLS["falling_away_score_sw"].get(rating, []),
    ))

    # 4. Run-up straightness (rhythm)
    val = run_up.get("straightness_score")
    band = REFERENCES["runup_straightness"][level]
    rating = _rate_band(float(val), band) if val is not None else "unknown"
    conf = "high" if val is not None else "low"
    findings.append(Finding(
        key="runup_straightness",
        label="Run-up straightness",
        category="rhythm",
        value=round(float(val), 2) if val is not None else None,
        unit="score",
        rating=rating,
        confidence=conf,
        direction="higher is better (run straight to the crease)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_straightness(val, rating),
        drills=DRILLS["runup_straightness"].get(rating, []),
    ))

    # 5. Stride consistency (rhythm) - categorical
    cons = run_up.get("consistency_label")
    rating = {"Consistent": "good", "Variable": "moderate", "Erratic": "poor"}.get(cons, "unknown")
    extra = []
    if run_up.get("final_stride_warning"):
        extra.append("final strides irregular")
    if run_up.get("overstride"):
        extra.append("over-striding at the crease")
    if run_up.get("understride"):
        extra.append("under-striding at the crease")
    findings.append(Finding(
        key="stride_consistency",
        label="Stride consistency",
        category="rhythm",
        value=cons,
        unit="",
        rating=rating,
        confidence="high" if cons else "low",
        direction="more repeatable strides into the crease",
        reference={"good": "Consistent", "moderate": "Variable", "poor": "Erratic"},
        view_reliability="back_view",
        message=_msg_stride(cons, rating, extra),
        drills=DRILLS["stride_consistency"].get(rating, []),
    ))

    # 6. Arm flare in approach (rhythm/balance)
    val = arm.get("peak_elbow_flare_ratio")
    band = REFERENCES["elbow_flare_ratio"][level]
    rating = _rate_band(float(val), band) if val is not None else (arm.get("rating") or "unknown")
    findings.append(Finding(
        key="elbow_flare_ratio",
        label="Arm flare in run-up",
        category="rhythm",
        value=round(float(val), 2) if val is not None else None,
        unit="ratio",
        rating=rating if rating != "excessive" else "poor",
        confidence="high" if val is not None else "low",
        direction="lower is better (keep elbows closer to the body)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_arm(val, rating),
        drills=DRILLS["elbow_flare_ratio"].get("poor" if rating in ("poor", "excessive") else rating, []),
    ))

    # 7. Follow-through direction (efficiency)
    val = ft.get("deviation_angle_deg")
    band = REFERENCES["follow_through_deviation_deg"][level]
    rating = _rate_band(float(val), band) if val is not None else (ft.get("rating") or "unknown")
    conf = _confidence(report, ("RELEASE", "FFC"), val)
    findings.append(Finding(
        key="follow_through_deviation_deg",
        label="Follow-through direction",
        category="efficiency",
        value=round(float(val), 1) if val is not None else None,
        unit="deg",
        rating=rating,
        confidence=conf,
        direction="lower is better (exit straight down the pitch)",
        reference=_band_to_dict(band),
        view_reliability="back_view",
        message=_msg_follow_through(val, rating),
        drills=DRILLS["follow_through_deviation_deg"].get(rating, []),
    ))

    # 8. Hip-shoulder separation - informational only
    val = sep.get("peak_bfc_to_ffc_deg")
    findings.append(Finding(
        key="hip_shoulder_separation_peak_deg",
        label="Hip-shoulder separation (peak BFC->FFC)",
        category="info",
        value=round(float(val), 1) if val is not None else None,
        unit="deg",
        rating="info",
        confidence=_confidence(report, ("BFC", "FFC"), val),
        direction="informational (2D back-view proxy; literature link to pace inconsistent)",
        reference={"note": sep.get("note", "informational")},
        view_reliability="back_view",
        message=_msg_separation(val),
        drills=[],
    ))

    return findings


# --------------------------------------------------------------------------- #
# Message templates (deterministic, number-aware)
# --------------------------------------------------------------------------- #

def _msg_lateral_flexion(val: Any, rating: str) -> str:
    if val is None:
        return "Lateral trunk flexion at release could not be measured (release frame uncertain)."
    if rating == "good":
        return f"Side-bend at release is controlled ({val:.0f} deg) - good for spine safety."
    if rating == "moderate":
        return f"Moderate side-bend at release ({val:.0f} deg) - work toward a taller release."
    return f"Excessive side-bend at release ({val:.0f} deg) - high lower-back load; prioritise reducing it."


def _msg_counter_rotation(val: Any, rating: str, mixed: bool) -> str:
    if val is None:
        return "Shoulder counter-rotation could not be measured (BFC/FFC uncertain)."
    if mixed or rating == "poor":
        return (f"Shoulders counter-rotate ~{val:.0f} deg after back-foot contact - a mixed-action "
                "pattern linked to lower-back stress. Keep alignment consistent through the crease.")
    if rating == "moderate":
        return f"Some shoulder counter-rotation (~{val:.0f} deg) - monitor it to stay clear of a mixed action."
    return f"Shoulder alignment BFC->FFC is consistent (~{val:.0f} deg counter-rotation) - safe pattern."


def _msg_falling(val: Any, rating: str) -> str:
    if val is None:
        return "Falling-away could not be measured (release frame uncertain)."
    if rating == "good":
        return f"You stay upright through release (lateral drift {val:.2f} shoulder-widths) - efficient."
    if rating == "moderate":
        return f"Some falling away at release ({val:.2f} shoulder-widths) - drive a bit taller down the pitch."
    return f"Strong falling away at release ({val:.2f} shoulder-widths) - costs momentum and accuracy."


def _msg_straightness(val: Any, rating: str) -> str:
    if val is None:
        return "Run-up straightness could not be measured."
    if rating == "good":
        return f"Run-up is straight to the crease (score {val:.2f})."
    if rating == "moderate":
        return f"Slight run-up drift (score {val:.2f}) - tighten your line to the crease."
    return f"Run-up drifts noticeably (score {val:.2f}) - run a straighter line for repeatability."


def _msg_stride(cons: Any, rating: str, extra: list[str]) -> str:
    if not cons:
        return "Stride pattern could not be measured."
    base = {"good": "Stride rhythm is consistent.",
            "moderate": "Stride rhythm is a bit variable.",
            "poor": "Stride rhythm is erratic."}.get(rating, "Stride rhythm measured.")
    if extra:
        base += " Also: " + ", ".join(extra) + "."
    return base


def _msg_arm(val: Any, rating: str) -> str:
    if val is None:
        return "Arm flare in the run-up could not be measured."
    if rating == "good":
        return f"Arms stay close to the body in the approach (flare {val:.2f})."
    if rating == "moderate":
        return f"Arms flare a little in the approach (flare {val:.2f}) - tidy them up for balance."
    return f"Arms flare wide in the approach (flare {val:.2f}) - keep elbows in for better balance."


def _msg_follow_through(val: Any, rating: str) -> str:
    if val is None:
        return "Follow-through direction could not be measured."
    if rating == "good":
        return f"Follow-through drives straight down the pitch (deviation {val:.0f} deg)."
    if rating == "moderate":
        return f"Follow-through slightly off line (deviation {val:.0f} deg)."
    return f"Follow-through goes across the line (deviation {val:.0f} deg) - drive through the target."


def _msg_separation(val: Any) -> str:
    if val is None:
        return "Hip-shoulder separation could not be measured."
    return (f"Peak hip-shoulder separation proxy ~{val:.0f} deg (informational; back-view 2D estimate). "
            "Larger, well-timed separation can help pace, but evidence is mixed.")


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def rank_findings(findings: list[Finding]) -> dict[str, Any]:
    usable = [f for f in findings if f.confidence != "low" and f.rating not in ("unknown",)]
    severity = {"poor": 2, "moderate": 1, "good": 0, "info": 0}

    safety_flags = [f for f in usable if f.category == "safety" and f.rating in ("moderate", "poor")]
    improvements = [
        f for f in usable
        if f.rating in ("moderate", "poor") and f.category != "info"
    ]
    improvements.sort(
        key=lambda f: (severity.get(f.rating, 0), 1 if f.category == "safety" else 0,
                       1 if f.confidence == "high" else 0),
        reverse=True,
    )
    strengths = [f for f in usable if f.rating == "good"]

    top_focus = improvements[:3]
    return {
        "top_focus": [f.key for f in top_focus],
        "safety_flags": [f.key for f in safety_flags],
        "improvements": [f.key for f in improvements],
        "strengths": [f.key for f in strengths],
    }


# --------------------------------------------------------------------------- #
# Deterministic coach summary (works with no LLM)
# --------------------------------------------------------------------------- #

def deterministic_summary(findings: list[Finding], ranking: dict[str, Any]) -> dict[str, Any]:
    by_key = {f.key: f for f in findings}
    lines: list[str] = []
    drills: list[str] = []

    if ranking["safety_flags"]:
        lines.append("Safety first:")
        for k in ranking["safety_flags"]:
            lines.append(f"  - {by_key[k].message}")

    if ranking["top_focus"]:
        lines.append("Top things to work on to bowl faster and safer:")
        for i, k in enumerate(ranking["top_focus"], 1):
            f = by_key[k]
            lines.append(f"  {i}. {f.message}")
            drills.extend(f.drills)

    if ranking["strengths"]:
        lines.append("What is already working:")
        for k in ranking["strengths"][:3]:
            lines.append(f"  - {by_key[k].message}")

    if not ranking["top_focus"] and not ranking["safety_flags"]:
        lines.append("No major back-view red flags detected. Keep grooving your action.")

    # de-duplicate drills, preserve order
    seen: set[str] = set()
    drills_unique = [d for d in drills if not (d in seen or seen.add(d))][:6]

    return {"coach_text": "\n".join(lines), "drills": drills_unique}


def _finding_card(f: Finding) -> dict[str, Any]:
    """Compact, UI-ready slice of one finding."""
    return {
        "key": f.key,
        "label": f.label,
        "category": f.category,
        "value": f.value,
        "unit": f.unit,
        "rating": f.rating,
        "confidence": f.confidence,
        "message": f.message,
        "direction": f.direction,
        "drills": f.drills,
    }


def _cards_for_keys(by_key: dict[str, Finding], keys: list[str]) -> list[dict[str, Any]]:
    return [_finding_card(by_key[k]) for k in keys if k in by_key]


def _build_headline(ranking: dict[str, Any], by_key: dict[str, Finding]) -> str:
    if ranking.get("safety_flags"):
        k = ranking["safety_flags"][0]
        return f"Address {by_key[k].label.lower()} first — injury-risk pattern detected."
    if ranking.get("top_focus"):
        k = ranking["top_focus"][0]
        return f"Main focus: {by_key[k].label.lower()}."
    if ranking.get("strengths"):
        return "Solid back-view mechanics — keep grooving this action."
    return "Analysis complete — review the detailed findings below."


def _build_summary_paragraph(
    ranking: dict[str, Any],
    by_key: dict[str, Finding],
    drills: list[str],
) -> str:
    parts: list[str] = []
    if ranking.get("safety_flags"):
        msgs = [by_key[k].message for k in ranking["safety_flags"] if k in by_key]
        parts.append("Safety: " + " ".join(msgs))
    if ranking.get("top_focus"):
        focus_msgs = [
            by_key[k].message for k in ranking["top_focus"][:3] if k in by_key
        ]
        parts.append("Priorities: " + " ".join(focus_msgs))
    elif ranking.get("strengths"):
        parts.append(
            "No major back-view issues flagged. "
            + by_key[ranking["strengths"][0]].message
        )
    if drills:
        parts.append(f"Suggested drills: {', '.join(drills[:4])}.")
    return " ".join(parts)


def build_coach_report(
    findings: list[Finding],
    ranking: dict[str, Any],
    deterministic: dict[str, Any],
    llm_text: str | None,
) -> dict[str, Any]:
    """User-facing coaching block — everything the app or athlete should read."""
    by_key = {f.key: f for f in findings}
    top = set(ranking.get("top_focus") or [])
    other_improvements = [
        k for k in (ranking.get("improvements") or []) if k not in top
    ]

    return {
        "headline": _build_headline(ranking, by_key),
        "summary": _build_summary_paragraph(
            ranking, by_key, deterministic["drills"]
        ),
        "safety_flags": _cards_for_keys(by_key, ranking.get("safety_flags") or []),
        "priorities": _cards_for_keys(by_key, ranking.get("top_focus") or []),
        "strengths": _cards_for_keys(by_key, (ranking.get("strengths") or [])[:5]),
        "other_improvements": _cards_for_keys(by_key, other_improvements),
        "recommended_drills": deterministic["drills"],
        "full_text": deterministic["coach_text"],
        "llm_polish": llm_text,
    }


def build_user_feedback_document(
    report: dict[str, Any],
    report_path: str | None,
    findings: list[Finding],
    ranking: dict[str, Any],
    coach: dict[str, Any],
    level: str,
    llm_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single JSON document with meta, coach (human), and analysis (machine)."""
    doc: dict[str, Any] = {
        "meta": {
            "video_path": report.get("video_path"),
            "source_report": report_path,
            "level": level,
            "view": "back",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fps": report.get("fps"),
        },
        "coach": coach,
        "analysis": {
            "events": report.get("events", {}),
            "findings": [asdict(f) for f in findings],
            "ranking": ranking,
        },
        "not_available_from_back_view": NEEDS_SIDE_ON,
    }
    if llm_payload is not None:
        doc["analysis"]["llm_payload"] = llm_payload
    return doc


# --------------------------------------------------------------------------- #
# LLM payload + guarded prompt (API call optional)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a professional fast-bowling coach analysing a BACK-VIEW delivery. "
    "You will receive a JSON object of pre-computed biomechanical findings, each with a "
    "value, unit, rating, confidence, and reference range. Rules you MUST follow:\n"
    "1. Use ONLY the numbers and ratings provided. NEVER invent or estimate any value.\n"
    "2. Ignore or explicitly defer any metric whose confidence is 'low'.\n"
    "3. Do not comment on pace/release-speed numbers; they are not measurable from this view "
    "(listed under needs_side_on).\n"
    "4. When you mention a number, cite the metric label it came from.\n"
    "5. Be encouraging and specific. Prioritise safety_flags, then top_focus.\n"
    "6. Output: (a) a 2-3 sentence summary, (b) up to 3 prioritised improvements with the why, "
    "(c) the recommended drills provided. Keep it under 220 words."
)


def build_llm_payload(report: dict[str, Any], findings: list[Finding],
                      ranking: dict[str, Any], level: str) -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "view": "back",
        "level": level,
        "events": report.get("events", {}),
        "findings": [asdict(f) for f in findings],
        "ranking": ranking,
        "not_available_from_back_view": NEEDS_SIDE_ON,
        "instructions_for_model": (
            "Write the coaching feedback per the system prompt using only these findings."
        ),
    }


def call_llm(payload: dict[str, Any]) -> str | None:
    """
    Optional LLM polish. Returns None if no API is configured so callers fall
    back to the deterministic summary. Wire your provider here.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        user_content = json.dumps(
            {k: v for k, v in payload.items() if k != "system_prompt"},
            indent=2,
        )
        model = os.environ.get("BOWLFAST_LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": payload["system_prompt"]},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001 - stay usable even if SDK missing/offline
        return f"[LLM polish unavailable: {e}]"


# --------------------------------------------------------------------------- #
# Feedback orchestration
# --------------------------------------------------------------------------- #

def generate_feedback(
    report: dict[str, Any],
    level: str = "club",
    use_llm: bool = False,
    include_llm_payload: bool = False,
    source_report_path: str | None = None,
) -> dict[str, Any]:
    findings = score_findings(report, level)
    ranking = rank_findings(findings)
    deterministic = deterministic_summary(findings, ranking)
    payload = build_llm_payload(report, findings, ranking, level)
    llm_text = call_llm(payload) if use_llm else None
    coach = build_coach_report(findings, ranking, deterministic, llm_text)

    return build_user_feedback_document(
        report=report,
        report_path=source_report_path,
        findings=findings,
        ranking=ranking,
        coach=coach,
        level=level,
        llm_payload=payload if include_llm_payload else None,
    )


# =========================================================================== #
# END FEEDBACK ENGINE
# =========================================================================== #


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="BowlFast.AI back-view analyser + feedback engine")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--output-dir", default=str(root / "outputs"))
    ap.add_argument("--bowling-arm", choices=("left", "right"), required=True)
    ap.add_argument("--entry-side", choices=("left", "right"), required=True)
    ap.add_argument(
        "--ball-model",
        default=str(root / "models" / "ball_yolo.pt"),
        help="Path to ball YOLO weights",
    )
    ap.add_argument(
        "--debug-ball",
        action="store_true",
        help="Overlay raw YOLO ball candidates, ROI window, and Kalman prediction",
    )
    ap.add_argument(
        "--bowler-calibration-json",
        default=None,
        help="Optional bowlfast-style JSON with exclusion_zones [{cx, cy, radius}] in "
        "normalized 0–1 coords to down-rank static slips/fielders",
    )
    # --- Feedback engine options (merged from bowling_feedback.py) ---
    ap.add_argument(
        "--level",
        choices=LEVELS,
        default="club",
        help="Bowler level; tightens the 'good' reference bands for feedback.",
    )
    ap.add_argument(
        "--no-feedback",
        action="store_true",
        help="Run detection only and skip generating coaching feedback.",
    )
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Polish feedback with an LLM if OPENAI_API_KEY is set (otherwise deterministic).",
    )
    ap.add_argument(
        "--include-llm-payload",
        action="store_true",
        help="Include the full llm_payload block in the feedback JSON (for developers wiring an API).",
    )
    ap.add_argument(
        "--no-merge-feedback",
        action="store_true",
        help="Do not merge the coach summary back into the *_bowlfast.json report.",
    )
    args = ap.parse_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    out_v = str(od / f"{stem}_bowlfast.mp4")
    out_j = str(od / f"{stem}_bowlfast.json")
    report = analyse_video(
        args.video,
        out_v,
        out_j,
        args.bowling_arm,
        args.entry_side,
        args.ball_model,
        debug_ball=args.debug_ball,
        bowler_calibration_json=args.bowler_calibration_json,
    )
    print(f"Wrote {out_v} and {out_j}")

    if args.no_feedback:
        return

    # --- Detection -> Feedback in one pass ---
    result = generate_feedback(
        report,
        level=args.level,
        use_llm=args.use_llm,
        include_llm_payload=args.include_llm_payload,
        source_report_path=out_j,
    )

    feedback_out = str(od / f"{stem}_bowlfast_feedback.json")
    with open(feedback_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not args.no_merge_feedback:
        report["feedback"] = {
            "level": args.level,
            "generated_at": result["meta"]["generated_at"],
            "coach": result["coach"],
            "ranking": result["analysis"]["ranking"],
            "not_available_from_back_view": result["not_available_from_back_view"],
        }
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    coach = result["coach"]
    print("=" * 70)
    print(f"BowlFast.AI feedback  |  level={args.level}  |  {report.get('video_path')}")
    print("=" * 70)
    print(coach["headline"])
    print()
    print(coach["summary"])
    if coach.get("recommended_drills"):
        print("\nRecommended drills:")
        for d in coach["recommended_drills"]:
            print(f"  - {d}")
    if coach.get("llm_polish"):
        print("\n--- LLM polish ---")
        print(coach["llm_polish"])
    print(f"\nFeedback written to:\n  {feedback_out}")
    if not args.no_merge_feedback:
        print(f"Coach summary also merged into:\n  {out_j}")


if __name__ == "__main__":
    main()


    
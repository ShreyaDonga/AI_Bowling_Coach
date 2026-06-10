"""
BowlFast.AI — back-view fast bowling analysis (single file, spec-aligned).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
YOLO_PERSON_EVERY = 3
YOLO_BALL_EVERY = 3
BALL_YOLO_CONF_PRIMARY = 0.5
BALL_YOLO_CONF_LOW = 0.3
BALL_TRACK_FAIL_CONF = 0.4
# FIX 1: Increased from 0.42 → 0.65 so the box follows the bowler faster
# instead of lagging during the run-up.
EMA_BOX_ALPHA = 0.65
BOX_EXPAND_W = 0.35
BOX_EXPAND_H = 0.28
# FIX 2: Reduced from 2.8 → 2.0 to switch to a clearly better detection sooner.
BOWLER_SCORE_SWITCH_MARGIN = 2.0

_HSV_WHITE_LO = np.array([0, 0, 200], dtype=np.uint8)
_HSV_WHITE_HI = np.array([180, 40, 255], dtype=np.uint8)


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


def hsv_ball_candidates(frame: np.ndarray) -> list[tuple[int, int, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_WHITE_LO, _HSV_WHITE_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    out: list[tuple[int, int, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12 or area > 900:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        cx, cy = x + bw // 2, y + bh // 2
        out.append((cx, cy, min(0.95, area / 200.0)))
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

    def _score_box(
        self,
        box: tuple[int, int, int, int],
        before_ffc: bool,
        ball_pos: tuple[int, int] | None,
    ) -> float:
        x1, y1, x2, y2 = box
        h, wbox = y2 - y1, x2 - x1
        cx, cy = rect_center(box)
        nx = cx / max(1.0, self.fw)
        ny = cy / max(1.0, self.fh)
        score = 0.0
        score += min(1.0, h / max(1.0, 0.58 * self.fh)) * 2.6
        score += min(1.0, wbox / max(1.0, 0.22 * self.fw)) * 0.65

        if before_ffc:
            if self.entry_side == "left":
                score += max(0.0, 1.0 - nx) * 2.4
            else:
                score += max(0.0, nx) * 2.4
            # FIX 3: Reduced centrality penalty from 2.35 → 1.0.
            # The original value hurt scoring for bowlers who are centred
            # in frame at delivery — the most critical phase.
            centrality = 1.0 - min(1.0, abs(nx - 0.5) / 0.42)
            score -= 1.0 * centrality
            score -= max(0.0, ny - 0.76) * 3.2
            if ball_pos is not None:
                bx, by = ball_pos
                dist = math.hypot(cx - bx, cy - by)
                score += max(0.0, 1.0 - dist / max(1.0, 0.48 * self.fw)) * 3.6

        if self.prev_box is not None:
            score += iou_box(box, self.prev_box) * (7.2 if not before_ffc else 5.8)
            pcx, pcy = rect_center(self.prev_box)
            drift = math.hypot(cx - pcx, cy - pcy)
            score += max(0.0, 1.0 - drift / max(1.0, 0.17 * self.fw)) * 4.2

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
            ibox, self.smooth_box = expand_smooth_box(
                self.prev_box, self.fw, self.fh, self.smooth_box
            )
            return ibox

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

        candidates = raw_boxes
        if self.prev_box is not None and candidates:
            gate = make_search_gate(self.prev_box, self.fw, self.fh, self.lost_frames)
            gated = [b for b in candidates if box_inside_gate(b, gate)]
            if gated:
                candidates = gated

        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in candidates:
            sc = self._score_box(box, before_ffc, ball_pos)
            scored.append((sc, box))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_box: tuple[int, int, int, int] | None = None
        best_sc = -1e9
        if scored:
            best_sc, best_box = scored[0]

        # FIX 2: Reduced freeze window from 14 → 6 frames so a clearly
        # better detection wins sooner instead of being suppressed.
        if (
            self.prev_box is not None
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
            return ibox

        if best_box is not None:
            self.freeze_frames = 0
            self.lost_frames = 0
            self.last_good_score = best_sc

        if (
            self.prev_box is not None
            and best_box is not None
            and before_ffc
            and iou_box(best_box, self.prev_box) < 0.18
        ):
            prev_sc = self._score_box(self.prev_box, before_ffc, ball_pos)
            if best_sc < prev_sc + BOWLER_SCORE_SWITCH_MARGIN:
                best_box = self.prev_box
                self.freeze_frames = min(6, self.freeze_frames + 1)

        # FIX 4: Don't permanently freeze on a single low-IOU frame.
        # Require CONSECUTIVE_LOW_IOU_THRESHOLD bad frames before locking,
        # so a brief occlusion or detection hiccup doesn't strand the box.
        if not before_ffc and self.prev_box is not None and best_box is not None:
            if iou_box(best_box, self.prev_box) < 0.2:
                self._consecutive_low_iou += 1
                best_box = self.prev_box  # hold position while accumulating evidence
                if self._consecutive_low_iou > 10:
                    self.frozen_offscreen = True
            else:
                self._consecutive_low_iou = 0  # reset on any good overlap frame

        if best_box is None:
            if self.prev_box is None:
                w, h = self.fw, self.fh
                roi_w = int(w * 0.45)
                x1 = 0 if self.entry_side == "left" else w - roi_w
                best_box = (x1, 0, min(w, x1 + roi_w), h)
            else:
                self.lost_frames = min(30, self.lost_frames + 1)
                best_box = self.prev_box

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
) -> list[Any] | None:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
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


def try_detect_bfc(hist: list[tuple[int, float]]) -> int | None:
    if len(hist) < 8:
        return None
    for i in range(3, len(hist) - 4):
        _, y0 = hist[i - 3]
        _, y1 = hist[i - 2]
        _, y2 = hist[i - 1]
        fi, yi = hist[i]
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
) -> int | None:
    max_f = bfc + int(fps * 1.5)
    for i in range(3, len(hist) - 4):
        fi, _ = hist[i]
        if fi <= bfc or fi > max_f:
            continue
        _, y0 = hist[i - 3]
        _, y1 = hist[i - 2]
        _, y2 = hist[i - 1]
        yi = hist[i][1]
        yp = hist[i + 1][1]
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
    b_hist: deque = field(default_factory=deque)
    nb_hist: deque = field(default_factory=deque)
    phase: str = "RUN_UP"
    events: dict = field(
        default_factory=lambda: {"BFC": None, "FFC": None, "RELEASE": None}
    )
    ball_sep_streak: int = 0
    wrist_min_y: float | None = None
    wrist_rise_frames: int = 0
    prev_ball_dist: float | None = None
    wrist_y_hist: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.bowling_arm == "right":
            self.b_ankle_idx, self.nb_ankle_idx = RIGHT_ANKLE, LEFT_ANKLE
        else:
            self.b_ankle_idx, self.nb_ankle_idx = LEFT_ANKLE, RIGHT_ANKLE
        ml = int(self.fps * 4)
        self.b_hist = deque(maxlen=ml)
        self.nb_hist = deque(maxlen=ml)

    def update(
        self,
        frame_idx: int,
        lms: list[Any] | None,
        ball_px: tuple[int, int] | None,
        ball_spd: float,
        ball_conf: float,
    ) -> None:
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
            bfc = try_detect_bfc(list(self.b_hist))
            if bfc is not None:
                self.events["BFC"] = bfc
                self.phase = "DELIVERY"
        elif self.events["FFC"] is None:
            bfc = self.events["BFC"]
            assert bfc is not None
            ffc = try_detect_ffc(list(self.nb_hist), int(bfc), self.fps)
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
                    else:
                        self.ball_sep_streak = 0
                self.prev_ball_dist = d
                if self.ball_sep_streak >= 3:
                    released = True
            if not released and lm_vis(wlm) > 0.5:
                wy = float(wlm.y)
                if self.wrist_min_y is None or wy < self.wrist_min_y:
                    self.wrist_min_y = wy
                    self.wrist_rise_frames = 0
                elif self.wrist_min_y is not None and wy >= self.wrist_min_y + 0.02:
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
        h = [(f, y) for f, y in self.wrist_y_hist if f > ffc]
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


BALL_MAX_LOST = 8
BALL_MAX_JUMP = 72
BALL_GATE_SOFT = 95
BALL_DISP_SMOOTH = 0.2


def _kalman_init(kf: cv2.KalmanFilter, cx: int, cy: int) -> None:
    kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)


def _kalman_update(kf: cv2.KalmanFilter, cx: int, cy: int) -> tuple[int, int]:
    kf.predict()
    c = kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
    return int(c[0]), int(c[1])


def _kalman_predict(kf: cv2.KalmanFilter) -> tuple[int, int]:
    p = kf.predict()
    return int(p[0]), int(p[1])


def _ball_plausible(prev: tuple[int, int] | None, nxt: tuple[int, int]) -> bool:
    if prev is None:
        return True
    return math.hypot(nxt[0] - prev[0], nxt[1] - prev[1]) < BALL_MAX_JUMP


def _pick_ball_measurement(
    dets: list[tuple[int, int, float]],
    anchor: tuple[int, int] | None,
) -> tuple[int, int, float] | None:
    if not dets:
        return None
    if anchor is None:
        dets.sort(key=lambda t: t[2], reverse=True)
        return dets[0]
    ax, ay = anchor
    scored: list[tuple[float, tuple[int, int, float]]] = []
    for cx, cy, cf in dets:
        dist = math.hypot(cx - ax, cy - ay)
        if dist > BALL_GATE_SOFT and cf < 0.55:
            continue
        if dist > BALL_MAX_JUMP * 1.35 and cf < 0.75:
            continue
        s = cf * 1.65 - dist / 85.0
        scored.append((s, (cx, cy, cf)))
    if not scored:
        nearest = min(dets, key=lambda t: math.hypot(t[0] - ax, t[1] - ay))
        dmin = math.hypot(nearest[0] - ax, nearest[1] - ay)
        if dmin < BALL_MAX_JUMP * 2.5 and nearest[2] >= 0.38:
            return nearest
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


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

    def step(self, frame_idx: int, frame: np.ndarray) -> tuple[tuple[int, int] | None, float, float, bool]:
        run_yolo = frame_idx % YOLO_BALL_EVERY == 0 or not self.ready
        yolo_dets: list[tuple[int, int, float]] = []
        if run_yolo:
            res = self.model(frame, conf=BALL_YOLO_CONF_LOW, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    cf = float(b.conf.cpu().numpy())
                    if cf < BALL_YOLO_CONF_LOW:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy().ravel())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    yolo_dets.append((cx, cy, cf))

        anchor = self.last
        meas = _pick_ball_measurement(yolo_dets, anchor)

        if meas is None and run_yolo and (self.lost >= 2 or not self.ready):
            blobs = hsv_ball_candidates(frame)
            if blobs and anchor is not None:
                blobs.sort(
                    key=lambda t: (t[0] - anchor[0]) ** 2 + (t[1] - anchor[1]) ** 2
                )
                cx, cy, sc = blobs[0]
                if math.hypot(cx - anchor[0], cy - anchor[1]) < 88:
                    meas = (cx, cy, max(0.22, sc))
            elif blobs and anchor is None:
                blobs.sort(key=lambda t: t[2], reverse=True)
                cx, cy, sc = blobs[0]
                meas = (cx, cy, max(0.22, sc))

        spd = 0.0
        self.predicted = False
        if meas is not None:
            mx, my, mcf = meas
            plausible = _ball_plausible(self.last, (mx, my))
            if not plausible and mcf < 0.62:
                meas = None

        if meas is not None:
            mx, my, mcf = meas
            plausible = _ball_plausible(self.last, (mx, my))
            if not self.ready:
                _kalman_init(self.kf, mx, my)
                self.ready = True
                self.last = (mx, my)
                self.lost = 0
                self.conf = mcf
                self.max_conf = max(self.max_conf, mcf)
            elif plausible or mcf >= 0.68:
                sx, sy = _kalman_update(self.kf, mx, my)
                self.last = (sx, sy)
                self.lost = 0
                self.conf = mcf
                self.max_conf = max(self.max_conf, mcf)
            else:
                sx, sy = _kalman_predict(self.kf)
                self.last = (sx, sy)
                self.lost += 1
                self.predicted = True
                self.conf *= 0.88
        elif self.ready and self.lost < BALL_MAX_LOST:
            sx, sy = _kalman_predict(self.kf)
            self.last = (sx, sy)
            self.lost += 1
            self.predicted = True
            self.conf *= 0.92
        else:
            if self.lost >= BALL_MAX_LOST:
                self.ready = False
                self.lost = 0
                self.last = None
                self.disp_xy = None
            self.conf = 0.0

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
            self.raw_pts.append((frame_idx, trail_xy[0], trail_xy[1]))

        if len(self.raw_pts) >= 2 and self.raw_pts[-1][0] == frame_idx:
            a, b = self.raw_pts[-2], self.raw_pts[-1]
            df = max(1, b[0] - a[0])
            spd = math.hypot(b[1] - a[1], b[2] - a[2]) / df

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


def analyse_video(
    video_path: str,
    out_video: str,
    out_json: str,
    bowling_arm: str,
    entry_side: str,
    ball_model_path: str,
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
    ball_model = YOLO(ball_model_path)
    pose_est = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    bt = BowlerTracker(person_model, bowling_arm, entry_side, fw, fh)
    ph = PhaseDetector(fps, bowling_arm, fw, fh)
    ball_st = BallTrackState(ball_model)

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
    loadup_done = False
    loadup_report: dict[str, Any] = {}
    bfc_overlay_until: int | None = None
    last_lms: list[Any] | None = None
    frozen_arrow: tuple[tuple[int, int], tuple[int, int]] | None = None
    ay_buf: deque[tuple[int, float, str]] = deque(maxlen=5)
    prev_ball_px: tuple[int, int] | None = None

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        phase = ph.phase
        box = bt.update(fi, frame, phase, ph.events, ball_pos=prev_ball_px)
        lms = extract_pose_landmarks(frame, box, pose_est, fw, fh)
        if lms is None and last_lms is not None:
            lms = last_lms
        elif lms is not None:
            last_lms = lms

        bp, bcf, bspd, bpred = ball_st.step(fi, frame)
        if ball_st.last is not None:
            prev_ball_px = ball_st.last
        elif not ball_st.ready:
            prev_ball_px = None
        ph.update(fi, lms, bp, bspd, bcf)

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
                if ph.events["BFC"] is None:
                    run_trail.append((int(mx), int(my)))
                elif not frozen_trail and ph.events["BFC"] is not None:
                    frozen_trail = list(run_trail)

        if lms is not None:
            la, ra = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]
            if lm_vis(la) > 0.5 and lm_vis(ra) > 0.5:
                ay = (la.y + ra.y) / 2
                ay_buf.append((fi, ay, "L" if la.y >= ra.y else "R"))
                if len(ay_buf) >= 3:
                    f0, y0, _ = ay_buf[-3]
                    f1, y1, _ = ay_buf[-2]
                    f2, y2, s2 = ay_buf[-1]
                    if y2 >= y1 and y2 >= y0 and y2 - min(y0, y1) > 0.003:
                        if fi - last_stride_f >= int(fps * 0.10):
                            ax = (la.x + ra.x) / 2
                            ay_mid = (la.y + ra.y) / 2
                            strides.append(
                                {
                                    "frame": f2,
                                    "x": ax,
                                    "y": ay_mid,
                                    "side": s2,
                                }
                            )
                            last_stride_f = f2

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
            for i in range(1, len(pts)):
                t = i / len(pts)
                cv2.line(disp, tuple(pts[i - 1]), tuple(pts[i]), (255, 255, 0), 2)

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

        for si, st in enumerate(strides):
            px, py = int(st["x"] * fw), int(st["y"] * fh)
            cv2.circle(disp, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(
                disp,
                str(si + 1),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
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

        if (
            st_deg is not None
            and lms is not None
            and evb is not None
            and evr is not None
            and evb <= fi <= evr
        ):
            col = (0, 255, 0) if st_deg < 25 else ((0, 165, 255) if st_deg < 40 else (0, 0, 255))
            mh = (
                (lms[LEFT_HIP].x + lms[RIGHT_HIP].x) / 2 * fw,
                (lms[LEFT_HIP].y + lms[RIGHT_HIP].y) / 2 * fh,
            )
            ms = (
                (lms[LEFT_SHOULDER].x + lms[RIGHT_SHOULDER].x) / 2 * fw,
                (lms[LEFT_SHOULDER].y + lms[RIGHT_SHOULDER].y) / 2 * fh,
            )
            cv2.line(disp, (int(mh[0]), int(mh[1])), (int(ms[0]), int(ms[1])), col, 3)
            cv2.line(
                disp,
                (int(mh[0]), int(mh[1])),
                (int(mh[0]), int(mh[1] - 80)),
                (200, 200, 200),
                2,
            )
            cv2.putText(
                disp,
                f"{st_deg:.1f} deg",
                (int(mh[0]) + 5, int(mh[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                2,
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

        if bp is not None and len(ball_st.raw_pts) >= 2:
            tail = ball_st.raw_pts[-64:]
            for i in range(1, len(tail)):
                t = i / len(tail)
                c = (int(255 * t), 255, int(255 * (1 - t)))
                cv2.line(
                    disp,
                    (tail[i - 1][1], tail[i - 1][2]),
                    (tail[i][1], tail[i][2]),
                    c,
                    2,
                )
            cv2.circle(disp, bp, 5, (255, 255, 0) if not bpred else (0, 165, 255), -1 if not bpred else 2)

        ev_y = {"BFC": 0, "FFC": 1, "RELEASE": 2}
        for name, ev in (("BFC", evb), ("FFC", evf), ("RELEASE", evr)):
            if ev is not None and fi >= ev and fi < ev + int(0.5 * fps):
                cv2.putText(
                    disp,
                    f"{name} @ {ev}",
                    (20, 100 + 28 * ev_y[name]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        text_pill(
            disp,
            [(f"Phase: {ph.phase}", (200, 255, 200))],
            (12, 12),
        )

        writer.write(disp)
        fi += 1

    ph.finalize_release_fallback()
    pose_est.close()
    cap.release()
    writer.release()

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Post metrics
    evb, evf, evr = ph.events["BFC"], ph.events["FFC"], ph.events["RELEASE"]
    inliers = parabola_inlier_frames([(a[0], a[1], a[2]) for a in ball_st.raw_pts])
    ball_status = (
        "failed"
        if ball_st.max_conf < BALL_TRACK_FAIL_CONF
        else "success"
    )

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
        if len(pre) >= 2:
            s = np.array([pre[0][1], pre[0][2]], dtype=np.float64)
            e = np.array([pre[-1][1], pre[-1][2]], dtype=np.float64)
            idir = norm_vec(float(e[0] - s[0]), float(e[1] - s[1]))
            idl = list(idir)
            idl[0] *= 0.35
            idir2 = norm_vec(idl[0], idl[1])
        else:
            idir2 = (0.0, 1.0)
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

    coach: list[str] = []
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
        "loadup": loadup_report,
        "ball_tracking": {
            "status": ball_status,
            "frames_tracked": len({p[0] for p in ball_st.raw_pts if p[0] in inliers}),
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


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="BowlFast.AI back-view analyser")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--output-dir", default=str(root / "outputs"))
    ap.add_argument("--bowling-arm", choices=("left", "right"), required=True)
    ap.add_argument("--entry-side", choices=("left", "right"), required=True)
    ap.add_argument(
        "--ball-model",
        default=str(root / "models" / "ball_yolo.pt"),
        help="Path to ball YOLO weights",
    )
    args = ap.parse_args()
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    out_v = str(od / f"{stem}_bowlfast.mp4")
    out_j = str(od / f"{stem}_bowlfast.json")
    analyse_video(
        args.video,
        out_v,
        out_j,
        args.bowling_arm,
        args.entry_side,
        args.ball_model,
    )
    print(f"Wrote {out_v} and {out_j}")


if __name__ == "__main__":
    main()
    

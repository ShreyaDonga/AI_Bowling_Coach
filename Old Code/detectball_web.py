from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


CONF_THRESH = 0.45
MAX_LOST_FRAMES = 8
MAX_JUMP_PX = 120
YOLO_EVERY_N_FRAMES = 3
INFERENCE_SIZE = 640
CREASE_DISTANCE_M = 17.68
CREASE_WIDTH_M = 3.05
RELEASE_Y_M = 0.5
ARRIVE_Y_M = 17.2
WORLD_SCALE = 30

WORLD_W = int(CREASE_WIDTH_M * WORLD_SCALE)
WORLD_H = int(CREASE_DISTANCE_M * WORLD_SCALE)
DST_POINTS = np.float32(
    [[0, 0], [WORLD_W, 0], [WORLD_W, WORLD_H], [0, WORLD_H]]
)


def ffprobe_video_info(path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,codec_name,codec_tag_string",
        "-of", "json", path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{p.stderr}")
    data = json.loads(p.stdout)
    return data["streams"][0]


def parse_ffmpeg_fraction(frac: str) -> float:
    if not frac or frac == "0/0":
        return 0.0
    num, den = frac.split("/")
    num = float(num)
    den = float(den)
    return num / den if den else 0.0


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


def create_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def kalman_init(kf: cv2.KalmanFilter, cx: int, cy: int) -> None:
    kf.statePre = np.array(
        [[np.float32(cx)], [np.float32(cy)], [0], [0]],
        dtype=np.float32,
    )
    kf.statePost = np.array(
        [[np.float32(cx)], [np.float32(cy)], [0], [0]],
        dtype=np.float32,
    )


def kalman_update(kf: cv2.KalmanFilter, cx: int, cy: int) -> tuple[int, int]:
    kf.predict()
    measurement = np.array(
        [[np.float32(cx)], [np.float32(cy)]],
        dtype=np.float32,
    )
    corrected = kf.correct(measurement)
    corrected = np.asarray(corrected).reshape(-1)
    return int(corrected[0]), int(corrected[1])


def kalman_predict(kf: cv2.KalmanFilter) -> tuple[int, int]:
    predicted = kf.predict()
    predicted = np.asarray(predicted).reshape(-1)
    return int(predicted[0]), int(predicted[1])


def is_plausible(
    prev_pos: tuple[int, int] | None,
    new_pos: tuple[int, int],
    max_jump: int = MAX_JUMP_PX,
) -> bool:
    if prev_pos is None:
        return True
    return float(np.hypot(new_pos[0] - prev_pos[0], new_pos[1] - prev_pos[1])) < max_jump


def pixel_to_world(H: np.ndarray, px: int, py: int) -> tuple[float, float]:
    pt = np.float32([[[px, py]]])
    world = cv2.perspectiveTransform(pt, H)
    return world[0][0][0] / WORLD_SCALE, world[0][0][1] / WORLD_SCALE


def world_to_pixel(H_inv: np.ndarray, wx_m: float, wy_m: float) -> tuple[int, int]:
    pt = np.float32([[[wx_m * WORLD_SCALE, wy_m * WORLD_SCALE]]])
    px = cv2.perspectiveTransform(pt, H_inv)
    return int(px[0][0][0]), int(px[0][0][1])


def build_homography_from_points(
    calibration_points: list[list[float]] | list[tuple[float, float]] | None
) -> np.ndarray | None:
    if calibration_points is None:
        return None
    if len(calibration_points) != 4:
        raise ValueError("calibration_points must contain exactly 4 points")
    src = np.float32(calibration_points)
    H, _ = cv2.findHomography(src, DST_POINTS)
    return H


def track_ball(
    video_path: str,
    model_path: str,
    output_dir: str,
    calibration_points: list[list[float]] | list[tuple[float, float]] | None = None,
    save_annotated_video: bool = True,
    preferred_fps: int = 60,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    video_path = str(Path(video_path))
    model_path = str(Path(model_path))
    stem = Path(video_path).stem

    cfr_input_path, fps = ensure_cfr_input(video_path, preferred_fps=preferred_fps)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(cfr_input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfr_input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    homography = build_homography_from_points(calibration_points)
    H_inv = np.linalg.inv(homography) if homography is not None else None

    annotated_video_path = os.path.join(output_dir, f"{stem}_ball_annotated.mp4")
    out = make_writer(annotated_video_path, float(fps), width, height) if save_annotated_video else None

    crease_overlay = np.zeros_like(first_frame)
    if H_inv is not None:
        bc_l = world_to_pixel(H_inv, 0, 0)
        bc_r = world_to_pixel(H_inv, CREASE_WIDTH_M, 0)
        bat_l = world_to_pixel(H_inv, 0, CREASE_DISTANCE_M)
        bat_r = world_to_pixel(H_inv, CREASE_WIDTH_M, CREASE_DISTANCE_M)
        cv2.line(crease_overlay, bc_l, bc_r, (0, 140, 255), 3)
        cv2.line(crease_overlay, bat_l, bat_r, (255, 80, 0), 3)
        cv2.line(crease_overlay, bc_l, bat_l, (180, 180, 180), 1)
        cv2.line(crease_overlay, bc_r, bat_r, (180, 180, 180), 1)

    kf = create_kalman()
    kalman_ready = False
    lost_frames = 0
    last_pos: tuple[int, int] | None = None
    ball_path: deque[tuple[int, int]] = deque(maxlen=64)
    ball_track_log: dict[int, dict[str, Any]] = {}
    yolo_times: list[float] = []
    scale_x = width / INFERENCE_SIZE
    scale_y = height / int(height * INFERENCE_SIZE / width)

    speed_state = "WAITING"
    release_frame = None
    arrive_frame = None
    final_speed_kmh = None
    prev_wy_world = None

    t_start = time.time()
    frame_count = 1

    def update_speed_state(frame_no: int, world_y: float) -> None:
        nonlocal speed_state, release_frame, arrive_frame, final_speed_kmh
        if speed_state == "WAITING" and world_y >= RELEASE_Y_M:
            speed_state = "RELEASED"
            release_frame = frame_no
        if speed_state == "RELEASED" and world_y >= ARRIVE_Y_M:
            speed_state = "ARRIVED"
            arrive_frame = frame_no
            frames_taken = arrive_frame - release_frame
            if frames_taken > 0:
                time_sec = frames_taken / fps
                final_speed_kmh = (CREASE_DISTANCE_M / time_sec) * 3.6

    def process_frame(frame: np.ndarray, frame_no: int) -> np.ndarray:
        nonlocal kalman_ready, lost_frames, last_pos, final_speed_kmh, prev_wy_world

        detection_this_frame = False
        best_box = None
        best_score = 0.0

        if (frame_no % YOLO_EVERY_N_FRAMES == 0) or not kalman_ready:
            infer_h = int(height * INFERENCE_SIZE / width)
            small = cv2.resize(frame, (INFERENCE_SIZE, infer_h))
            t0 = time.time()
            results = model(small, conf=CONF_THRESH, verbose=False, imgsz=INFERENCE_SIZE)
            yolo_times.append(time.time() - t0)

            for r in results:
                for i, box in enumerate(r.boxes.xyxy):
                    score = float(r.boxes.conf[i])
                    if score < CONF_THRESH:
                        continue

                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if not is_plausible(last_pos, (cx, cy)):
                        continue

                    if score > best_score:
                        best_score = score
                        best_box = (x1, y1, x2, y2, cx, cy)

        if best_box is not None:
            x1, y1, x2, y2, cx, cy = best_box

            if not kalman_ready:
                kalman_init(kf, cx, cy)
                kalman_ready = True

            smooth_x, smooth_y = kalman_update(kf, cx, cy)
            last_pos = (smooth_x, smooth_y)
            lost_frames = 0
            detection_this_frame = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{best_score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        elif kalman_ready and lost_frames < MAX_LOST_FRAMES:
            smooth_x, smooth_y = kalman_predict(kf)
            last_pos = (smooth_x, smooth_y)
            lost_frames += 1
            detection_this_frame = True

            cv2.circle(frame, (smooth_x, smooth_y), 10, (0, 165, 255), 2)
            cv2.putText(
                frame,
                f"pred+{lost_frames}",
                (smooth_x + 12, smooth_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 165, 255),
                1,
            )

        else:
            if lost_frames >= MAX_LOST_FRAMES:
                kalman_ready = False
                lost_frames = 0
                last_pos = None

        if detection_this_frame and last_pos is not None and homography is not None:
            ball_path.append(last_pos)
            _, wy = pixel_to_world(homography, last_pos[0], last_pos[1])

            if prev_wy_world is not None:
                _ = (wy - prev_wy_world) * fps

            prev_wy_world = wy
            update_speed_state(frame_no, wy)

        if H_inv is not None:
            frame = cv2.addWeighted(frame, 1.0, crease_overlay, 0.65, 0.0)

        for i in range(1, len(ball_path)):
            alpha = i / len(ball_path)
            colour = (0, int(255 * (1 - alpha)), int(255 * alpha))
            cv2.line(frame, ball_path[i - 1], ball_path[i], colour, 2)

        if detection_this_frame and last_pos:
            cv2.circle(frame, last_pos, 5, (0, 0, 255), -1)
            source = "yolo" if best_box is not None else "kalman"
            ball_track_log[frame_no] = {
                "cx": int(last_pos[0]),
                "cy": int(last_pos[1]),
                "source": source,
            }

        status = (
            "TRACKING"
            if kalman_ready and lost_frames == 0
            else f"PREDICTED ({lost_frames}f)"
            if kalman_ready
            else "SEARCHING"
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
        )

        if final_speed_kmh is not None:
            cv2.putText(
                frame,
                f"Speed: {final_speed_kmh:.1f} km/h",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return frame

    annotated = process_frame(first_frame.copy(), frame_count)
    if out is not None:
        out.write(annotated)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = process_frame(frame, frame_count)

        if out is not None:
            out.write(annotated)

    cap.release()
    if out is not None:
        out.release()

    total_time = time.time() - t_start
    balltrack_path = os.path.join(output_dir, f"{stem}_balltrack.json")

    balltrack_out = {
        "video_path": video_path,
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames": {str(k): v for k, v in ball_track_log.items()},
        "speed": {
            "state": speed_state,
            "release_frame": release_frame,
            "arrive_frame": arrive_frame,
            "final_speed_kmh": final_speed_kmh,
        },
        "calibration_points": calibration_points,
    }

    with open(balltrack_path, "w", encoding="utf-8") as f:
        json.dump(balltrack_out, f, indent=2)

    if cfr_input_path != video_path:
        shutil.rmtree(os.path.dirname(cfr_input_path), ignore_errors=True)

    return {
        "video_path": video_path,
        "annotated_video_path": annotated_video_path if save_annotated_video else None,
        "balltrack_json_path": balltrack_path,
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames_processed": frame_count,
        "processing_seconds": total_time,
        "yolo_avg_ms": float(np.mean(yolo_times) * 1000.0) if yolo_times else None,
        "speed_kmh": final_speed_kmh,
        "calibration_used": homography is not None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web-safe ball tracking")
    parser.add_argument("video_path")
    parser.add_argument("--model-path", default="ball_yolo.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--calibration-json", default=None, help="JSON file with 4 pitch points")
    args = parser.parse_args()

    calibration_points = None
    if args.calibration_json:
        with open(args.calibration_json, "r", encoding="utf-8") as f:
            calibration_points = json.load(f)

    result = track_ball(
        video_path=args.video_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_points=calibration_points,
    )
    print(json.dumps(result, indent=2))
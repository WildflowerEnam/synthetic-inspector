"""
    Edge simulator for Synthetic Inspector. 
    Simulates a Jetson Orin Nano running defect detection on a 
    live camera feed, but instead of physical hardware it reads a video file and sends each
    frame to the inference server over HTTP. 
"""
import json 
import time 
from pathlib import Path 

import cv2
import numpy as np 
import requests 
import argparse

from src.config import (
    ENDPOINT_URL,
    LATENCY_BUDGET_MS,
    DASHBOARD_WINDOW_NAME,
    DEFAULT_VIDEO_FPS,
)


# Policy
PREDICT_PATH = "/predict"
HEALTH_PATH = "/health"
TIMEOUT_SECONDS = 5
JPEG_QUALITY = 85


# HTTP client
def check_server(endpoint_base: str) -> None:
    """
        Ping /health before starting. Fail fast if the 
        server is down or 
        the model is not loaded. Costs 50ms and saves 5 mins of 
        debugging
    """
    url = endpoint_base.rstrip("/") + HEALTH_PATH
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Server not reachable at {url}. Start server.py first. Error: {e}"
        )
    
    data = response.json()
    if not data.get("model_loaded", False):
        raise RuntimeError(
            f"Server is up at {url} but the model is not loaded."
            f"Run train.py to produce a model, then restart server.py"
        )

    print(f"[edge] Server OK at {endpoint_base}")

     
def predict_frame(endpoint_base: str, frame: np.ndarray) -> tuple:
    """
        Send one frame to the server. Return (result_dict, latency_ms).
        ...
    """
    url = endpoint_base.rstrip("/") + PREDICT_PATH
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed - frame is not a valid image")

    start = time.perf_counter()
    try:
        response = requests.post(
            url, 
            files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Prediction request failed: {e}")
    
    latency_ms = (time.perf_counter() - start) * 1000.0
    return response.json(), latency_ms


def draw_overlay(frame, result, latency_ms, frame_idx, total_frames) -> np.ndarray:
    """
        Draw PASS/FAIL, class, confidence, latency, frame counter. 
    """
    out = frame.copy()
    is_defect = result.get("is_defect", False)
    class_name = result.get("class", "?")
    confidence = result.get("confidence", 0.0)

    over_budget = latency_ms > LATENCY_BUDGET_MS

    if is_defect:
        verdict = "FAIL"
        color = (0, 0, 255)
    else:
        verdict = "PASS"
        color = (0, 200, 0)
    
    cv2.rectangle(out, (0, 0), (out.shape[1], 80), color, thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, verdict, (20, 55), font, 1.6, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(
        out, 
        f"{class_name} {confidence:.2%}",
        (200, 50),
        font, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )

    latency_text = f"{latency_ms:.0f} ms"
    latency_color = (0, 0, 255) if over_budget else (0, 200, 0)
    cv2.putText(out, latency_text, (20, 110), font, 0.8, latency_color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"budget {LATENCY_BUDGET_MS} ms",
        (140, 110), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )

    counter = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
    h, w = out.shape[:2]

    cv2.putText(out, counter, (w - 150, h - 20), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out 


def summarize_latencies(latencies: list) -> dict:
    """
        Return mean, P50, P95, P99, and budget-compliance fraction.
    """
    if not latencies:
        return {"count": 0}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),

        "under_budget_pct": float((arr <= LATENCY_BUDGET_MS).mean() * 100.0),

    }
   


def run(source: str, endpoint_base: str, headless: bool) -> None:
    check_server(endpoint_base)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    delay_ms = int(1000.0 / fps) if fps > 0 else 33
    writer = None
    if headless:
        writer = cv2.VideoWriter(
            "demo/output_with_overlay.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

    print(f"[edge] Source: {source}")
    print(f"[edge] FPS: {fps:.1f}, total frames: {total_frames or 'unknown'}")
    print(f"[edge] Press 'q' in the window to quit (headless={headless})")

    frame_idx = 0
    latencies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[edge] End of video")
            break
        frame_idx += 1
        result, latency_ms = predict_frame(endpoint_base, frame)
        latencies.append(latency_ms)

        overlay = draw_overlay(frame, result, latency_ms, frame_idx, total_frames)
        if writer is not None:
            writer.write(overlay)
        else:
            cv2.imshow(DASHBOARD_WINDOW_NAME, overlay)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("[edge] Quit by user")
                break
        
        if frame_idx % 10 == 0:
            print(
                f"[edge] frame {frame_idx}/{total_frames or '?'}"
                f"| {result.get('class')}"
                f"({result.get('confidence', 0):.2f})"
                f"| {latency_ms:.0f} ms"
            )


    cap.release()
    if writer is not None:
        writer.release()
        print("[edge] Saved overlay video -> demo/output_with_overlay.mp4")
        
    if not headless:
        cv2.destroyAllWindows()
    
    stats = summarize_latencies(latencies)
    print("\n[edge] Latency summary:")
    print(json.dumps(stats, indent=2))


    if stats["count"] > 0:
        print(
            f"\n[edge] {stats['under_budget_pct']:.1f}% of frames under "
            f"{LATENCY_BUDGET_MS} ms budget"
        )  



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="Video path or camera index ('0' for default webcam)")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT_URL or "http://localhost:8080",
                        help="Server base URL (default: ENDPOINT_URL from config, else localhost)")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a window (useful for CI or scripted runs)")
    args = parser.parse_args()
    run(args.source, args.endpoint, args.headless)


if __name__ == "__main__":
    main()

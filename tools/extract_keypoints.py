"""Extract MediaPipe Holistic keypoints from videos listed in a CSV (video_path, part, video_id, view)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract(video_path, holistic, cv2):
    cap = cv2.VideoCapture(str(video_path))
    pose, lh, rh = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose.append(np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark],
                             np.float32) if res.pose_landmarks else np.zeros((33, 4), np.float32))
        lh.append(np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark], np.float32)
                  if res.left_hand_landmarks else np.zeros((21, 3), np.float32))
        rh.append(np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark], np.float32)
                  if res.right_hand_landmarks else np.zeros((21, 3), np.float32))
    cap.release()
    z = lambda n, c: np.zeros((0, n, c), np.float32)  # noqa: E731
    return (np.stack(pose) if pose else z(33, 4), np.stack(lh) if lh else z(21, 3),
            np.stack(rh) if rh else z(21, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_complexity", type=int, default=1)
    a = ap.parse_args()
    import cv2
    import mediapipe as mp
    df = pd.read_csv(a.manifest)
    with mp.solutions.holistic.Holistic(static_image_mode=False,
                                        model_complexity=a.model_complexity) as h:
        for i, r in enumerate(df.itertuples(), 1):
            dst = Path(a.out) / f"Part_{int(r.part)}_{r.view}" / f"{int(r.video_id):06d}.npz"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            pose, lh, rh = extract(r.video_path, h, cv2)
            np.savez_compressed(dst, pose=pose, left_hand=lh, right_hand=rh)
            if i % 500 == 0:
                print(f"  {i:,}/{len(df):,}", flush=True)


if __name__ == "__main__":
    main()

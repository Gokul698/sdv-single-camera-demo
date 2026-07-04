"""
Session 3: Live Stream
------------------------
Opens the camera and displays a live video feed with an FPS overlay.

Controls:
    q  -> quit
    s  -> save a snapshot to disk

Usage:
    python3 session3_live_stream.py [camera_index]
"""

import sys
import time
import cv2

DEFAULT_CAMERA_INDEX = 0


def main():
    camera_index = DEFAULT_CAMERA_INDEX
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: could not open camera at index {camera_index}")
        print("Run session2_device_check.py first to find a working index.")
        return

    print("Live stream started. Press 'q' to quit, 's' to save a snapshot.")

    prev_time = time.time()
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: failed to grab frame. Retrying...")
            continue

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Session 3 - Live Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_count += 1
            filename = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Live stream stopped.")


if __name__ == "__main__":
    main()
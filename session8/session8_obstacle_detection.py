"""
Session 8: Obstacle Detection
------------------------------
Motion-based obstacle detection using background subtraction (MOG2).
This is NOT AI object classification -- it flags anything that moves
against the learned background as a potential obstacle. Good as a
lightweight demo when torch/ultralytics/OpenVINO aren't installed.

Usage:
    python3 session8_obstacle_detection.py [camera_index]
"""

import sys
import cv2

DEFAULT_CAMERA_INDEX = 0
MIN_CONTOUR_AREA = 800  # ignore tiny noise blobs


def main():
    camera_index = DEFAULT_CAMERA_INDEX
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: could not open camera at index {camera_index}")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True
    )

    print("Obstacle detection started. Press 'q' to quit.")
    print("Hold still for a second at startup so the background model can learn.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: failed to grab frame. Retrying...")
            continue

        fg_mask = back_sub.apply(frame)

        # Remove shadow pixels (gray value 127) and clean up noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.erode(fg_mask, None, iterations=1)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output = frame.copy()
        obstacle_count = 0

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue

            obstacle_count += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                output,
                "Obstacle",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            output,
            f"Obstacles detected: {obstacle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Session 8 - Obstacle Detection", output)
        cv2.imshow("Foreground Mask (debug)", fg_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Obstacle detection stopped.")


if __name__ == "__main__":
    main()
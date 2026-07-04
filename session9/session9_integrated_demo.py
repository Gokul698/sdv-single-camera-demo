"""
Session 9: Integrated Demo
------------------------------
Combines Session 8's lane detection and obstacle detection into a
single pipeline running on one camera feed, in real time.

Usage:
    python3 session9_integrated_demo.py [camera_index]
"""

import sys
import numpy as np
import cv2

DEFAULT_CAMERA_INDEX = 0
MIN_CONTOUR_AREA = 800


# ---------- Lane detection helpers (from session8_lane_detection.py) ----------

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.45 * width), int(0.55 * height)),
        (int(0.55 * width), int(0.55 * height)),
        (int(0.95 * width), height),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def unpack_line(line):
    arr = np.array(line).reshape(-1)
    x1, y1, x2, y2 = arr[:4]
    return int(x1), int(y1), int(x2), int(y2)


def draw_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=100,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = unpack_line(line)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame


# ---------- Obstacle detection helper (from session8_obstacle_detection.py) ----------

def draw_obstacles(frame, back_sub):
    fg_mask = back_sub.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.erode(fg_mask, None, iterations=1)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    obstacle_count = 0
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        obstacle_count += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Obstacle",
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return frame, obstacle_count


# ---------- Main integrated loop ----------

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

    print("Integrated demo started (lane + obstacle detection). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: failed to grab frame. Retrying...")
            continue

        output = frame.copy()
        output = draw_lanes(output)
        output, obstacle_count = draw_obstacles(output, back_sub)

        cv2.putText(
            output,
            f"Obstacles detected: {obstacle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Session 9 - Integrated Demo (Lane + Obstacle)", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Integrated demo stopped.")


if __name__ == "__main__":
    main()
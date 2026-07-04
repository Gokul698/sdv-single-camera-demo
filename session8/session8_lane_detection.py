"""
Session 8: Lane Detection
------------------------------
Classic OpenCV lane-detection pipeline:
    1. Grayscale + blur
    2. Canny edge detection
    3. Region-of-interest mask (bottom trapezoid of the frame)
    4. HoughLinesP to find line segments
    5. Draw lines back onto the original frame

Includes a robust line-unpacking helper because different OpenCV builds
return HoughLinesP results in slightly different shapes
([[x1,y1,x2,y2]] vs [x1,y1,x2,y2] vs [[[x1,y1,x2,y2]]]).

Usage:
    python3 session8_lane_detection.py [camera_index]
"""

import sys
import numpy as np
import cv2

DEFAULT_CAMERA_INDEX = 0


def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Trapezoid covering the lower half of the frame
    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.45 * width), int(0.55 * height)),
        (int(0.55 * width), int(0.55 * height)),
        (int(0.95 * width), height),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def unpack_line(line):
    """
    Handle the different shapes OpenCV may return for a single line
    from HoughLinesP: [[x1,y1,x2,y2]], [x1,y1,x2,y2], or [[[x1,y1,x2,y2]]].
    """
    arr = np.array(line).reshape(-1)
    x1, y1, x2, y2 = arr[:4]
    return int(x1), int(y1), int(x2), int(y2)


def detect_lanes(frame):
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

    out = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = unpack_line(line)
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return out, edges


def main():
    camera_index = DEFAULT_CAMERA_INDEX
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: could not open camera at index {camera_index}")
        return

    print("Lane detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: failed to grab frame. Retrying...")
            continue

        result, edges = detect_lanes(frame)

        cv2.imshow("Session 8 - Lane Detection", result)
        cv2.imshow("Edges (debug)", edges)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Lane detection stopped.")


if __name__ == "__main__":
    main()
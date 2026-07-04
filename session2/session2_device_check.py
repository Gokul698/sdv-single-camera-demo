"""
Session 2: Device Check
------------------------
Scans for available camera devices (index 0-5), and for each working
device prints its resolution, FPS, and grabs one test frame.

Usage:
    python3 session2_device_check.py
"""

import cv2

MAX_DEVICES_TO_CHECK = 6


def check_device(index):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    info = {
        "index": index,
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_shape": frame.shape,
    }

    cap.release()
    return info


def main():
    print("Scanning for camera devices...\n")
    found_any = False

    for i in range(MAX_DEVICES_TO_CHECK):
        info = check_device(i)
        if info is None:
            print(f"  /dev/video{i} (index {i}) -> not available")
            continue

        found_any = True
        print(f"  /dev/video{i} (index {i}) -> AVAILABLE")
        print(f"      Resolution : {info['width']} x {info['height']}")
        print(f"      FPS        : {info['fps']:.2f}")
        print(f"      Frame shape: {info['frame_shape']}")
        print()

    if not found_any:
        print("\nNo camera devices found.")
        print("Check that the camera is plugged in and not in use by another process.")
        print("On Linux, you can also run: ls /dev/video*")
    else:
        print("Device check complete. Use the working index above in session3+.")


if __name__ == "__main__":
    main()
import cv2

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("No camera detected")
    raise SystemExit
print("Camera Connected")
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

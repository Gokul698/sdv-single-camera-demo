import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape

    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width // 2, int(height * 0.6)),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=50
    )

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            cv2.line(frame,
                     (x1, y1),
                     (x2, y2),
                     (0,255,0),
                     3)

    cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
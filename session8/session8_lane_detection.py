
import cv2
import numpy as np

def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, h),
        (int(w*0.45), int(h*0.60)),
        (int(w*0.55), int(h*0.60)),
        (w, h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera")
    raise SystemExit

print("Lane Detection Started - Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(roi,1,np.pi/180,40,minLineLength=40,maxLineGap=80)

   if lines is not None:
    for line in lines:

        # OpenCV may return [[x1,y1,x2,y2]] or [x1,y1,x2,y2]
        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line

        cv2.line(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )

    cv2.putText(out,"Lane Detection",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Session8 Lane Detection", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

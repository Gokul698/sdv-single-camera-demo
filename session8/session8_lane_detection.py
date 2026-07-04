import cv2, numpy as np

def region(img):
    h,w=img.shape
    mask=np.zeros_like(img)
    pts=np.array([[(0,h),(w//2,h//2),(w,h)]],np.int32)
    cv2.fillPoly(mask,pts,255)
    return cv2.bitwise_and(img,mask)

cap=cv2.VideoCapture(0)
while True:
    ok,f=cap.read()
    if not ok: break
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    b=cv2.GaussianBlur(g,(5,5),0)
    e=cv2.Canny(b,50,150)
    roi=region(e)
    lines=cv2.HoughLinesP(roi,1,np.pi/180,50,minLineLength=50,maxLineGap=50)
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2=l[0]
            cv2.line(f,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.imshow("Lane Detection",f)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release();cv2.destroyAllWindows()

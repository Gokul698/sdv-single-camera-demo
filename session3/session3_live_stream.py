import cv2,time
cap=cv2.VideoCapture(0)
prev=time.time()
while True:
    ok,frame=cap.read()
    if not ok: break
    now=time.time()
    fps=1/(now-prev)
    prev=now
    cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),0,1,(0,255,0),2)
    cv2.imshow("Live Stream",frame)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'): break
cap.release()
cv2.destroyAllWindows()

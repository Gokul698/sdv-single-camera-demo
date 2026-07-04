# Simple monocular obstacle approximation using Haar cascade.
import cv2
cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while True:
    ok,f=cap.read()
    if not ok: break
    gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    objs=cascade.detectMultiScale(gray,1.2,5)
    for (x,y,w,h) in objs:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(f,"Obstacle",(x,y-10),0,0.8,(0,0,255),2)
    cv2.imshow("Obstacle Detection",f)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release();cv2.destroyAllWindows()

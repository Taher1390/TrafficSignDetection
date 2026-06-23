import cv2
from traffic_sign import TrafficSign

cap = cv2.VideoCapture(0)
n = 0
traffic_sign_ = None

while True:
    ret, img = cap.read()
    blur_img = cv2.blur(img, (7, 7))

    tr_sign = TrafficSign(img, blur_img, filter=0.9)
    if n:
        if traffic_sign_ == tr_sign.traffic_sign:
            n += 1
        else:
            n = 0
    else:
        traffic_sign_ = tr_sign.traffic_sign
        n = 1
        
    if n == 10:         
        print(traffic_sign_)
        
    cv2.imshow('hsv img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from traffic_sign import TrafficSign

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    blur_img = cv2.blur(img, (7, 7))
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    tr_sign = TrafficSign(img, hsv_img)
    print(tr_sign.traffic_sign)
    cv2.imshow('hsv img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

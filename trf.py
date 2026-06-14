import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_pixels = 10000
    enough_area = False
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([135, 255, 255])

    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2

    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)
    
    if red_pixels >= min_pixels:
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(red_mask, contours, -1, (0, 255, 0), 2)
            
    elif blue_pixels >= min_pixels:
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blue_mask, contours, -1, (0, 255, 0), 2)
        
    cv2.imshow('red mask', red_mask)
    cv2.imshow('blue mask', blue_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

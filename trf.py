import cv2
import numpy as np
import math
import onnxruntime as ort
import ultralytics

cap = cv2.VideoCapture(0)

session = ort.InferenceSession("traffic_sign.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

keys = {'stop': 0, 'straight': 1, 'left': 2, 'right': 3}

while True:
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_pixels = 10000
    
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
    
    red_blue_mask = cv2.bitwise_or(blue_mask, red_mask)
    
    if red_pixels >= min_pixels or blue_pixels >= min_pixels:
        contours, _ = cv2.findContours(red_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"]:
                area = cv2.contourArea(cnt)
                if area >= min_pixels:
                    cx = abs(int(M["m10"] / M["m00"]))
                    cy = abs(int(M["m01"] / M["m00"]))
                    r = int((area / math.pi) ** 0.5)
                    w = 0.2
                    crop = frame[int(cy - r - (w * r)): int(cy + r + (w * r)), int(cx - r - (w * r)): int(cx + r + (w * r))]
                    crop = np.expand_dims(cv2.resize(crop, (128, 128)).astype(np.float32) / 255.0, axis=0)
                    
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        result = session.run([output_name], {input_name: crop})
                        print(result)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import math
import onnxruntime as ort

class TrafficSign:
    keys = ['stop', 'forward', 'left', 'right', 'no_entry', 'dead_end']
    is_traffic_sign = False
    traffic_sign = None
    
    def __init__(self, bgr_img, hsv_img, min_area = 4000, model_name = "traffic_sign.onnx"):
        self.hsv_img = hsv_img
        self.bgr_img = bgr_img
        self.min_area = min_area
        self.model_name = model_name
        
        self.session = ort.InferenceSession(self.model_name)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_blue = np.array([100, 100, 100])
        self.upper_blue = np.array([135, 255, 255])

        self.red_mask1 = cv2.inRange(self.hsv_img, self.lower_red1, self.upper_red1)
        self.red_mask2 = cv2.inRange(self.hsv_img, self.lower_red2, self.upper_red2)
        self.red_mask = self.red_mask1 + self.red_mask2

        self.blue_mask = cv2.inRange(self.hsv_img, self.lower_blue, self.upper_blue)

        self.red_pixels = cv2.countNonZero(self.red_mask)
        self.blue_pixels = cv2.countNonZero(self.blue_mask)
        
        self.red_blue_mask = cv2.bitwise_or(self.blue_mask, self.red_mask)

        if self.red_pixels >= self.min_area or self.blue_pixels >= self.min_area:
            self.contours, _ = cv2.findContours(self.red_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
            for cnt in self.contours:
                M = cv2.moments(cnt)
                if M["m00"]:
                    self.area = cv2.contourArea(cnt)
                    if self.area >= self.min_area:
                        cx = abs(int(M["m10"] / M["m00"]))
                        cy = abs(int(M["m01"] / M["m00"]))
                        r = int((self.area / math.pi) ** 0.5)
                        w = 0.2
                        self.crop = self.bgr_img[int(cy - r - (w * r)): int(cy + r + (w * r)), int(cx - r - (w * r)): int(cx + r + (w * r))]
                        
                        if self.crop.shape[0] > 0 and self.crop.shape[1] > 0:
                            self.crop = np.expand_dims(cv2.resize(self.crop, (128, 128)).astype(np.float32) / 255.0, axis=0)
                            self.result = list(self.session.run([self.output_name], {self.input_name: self.crop})[0][0])
                            for prob in self.result:
                                if prob > 0.8:
                                    self.is_traffic_sign = True
                                    break
                            if self.is_traffic_sign:
                                self.is_traffic_sign = False
                                self.traffic_sign = self.keys[self.result.index(max(self.result))]
                            else:
                                self.traffic_sign = None   

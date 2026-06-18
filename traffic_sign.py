import cv2
import numpy as np
import math
import onnxruntime as ort

class TrafficSign:
    keys = ['stop', 'forward', 'left', 'right', 'no_entry', 'dead_end']
    traffic_sign = None
    
    def __init__(self, img, blur_img, min_area = 4000, model_name = "traffic_sign.onnx", filter = 0.85, mode = 'bgr'):
        if mode == 'bgr':
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        self.hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2HSV)
        self.img = img
        self.min_area = min_area
        self.model_name = model_name
        self.filter = filter
        
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
                        self.crop = self.img[int(cy - r - (w * r)): int(cy + r + (w * r)), int(cx - r - (w * r)): int(cx + r + (w * r))]
                        
                        if self.crop.shape[0] > 0 and self.crop.shape[1] > 0:
                            self.crop = np.expand_dims(cv2.resize(self.crop, (128, 128)).astype(np.float32) / 255.0, axis=0)
                            self.result = list(self.session.run([self.output_name], {self.input_name: self.crop})[0][0])
                            prob = max(self.result)
                            if prob >= self.filter:
                                self.traffic_sign = self.keys[self.result.index(prob)]

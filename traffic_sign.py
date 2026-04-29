import cv2
import numpy as np
import math

class TrafficSign:
    def __init__(self, hsv_img, min_pixels = 10000):
        self.hsv_img = hsv_img
        self.min_pixels = min_pixels
        self.enough_area = False

        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_blue = np.array([100, 100, 100])
        self.upper_blue = np.array([135, 255, 255])

        self.red_mask1 = cv2.inRange(self.hsv_img, self.lower_red1, self.upper_red1)
        self.red_mask2 = cv2.inRange(self.hsv_img, self.lower_red2, self.upper_red2)
        self.red_mask = self.red_mask1 + self.red_mask2

        self.blue_mask = cv2.inRange(hsv_img, self.lower_blue, self.upper_blue)

        self.red_pixels = cv2.countNonZero(self.red_mask)
        self.blue_pixels = cv2.countNonZero(self.blue_mask)

        self.sign = None

        if self.red_pixels >= self.min_pixels:
            self.contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.red_mask, self.contours, -1, (0, 255, 0), 2)
            for cnt in self.contours:
                self.m = cv2.moments(cnt)
                if self.m["m00"]:
                    self.area = cv2.contourArea(cnt)
                    if self.area >= self.min_pixels:
                        self.sign = 'stop'
                    
        elif self.blue_pixels >= self.min_pixels:
            self.contours, _ = cv2.findContours(self.blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.blue_mask, self.contours, -1, (0, 255, 0), 2)
            for cnt in self.contours:
                self.M = cv2.moments(cnt)
                if self.M["m00"]:
                    self.area = cv2.contourArea(cnt)
                    if self.area >= self.min_pixels:
                        self.cx = int(self.M["m10"] / self.M["m00"])
                        self.cy = int(self.M["m01"] / self.M["m00"])
                        self.r = int((self.area / math.pi) ** 0.5)
                        self.enough_area = True
                        break
                    else:
                        self.enough_area = False
            
            if self.enough_area:
                w = self.cx - self.r
                h = self.cy - self.r
                diff = int((45 / 364) * 2 * self.r)
                self.mask_right = self.blue_mask[h: h+2*self.r, w+self.r: w+2*self.r-diff]
                self.mask_left = self.blue_mask[h: h+2*self.r, w+diff: w+self.r]
                self.mask_up = self.blue_mask[h: h+self.r, w+diff: w+2*self.r-diff]
                self.mask_down = self.blue_mask[h+self.r: h+2*self.r, w+diff: w+2*self.r-diff]

                self.right_pixels = cv2.countNonZero(self.mask_right)
                self.left_pixels = cv2.countNonZero(self.mask_left)
                self.up_pixels = cv2.countNonZero(self.mask_up)
                self.down_pixels = cv2.countNonZero(self.mask_down)
                self.w_dif = self.left_pixels - self.right_pixels
                self.h_dif = self.up_pixels - self.down_pixels

                if abs(self.h_dif) > abs(self.w_dif):
                    self.sign = 'go straight'
                
                elif abs(self.h_dif) < abs(self.w_dif):
                    if self.w_dif > 0:
                        self.sign = 'go right'
                    elif self.w_dif < 0:
                        self.sign = 'go left'

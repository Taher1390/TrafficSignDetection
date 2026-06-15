import ultralytics
import cv2

cap = cv2.VideoCapture(0)
model = ultralytics.YOLO('model_- 24 december 2025 18_01.onnx') # 'model_- 24 december 2025 18_01.onnx' is the best

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()
    results = model(frame, imgsz=640, conf=0.87, device='cpu')

    annotated_frame = results[0].plot()

    cv2.imshow('annotated_frame', annotated_frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

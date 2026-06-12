import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

# dá tempo da câmera estabilizar
import time
time.sleep(2)

ret, frame = cap.read()

if ret:
    cv2.imwrite("base.jpg", frame)
    print("Imagem base salva!")

cap.release()
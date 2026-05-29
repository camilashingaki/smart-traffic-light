import cv2
import time

cap = cv2.VideoCapture(2)

time.sleep(2)  # dá tempo da câmera iniciar

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar frame")
        continue

    x1, y1 = 80, 50
    x2, y2 = 300, 600

    frame_crop = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Cropado", frame_crop)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
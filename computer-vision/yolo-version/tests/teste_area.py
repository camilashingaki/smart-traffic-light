import cv2

cap = cv2.VideoCapture(2)

x1, y1 = 70, 95 #esquerda, topo 
x2, y2 = 220, 400 #direita, baixo

while True:
    ret, frame = cap.read()

    if not ret:
        break

    crop = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Crop", crop)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
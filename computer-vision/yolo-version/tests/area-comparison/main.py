import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

background = cv2.imread("base.jpg")
if background is None:
    print("Erro: base.jpg não carregou")
    exit()

background = cv2.resize(background, (640, 480))
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# azul escuro (BGR)
color = (139, 0, 0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # diferença com o fundo
    diff = cv2.absdiff(background, gray)

    # binarização
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # limpeza de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_count = 0
    pedestre_count = 0

    for c in contours:
        area = cv2.contourArea(c)

        # ignora ruído
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # pedestre
        if 300 <= area <= 2000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "PEDESTRE", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            pedestre_count += 1

        # carro
        elif 2000 < area < 5000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "CARRO", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            car_count += 1

    cv2.putText(frame, f"Carros: {car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Pedestres: {pedestre_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mascara", clean)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

background = cv2.imread("base.jpg")
if background is None:
    print("Erro: base.jpg não carregou")
    exit()

background = cv2.resize(background, (640, 480))
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # diferença com fundo
    diff = cv2.absdiff(background, gray)

    # binariza
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # limpeza de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)

        # ignora ruído pequeno
        if area < 1200:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # desenha bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # mostra área
        cv2.putText(frame, f"{int(area)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mascara", clean)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

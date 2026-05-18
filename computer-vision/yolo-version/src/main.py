from ultralytics import YOLO
import cv2

# Modelo leve e rápido
model = YOLO("yolov8s.pt")

# Webcam
cap = cv2.VideoCapture(2)

# ! essa qualidade aqui ta pessima, muito pesada, mas qualidades mais baixas prejudicam a identificação
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Classes de veículos
VEHICLE_CLASSES = [2, 3, 5, 7]

while True:
    success, frame = cap.read()

    if not success:
        print("Erro ao acessar câmera")
        break

    # Inferência balanceada
    results = model(frame, imgsz=640)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filtra veículos
            if cls in VEHICLE_CLASSES and confidence > 0.25: #? tive que baixar porque ele nao identifica direito, talvez seja porque não está na situação certa (rua)

                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{model.names[cls]} {confidence:.2f}"

                # Caixa
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Texto
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Contador
    cv2.putText(
        frame,
        f"Veiculos: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.imshow("YOLO - Veiculos", frame)

    # Sai com Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera câmera
cap.release()
cv2.destroyAllWindows()
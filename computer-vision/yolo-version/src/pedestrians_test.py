from ultralytics import YOLO
import cv2

from area_crop_pedestrian import crop_pedestrian_area

# Modelo YOLO um pouco maior que o atual da main
#! identificação ta meio ruim
model = YOLO("yolov8s.pt")
#TODO: base de dados.. o modelo não identifica pessoas de costas somente de frente:(

# Webcam
cap = cv2.VideoCapture(2)

# Classe pessoa no COCO
PERSON_CLASS = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Erro ao acessar câmera")
        break

    # Crop da área de pedestres
    pedestrian_frame = crop_pedestrian_area(frame)

    # YOLO
    results = model(pedestrian_frame, imgsz=960)

    person_count = 0

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            # Detecta pessoas
            if cls == PERSON_CLASS and confidence > 0:

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"Pessoa {confidence:.2f}"

                # Caixa
                cv2.rectangle(
                    pedestrian_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                # Texto
                cv2.putText(
                    pedestrian_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

    # Contador
    cv2.putText(
        pedestrian_frame,
        f"Pessoas: {person_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.imshow(
        "YOLO - Pedestres",
        pedestrian_frame
    )

    # Sai com Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
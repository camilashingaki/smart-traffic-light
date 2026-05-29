import cv2
import os
import time

from area_crop_pedestrian import crop_pedestrian_area

# Webcam
cap = cv2.VideoCapture(2)

# Pasta das imagens
output_folder = "dataset_images"

# Cria pasta se não existir
os.makedirs(output_folder, exist_ok=True)

image_count = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Erro ao acessar câmera")
        break

    # Crop da área de pedestres
    pedestrian_frame = crop_pedestrian_area(frame)

    # Nome da imagem
    image_name = f"pedestrian_{image_count}.jpg"

    # Caminho completo
    image_path = os.path.join(
        output_folder,
        image_name
    )

    # Salva imagem cropada
    cv2.imwrite(
        image_path,
        pedestrian_frame
    )

    print(f"Imagem salva: {image_name}")

    # Mostra preview
    cv2.imshow(
        "Dataset Pedestres",
        pedestrian_frame
    )

    image_count += 1

    # Sai com Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Espera 30 segundos
    time.sleep(30)

cap.release()
cv2.destroyAllWindows()
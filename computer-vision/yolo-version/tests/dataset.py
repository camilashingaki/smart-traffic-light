import cv2
import os

from area_crop_pedestrian import crop_pedestrian_area

# Webcam
cap = cv2.VideoCapture(0)

# Pasta das imagens
output_folder = "dataset_images2"

# Cria pasta
os.makedirs(output_folder, exist_ok=True)

image_count = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Erro ao acessar câmera")
        break

    # Crop
    pedestrian_frame = crop_pedestrian_area(frame)

    # Preview
    cv2.imshow(
        "Dataset Pedestres",
        pedestrian_frame
    )

    key = cv2.waitKey(1)

    # Tira foto com SPACE
    if key == 32:

        image_name = f"pedestrian_{image_count}.jpg"

        image_path = os.path.join(
            output_folder,
            image_name
        )

        cv2.imwrite(
            image_path,
            pedestrian_frame
        )

        print(f"Imagem salva: {image_name}")

        image_count += 1

    # Sai com Q
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
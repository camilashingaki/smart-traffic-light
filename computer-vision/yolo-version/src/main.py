"""
Visão Computacional Unificada com Stream Web — Semáforo Inteligente

Detecta veículos e pedestres a partir de UMA câmera (dois recortes da
mesma imagem), conta cada grupo na sua área e transmite o resultado
como uma página web acessível por qualquer dispositivo na mesma rede
wifi (ex: tablet).

- Veículos: YOLOv8m pré-treinado (COCO) na área de tráfego
- Pedestres: modelo fine-tuned (452 fotos) na área de pedestres

Como usar:
    python unified_vision_web.py

Depois, no tablet (mesma rede wifi), abra no navegador:
    http://IP_DO_RASPBERRY:8000

Para descobrir o IP do Raspberry:  hostname -I
"""

from __future__ import annotations

import time
import json
import threading

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

from area_crop_vehicle import crop_traffic_area
from area_crop_pedestrian import crop_pedestrian_area

# ── Configurações ─────────────────────────────────────────────────────────────

MODEL_VEHICLES    = "yolov8m.pt"
MODEL_PEDESTRIANS = "../models/pedestrians_v2_452.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)

# Índice da câmera:
#   - No Ubuntu de desenvolvimento costuma ser 2
#   - No Raspberry Pi com câmera USB costuma ser 0
# Ajuste aqui conforme o dispositivo.
CAMERA_INDEX = 2

CONF_VEHICLES    = 0.3
CONF_PEDESTRIANS = 0.3

TICK_SECONDS = 5
OUTPUT_JSON  = "traffic_data.json"

# Porta do servidor web
WEB_PORT = 8000

# ── Cores (BGR) ───────────────────────────────────────────────────────────────
COR_VEICULO   = (90, 220, 90)     # verde
COR_PEDESTRE  = (245, 150, 60)    # azul-claro
COR_PAINEL    = (28, 28, 38)
COR_TEXTO     = (235, 235, 245)
COR_TEXTO_DIM = (150, 150, 170)
COR_BORDA     = (70, 70, 95)
COR_AREA_VEH  = (90, 220, 90)
COR_AREA_PED  = (245, 150, 60)

FONTE    = cv2.FONT_HERSHEY_DUPLEX
FONTE_SM = cv2.FONT_HERSHEY_SIMPLEX

# ── Estado compartilhado entre threads ────────────────────────────────────────
frame_lock   = threading.Lock()
latest_frame = None   # último frame processado (JPEG bytes)
contagens    = {"veh_ns": 0, "ped_total": 0}


# ── Coordenadas dos crops (para desenhar as áreas no frame completo) ──────────
# Espelham as funções crop_traffic_area e crop_pedestrian_area
AREA_VEH = (80, 150, 640, 650)    # x1, y1, x2, y2
AREA_PED = (70, 95, 220, 400)


# ── Detecção ──────────────────────────────────────────────────────────────────

def detectar_veiculos(frame, model_veh):
    cropped = crop_traffic_area(frame)
    results = model_veh(cropped, imgsz=640, verbose=False)
    count, boxes = 0, []
    ox, oy = AREA_VEH[0], AREA_VEH[1]   # offset para mapear de volta ao frame
    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in VEHICLE_CLASSES and conf > CONF_VEHICLES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return count, boxes


def detectar_pedestres(frame, model_ped):
    cropped = crop_pedestrian_area(frame)
    results = model_ped(cropped, imgsz=640, verbose=False)
    count, boxes = 0, []
    ox, oy = AREA_PED[0], AREA_PED[1]
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > CONF_PEDESTRIANS:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return count, boxes


# ── Desenho ───────────────────────────────────────────────────────────────────

def desenhar_caixas(img, boxes, cor):
    """Desenha apenas as caixas, sem rótulo nem confiança."""
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)


def desenhar_area(img, area, cor):
    """Desenha o contorno da região de detecção com linha tracejada sutil."""
    x1, y1, x2, y2 = area
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 1)


def painel_inferior(img, veh_ns, ped_total):
    """
    Desenha os contadores de veículos e pedestres lado a lado,
    na parte INFERIOR da tela, no mesmo estilo.
    """
    h, w = img.shape[:2]
    ph = 90                      # altura do painel
    y0 = h - ph

    # Fundo translúcido cobrindo a largura inferior
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), COR_PAINEL, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.line(img, (0, y0), (w, y0), COR_BORDA, 1)

    def bloco(x_centro, titulo, valor, cor):
        # Barra colorida acima do título
        cv2.rectangle(img, (x_centro - 90, y0 + 16),
                      (x_centro - 86, y0 + 70), cor, -1)
        # Título
        cv2.putText(img, titulo, (x_centro - 78, y0 + 36),
                    FONTE_SM, 0.6, COR_TEXTO_DIM, 1, cv2.LINE_AA)
        # Valor grande
        cv2.putText(img, str(valor), (x_centro - 78, y0 + 72),
                    FONTE, 1.3, cor, 2, cv2.LINE_AA)

    # Dois blocos: veículos à esquerda, pedestres à direita
    bloco(int(w * 0.28), "VEICULOS",  veh_ns,    COR_VEICULO)
    bloco(int(w * 0.72), "PEDESTRES", ped_total, COR_PEDESTRE)


# ── Saída para o ML ───────────────────────────────────────────────────────────

def salvar_saida(veh_ns, ped_total):
    ped_l = ped_total // 2
    ped_o = ped_total - ped_l
    data = {
        "veh_ns": veh_ns, "ped_l": ped_l, "ped_o": ped_o,
        "ped_total": ped_total, "timestamp": time.time(),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f)


# ── Thread de processamento de vídeo ──────────────────────────────────────────

def loop_visao():
    global latest_frame

    print("Carregando modelos...")
    model_veh = YOLO(MODEL_VEHICLES)
    model_ped = YOLO(MODEL_PEDESTRIANS)
    print("Modelos carregados.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERRO: câmera não encontrada no índice {CAMERA_INDEX}.")
        return

    print(f"\nVisão iniciada. Captura a cada {TICK_SECONDS}s.")

    ultima = 0
    veh_ns = ped_total = 0
    veh_boxes = ped_boxes = []

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.5)
            continue

        agora = time.time()
        if agora - ultima >= TICK_SECONDS:
            ultima = agora
            veh_ns, veh_boxes = detectar_veiculos(frame, model_veh)
            ped_total, ped_boxes = detectar_pedestres(frame, model_ped)
            salvar_saida(veh_ns, ped_total)
            contagens["veh_ns"]    = veh_ns
            contagens["ped_total"] = ped_total
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Veiculos: {veh_ns:2d} | Pedestres: {ped_total:2d}")

        # Monta o frame completo com as detecções
        img = frame.copy()
        desenhar_area(img, AREA_VEH, COR_AREA_VEH)
        desenhar_area(img, AREA_PED, COR_AREA_PED)
        desenhar_caixas(img, veh_boxes, COR_VEICULO)
        desenhar_caixas(img, ped_boxes, COR_PEDESTRE)
        painel_inferior(img, veh_ns, ped_total)

        # Codifica como JPEG para o stream web
        ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_frame = buffer.tobytes()

        time.sleep(0.03)   # ~30 fps de exibição


# ── Servidor web ──────────────────────────────────────────────────────────────

app = Flask(__name__)

PAGINA = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Semáforo Inteligente — Visão</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0a0a14;
      color: #e0e0f0;
      font-family: system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
    }
    h1 { font-size: 1.4rem; margin-bottom: 4px; }
    p  { color: #888; font-size: 0.9rem; margin-bottom: 20px; }
    .video-wrap {
      border: 2px solid #2a2a4a;
      border-radius: 12px;
      overflow: hidden;
      max-width: 100%;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    img { display: block; max-width: 100%; height: auto; }
  </style>
</head>
<body>
  <h1>🚦 Semáforo Inteligente</h1>
  <p>Visão computacional em tempo real</p>
  <div class="video-wrap">
    <img src="/stream" alt="Stream da câmera">
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGINA)


def gerar_stream():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/stream")
def stream():
    return Response(gerar_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Thread da visão roda em paralelo ao servidor web
    t = threading.Thread(target=loop_visao, daemon=True)
    t.start()

    print(f"\n{'='*50}")
    print("  Servidor web iniciado!")
    print(f"  No tablet, abra: http://IP_DO_RASPBERRY:{WEB_PORT}")
    print(f"  Descubra o IP com: hostname -I")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
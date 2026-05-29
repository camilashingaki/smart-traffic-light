# 🚦 Visão Computacional com YOLO

## Objetivo

Implementar um sistema de visão computacional utilizando YOLO (You Only Look Once) para:

* detectar veículos na maquete
* contar veículos automaticamente
* detectar pedestres
* futuramente controlar semáforos de forma inteligente via ESP32

---

# Tecnologias utilizadas

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Webcam USB
* Raspberry Pi (planejado)

---

# Estrutura objetivo do projeto

```text
computer-vision/
├── area-comparison/
├── yolo-version/
│   ├── src/
│   │   ├── main.py
│   │   ├── main2.py
│   │   ├── area_crop_vehicle.py
│   │   └── area_crop_pedestrian.py
│   │
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── README.md
```

---

# Primeiros testes com YOLO

Inicialmente foi utilizado o modelo:

```python
YOLO("yolov8s.pt")
```

Depois foram realizados testes com:

```python
YOLO("yolov8n.pt")
```

O modelo `n` (nano) apresentou:

* melhor desempenho
* maior FPS
* menor consumo
* maior compatibilidade futura com Raspberry Pi

---

# Detecção de veículos

## Classes utilizadas do COCO Dataset

```python
VEHICLE_CLASSES = [2, 3, 5, 7]
```

Correspondendo a:

* carro
* moto
* ônibus
* caminhão

---

# Problemas encontrados

## 1. Detecção ruim de veículos

Inicialmente o YOLO só detectava carros muito próximos da câmera.

### Causas identificadas:

* muito ruído visual
* não existia contexto da maquete (rua, tamanho)

---

# Solução aplicada: Crop da área útil e visão dentro da maquete.

Foi criado um sistema de recorte da área da maquete:

```python
def crop_vehicle_area(frame):
```

Isso permitiu:

* remover fundo desnecessário
* melhorar FPS
* melhorar precisão
* adicionar contexto para que a IA idenificasse carros

Resultado:
✅ detecção significativamente melhor (>70% de confidence no 8s).

---

# Ajustes realizados

## Resolução da câmera

Foi identificado que resoluções muito baixas prejudicavam a detecção.

A resolução padrão da webcam apresentou melhor resultado.

---

## Modelo YOLO

Testes realizados:

* yolov8m
* yolov8s
* yolov8n

Resultado:

* `yolov8n` foi o melhor equilíbrio entre velocidade e precisão

---

# Detecção de pedestres

# Problemas encontrados na detecção de pedestres

A detecção apresentou inconsistências devido a:

* miniaturas muito pequenas
* pouca textura nos bonecos
* poses não presentes no dataset COCO
* pessoas vistas de costas
* alta densidade de objetos próximos

---

# Descobertas importantes

O YOLO consegue detectar:
✅ bonecos de frente
✅ algumas poses laterais

Mas falha frequentemente:
❌ bonecos de costas
❌ grupos muito densos
❌ miniaturas pequenas demais

---

# Conclusão técnica atual

Pela maquete ser um cenário fixo e controlável, o projeto torna-se favorável para treinamento customizado.

---

# Próximo passo planejado

## Treinamento customizado do YOLO

Objetivo:
criar um modelo especializado na maquete.

---

# Plano de treinamento

## Dataset

Capturar imagens contendo:

* carros
* pedestres
* diferentes posições
* diferentes quantidades
* diferentes iluminações

---

## Anotação

Utilizar ferramentas como:

* LabelImg
* Roboflow

Para marcar:

* veículos
* pedestres

---

## Fine-tuning

Treinar o YOLO utilizando:

* imagens reais da maquete
* câmera real do projeto
* ângulo real do sistema

---

# Objetivos futuros

## Curto prazo

* melhorar estabilidade da contagem
* separar arquitetura por módulos
* melhorar FPS
* criar lógica de semáforo

---

## Médio prazo

* treinamento customizado
* comunicação Raspberry ↔ ESP32
* múltiplas faixas
* múltiplas áreas de detecção

---

## Longo prazo

* sistema completo de semáforo inteligente
* temporização adaptativa
* priorização baseada em fluxo
* integração completa embarcada na Raspberry Pi

---

# Status atual

## Veículos

✅ funcionando bem

## Pedestres

⚠️ funcionando parcialmente

## Treinamento customizado

🟡 planejado

## Integração ESP32

🟡 futura implementação

## Raspberry Pi

🟡 futura otimização

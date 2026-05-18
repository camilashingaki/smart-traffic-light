# 👀
# Sistema de Visão Computacional 🚦

## Visão geral

Este módulo do projeto tem como objetivo detectar veículos em uma maquete urbana utilizando visão computacional, fornecendo dados para um sistema de controle adaptativo de semáforos.

O sistema utiliza uma webcam USB posicionada acima da maquete para capturar imagens em tempo real. O processamento é realizado com Python e OpenCV em ambiente Linux.

A arquitetura atual do sistema é:

```text
Webcam USB
    ↓
Captura de vídeo
    ↓
Processamento OpenCV
    ↓
Detecção de veículos
    ↓
Contagem de carros
    ↓
Lógica semafórica
```

---

# Primeira abordagem

A primeira solução desenvolvida foi baseada em Background Subtraction.

A ideia consiste em capturar uma imagem da maquete vazia (`base.jpg`) e utilizá-la como referência fixa do cenário.

Cada frame da câmera é comparado com essa imagem base:

```text
Imagem base
    ↓
Frame atual
    ↓
Diferença entre imagens
    ↓
Binarização
    ↓
Limpeza de ruído
    ↓
Contornos
    ↓
Contagem de veículos
```

O sistema utiliza:

* `cv2.absdiff()` para comparação entre imagens;
* threshold binário para segmentação;
* operações morfológicas (`MORPH_OPEN` e `MORPH_CLOSE`) para estabilização da máscara;
* contornos para identificar veículos individualmente.

Essa abordagem foi escolhida inicialmente por:

* baixo custo computacional;
* simplicidade de implementação;
* funcionamento com carros parados;
* compatibilidade futura com Raspberry Pi.

---

# Problemas encontrados

Durante os testes, foram observados:

* flutuação na contagem por área;
* ruídos causados por iluminação e textura;
* instabilidade causada por pequenas diferenças entre frames.

Por isso, o sistema evoluiu de:

```text
contagem por área
```

para:

```text
contagem por contornos
```

resultando em uma detecção mais estável.

---

# Evolução futura com YOLO

Uma evolução futura prevista é a utilização do modelo YOLO (You Only Look Once).

Essa abordagem permitiria:

* reconhecimento visual real de veículos;
* detecção sem imagem base;
* maior robustez;
* classificação de objetos.

Entretanto, o uso de YOLO ainda apresenta limitações para o projeto atual:

* maior custo computacional;
* necessidade de modelos adaptados para visão top-down;
* desempenho limitado em Raspberry Pi 3.

Por esse motivo, a abordagem clássica baseada em OpenCV permanece como solução principal nesta etapa.

---

# Integração futura

A arquitetura planejada para o sistema completo é:

```text
Câmera
    ↓
Visão computacional
    ↓
Lógica adaptativa
    ↓
ESP32
    ↓
Controle físico do semáforo
```

---

# Cronograma

| Data       | Atividade                                               |
| ---------- | ------------------------------------------------------- |
| 10/05/2026 | Definição inicial da arquitetura da visão computacional |
| 10/05/2026 | Escolha do OpenCV e webcam USB                          |
| 13/05/2026 | Configuração do ambiente Python no Linux                |
| 13/05/2026 | Integração da webcam USB                                |
| 13/05/2026 | Primeiros testes de captura de vídeo                    |
| 13/05/2026 | Implementação de detecção por contornos                 |
| 13/05/2026 | Desenvolvimento da abordagem por background subtraction |
| 13/05/2026 | Implementação da contagem de veículos                   |
| Futuro     | Integração com ESP32                                    |
| Futuro     | Avaliação do uso de YOLO                                |

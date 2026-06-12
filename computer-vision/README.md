# 🚦 Sistema de Visão Computacional para Semáforos Inteligentes

## 📖 Sobre o Projeto

Este repositório contém o módulo de visão computacional desenvolvido para um sistema de semáforos inteligentes.

O objetivo do sistema é monitorar, em tempo real, o fluxo de veículos e pedestres em uma maquete urbana utilizando uma câmera posicionada acima da via. As informações obtidas são processadas por modelos YOLO e convertidas em dados de tráfego, que podem ser utilizados por uma lógica de controle semafórico adaptativo.

Atualmente, o sistema utiliza:

* Um modelo YOLO pré-treinado para detecção de veículos;
* Um modelo YOLO treinado especificamente para detecção de pedestres na maquete;
* Processamento em tempo real utilizando Python e OpenCV;
* Geração de dados de tráfego em formato JSON.

---

## ✨ Funcionalidades

* 🚗 Detecção automática de veículos
* 🚶 Detecção automática de pedestres
* 📊 Contagem de objetos em tempo real
* 🎥 Captura de vídeo por webcam
* 🤖 Treinamento de modelos YOLO personalizados
* 📄 Exportação dos dados de tráfego

---

## 🏗 Arquitetura do Sistema

```text
Webcam USB
    ↓
Captura de vídeo
    ↓
Recorte das áreas monitoradas
    ├── Veículos
    └── Pedestres
            ↓
       Modelos YOLO
            ↓
      Contagem dos objetos
            ↓
      traffic_data.json
            ↓
      Sistema Semafórico
```

---

## 📂 Estrutura do Projeto

```text
computer-vision/
│
├── README.md
│
└── yolo-version/
    │
    ├── dataset/
    │   ├── train/
    │   ├── valid/
    │   ├── test/
    │   ├── data.yaml
    │   ├── README.dataset.txt
    │   └── README.roboflow.txt
    │
    ├── models/
    │   ├── pedestrian_model.pt
    │   └── pedestrians_v2_452.pt
    │
    ├── src/
    │   ├── main.py
    │   ├── train.py
    │   ├── area_crop_vehicle.py
    │   ├── area_crop_pedestrian.py
    │   ├── traffic_data.json
    │   ├── yolov8m.pt
    │   └── yolov8n.pt
    │
    ├── tests/
    │   ├── area-comparison
    │   ├── area-crop.py
    │   ├── dataset.py
    │   ├── pedestrians_test.py
    │   └── teste_area.py
    │
    └── venv/
```

---

# 📁 Descrição das Pastas

## dataset/

Contém o conjunto de dados utilizado para treinamento, validação e testes do detector de pedestres.

### train/

Imagens e anotações utilizadas durante o treinamento do modelo.

### valid/

Imagens utilizadas durante a validação do treinamento.

### test/

Imagens utilizadas para testes finais do modelo.

### data.yaml

Arquivo de configuração utilizado pelo YOLO contendo os caminhos do dataset e as classes utilizadas.

### README.dataset.txt

Arquivo gerado durante a exportação do conjunto de dados.

### README.roboflow.txt

Informações sobre a exportação realizada pelo Roboflow.

---

## models/

Contém os modelos treinados para utilização no projeto.

### pedestrian_model.pt

Modelo principal utilizado para detecção de pedestres.

### pedestrians_v2_452.pt

Versão alternativa gerada durante o processo de treinamento.

---

## src/

Contém os arquivos principais da aplicação.

### main.py

Arquivo responsável pela execução do sistema.

Funções principais:

* captura de vídeo;
* carregamento dos modelos;
* detecção de veículos;
* detecção de pedestres;
* contagem dos objetos;
* geração dos dados de tráfego.

### area_crop_vehicle.py

Define a região da imagem utilizada para monitoramento dos veículos.

### area_crop_pedestrian.py

Define a região da imagem utilizada para monitoramento dos pedestres.

### train.py

Script utilizado para treinamento do modelo YOLO de pedestres.

### traffic_data.json

Arquivo onde são armazenadas as informações geradas pelo sistema.

### yolov8m.pt e yolov8n.pt

Modelos YOLO pré-treinados utilizados durante os testes e desenvolvimento.

---

## tests/

Contém scripts auxiliares utilizados durante o desenvolvimento e validação da solução.

### area-comparison

Comparação entre diferentes áreas monitoradas.

### area-crop.py

Testes de recorte das regiões de interesse.

### dataset.py

Ferramentas auxiliares para validação do dataset.

### pedestrians_test.py

Testes relacionados à detecção de pedestres.

### teste_area.py

Validação das áreas utilizadas pelo sistema.

---

## ⚙️ Instalação

### Pré-requisitos

* Python 3.10 ou superior
* Git
* Webcam USB
* Sistema operacional Linux ou Windows

---

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd computer-vision
```

---

### 2. Acessar o projeto

```bash
cd yolo-version
```

---

### 3. Criar um ambiente virtual

#### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

### 5. Conectar a câmera

Conecte uma webcam USB ao computador antes de iniciar o sistema.

---

### 6. Executar o sistema

```bash
cd src
python main.py
```

Após a execução, a câmera será aberta e o sistema iniciará a detecção de veículos e pedestres em tempo real.

---

## 🧠 Treinamento do Modelo

Para treinar um novo modelo de detecção de pedestres:

```bash
cd src
python train.py
```

O treinamento utiliza as configurações presentes em:

```text
dataset/data.yaml
```

Ao final do processo, novos pesos serão gerados para utilização no sistema.

---

## 🚀 Desenvolvimento

O desenvolvimento deste módulo envolveu a criação de um conjunto de dados próprio para detecção de pedestres na maquete, incluindo a coleta de imagens, rotulação manual dos exemplos e treinamento de um modelo YOLO específico para o cenário do projeto.

Após o treinamento, o detector de pedestres foi integrado ao sistema principal juntamente com a detecção de veículos, permitindo a obtenção automática de informações de tráfego em tempo real.

---

## 📅 Cronograma

| Data       | Atividade                                          |
| ---------- | -------------------------------------------------- |
| Maio/2026  | Estudos iniciais sobre visão computacional         |
| Maio/2026  | Configuração do ambiente Python                    |
| Maio/2026  | Testes iniciais de captura de vídeo                |
| Maio/2026  | Desenvolvimento dos recortes das áreas monitoradas |
| Junho/2026 | Construção do dataset de pedestres                 |
| Junho/2026 | Rotulação manual das imagens                       |
| Junho/2026 | Treinamento do modelo YOLO                         |
| Junho/2026 | Integração entre detecção de veículos e pedestres  |
| Junho/2026 | Geração automática dos dados de tráfego            |
| Atual      | Ajustes e otimizações do sistema                   |

---

## 👥 Autoria

Projeto desenvolvido como parte do sistema de Semáforos Inteligentes, com foco na aplicação de visão computacional para monitoramento de fluxo urbano em maquetes.

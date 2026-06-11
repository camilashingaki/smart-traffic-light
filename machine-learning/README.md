# Semáforo Inteligente 🚦

Sistema de semáforo adaptativo utilizando Visão Computacional, Machine Learning, ESP32 e Raspberry Pi para o trabalho de PCS

---

## Visão Geral

Este projeto tem como objetivo desenvolver um sistema de semáforo inteligente capaz de se adaptar em tempo real de acordo com o fluxo de veículos e pedestres.

O sistema utiliza visão computacional para detectar carros e pessoas, processa essas informações por meio de um modelo de machine learning e controla uma maquete física de semáforo utilizando um ESP32.

---

## Arquitetura do Sistema

```text
Câmera
   ↓
Visão Computacional (OpenCV)
   ↓
Dados do Tráfego (Arquivo json -> quantidade de pedestres e veículos)
   ↓
Modelo / Sistema de Decisão (retorna outro json -> farol vermelho ou verde)
   ↓
ESP32 (Liga efetivamente o farol para verde ou vermelho)
   ↓
Controle dos LEDs do Semáforo
```

---

## Estrutura do Projeto

```text
smart-traffic-light/
│
├── computer-vision/
├── machine-learning/
├── esp32-controller/
├── integration/
└── docs/
```

### computer-vision

Responsável por:

* detecção de veículos
* detecção de pedestres
* geração de dados

### machine-learning

Responsável por:

* tomada de decisão
* ajuste adaptativo do tempo dos semáforos (evitar ciclos fixos - toda a solução gira em torno disso)
* considerar, também, o período do dia e da semana na análise do fluxo de carros e pedestres

### esp32-controller

Responsável por:

* controle dos LEDs
* execução da lógica do semáforo
* comunicação serial
* funcionamento físico da maquete

### integration

Responsável por:

* integração entre módulos
* testes de comunicação

### docs

Documentação do projeto:

* diagramas
* arquitetura
* análise de requisitos funcionais e não funcionais
* projeto teorizado - problema e solução

---

## Objetivos

* Detectar fluxo de veículos e pedestres em tempo real
* Adaptar automaticamente o tempo dos semáforos
* Reduzir tempo de espera desnecessário
* Poder criar algo escalável

---

## Status Atual

* [ ] Estrutura inicial do projeto
* [ ] Pipeline de visão computacional
* [ ] Detecção de tráfego
* [ ] Sistema de decisão
* [ ] Comunicação com ESP32
* [ ] Integração completa

---
## Cronograma de metas 🎯

| Semana            | Metas                                                                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **09/05 → 15/05** | • Fazer semáforo básico no ESP32 ✅<br>• OpenCV contornando o que não pertence à foto base ✅<br>• Terminar cenários de teste do ML ✅<br>• Comprar/finalizar materiais da maquete ✅<br>• Organizar GitHub e pastas ✅<br>• Criar arquivo com requisitos, riscos etc ✅|
| **16/05 → 22/05** | • Detectar carros e pessoas na maquete com OpenCV ✅<br>• Fazer o outro semáforo ✅<br>• Pensar integração da eletrônica + display + microcontroladores ✅<br>• C-Montar o ML (dataset → decisão do farol) ✅                                                       |
| **23/05 → 29/05** | • Melhorar precisão da visão computacional ✅<br>• Treinar ML ✅<br>• Fazer display mostrando carros/pedestres → decisão do farol ✅<br>• Finalizar maquete 3D e definir impressão ✅                                                                              |
| **30/05 → 05/06** | • ML funcionando ✅<br>• Integrar visão computacional + ML ✅<br>• Ajustar tempos dos semáforos ✅<br>• Começar estética da maquete ✅                                                                                                                             |
| **06/06 → 12/06** | • Sistema completo funcionando ✅<br>• Finalizar maquete ✅<br>• Ensaiar apresentação ✅<br>• Finalizar README/documentação ✅                                                                                                                                     |

---

## Detalhamento Técnico — Visão Computacional

O módulo de visão computacional é responsável por traduzir a imagem da câmera em dados numéricos consumíveis pelo sistema de decisão. A cada ciclo, um frame é capturado e submetido a dois recortes espaciais independentes (`crop_traffic_area` e `crop_pedestrian_area`), que isolam a região de veículos no sentido Norte-Sul e a faixa de espera de pedestres. Essa separação espacial por coordenadas fixas reduz falsos positivos e permite a aplicação de modelos distintos para cada classe de objeto.

A detecção de veículos utiliza **YOLOv8m** com pesos pré-treinados no dataset COCO, filtrando as classes correspondentes a `car`, `motorcycle`, `bus` e `truck`. A detecção de pedestres utiliza um modelo **YOLOv8n** submetido a *fine-tuning* com uma base de **452 imagens** coletadas do setup físico real e anotadas via Roboflow — abordagem necessária porque modelos genéricos não reconhecem de forma confiável os elementos em escala de maquete. O dataset passou por *data augmentation* (flip horizontal, rotação, variação de brilho/contraste e ruído) e foi dividido em conjuntos de treino, validação e teste, com desempenho avaliado por mAP@50, *precision* e *recall*.

A saída do módulo são as contagens agregadas de veículos e pedestres, transmitidas ao sistema de decisão. Como ambos os lados da travessia de pedestres são atendidos simultaneamente, a contagem de pedestres é tratada como um único grupo para fins de decisão.

---

## Detalhamento Técnico — Machine Learning

O sistema de decisão foi desenvolvido com **Reinforcement Learning (RL)**, abordagem em que um agente aprende a política ótima de controle por meio de interações repetidas com um ambiente de simulação, sem regras programadas manualmente. O algoritmo utilizado é o **PPO (Proximal Policy Optimization)** da biblioteca Stable-Baselines3, escolhido por sua estabilidade e eficiência de amostragem em espaços de ação discretos.

### Modelagem do ambiente

A simulação foi modelada como um ambiente compatível com o padrão **Gymnasium** (`TrafficLightEnv`), com as seguintes definições formais:

* **Unidade temporal:** 1 tick = 5 segundos simulados. Um dia completo equivale a 17.280 ticks.
* **Espaço de observação:** vetor contínuo de 7 dimensões normalizado para [0, 1], contendo o tamanho das três filas lógicas (veículos N→S, pedestres leste, pedestres oeste), a fase atual do semáforo, o tempo decorrido na fase e a maior espera individual de veículos e pedestres.
* **Espaço de ação:** discreto com duas opções — manter a fase atual (0) ou solicitar a troca de fase (1). Ações de troca antes do tempo mínimo de verde são silenciosamente ignoradas pelo ambiente, sem penalização artificial.
* **Função de recompensa:** combinação ponderada de múltiplos componentes — espera acumulada, tamanho total das filas, espera máxima individual, desequilíbrio entre filas, excesso acima do teto de espera por tipo de agente e custo fixo por troca de fase. Todos os pesos são configuráveis via arquivo YAML, sem necessidade de alterar o código.

### Geração de dados

Na ausência de uma base de dados real estruturada, os cenários de tráfego foram gerados sinteticamente através de uma arquitetura de duas camadas: um perfil-base de chegadas, calibrado por combinação de tipo de dia e faixa horária, modulado por cinco famílias de comportamento (`equilibrado`, `pico_veic`, `pico_ped`, `baixa_mov` e `imprevisivel`). As chegadas são sorteadas a partir de uma **distribuição de Poisson** parametrizada pela média contextual, com controle de semente para reprodutibilidade. Foram gerados 50 cenários de treino e 32 de avaliação, cada um representando um dia completo.

### Arquitetura DualAgent

A avaliação do agente inicial revelou um comportamento assimétrico: bom desempenho na maioria dos cenários, mas degradação em situações de alto volume de veículos (família `pico_veic`), causada pelo desbalanceamento natural dos dados de treino. A solução adotada foi o **DualAgent** — uma arquitetura que combina dois modelos especializados:

* **Agente geral:** treinado com todos os cenários, eficaz em `baixa_mov`, `equilibrado` e `pico_ped`.
* **Agente especialista:** treinado exclusivamente com cenários `pico_veic`, capaz de lidar com alto fluxo de veículos.

A seleção entre os dois é feita de forma reativa ao estado atual das filas: quando a fila de veículos supera significativamente a de pedestres, o especialista assume o controle; caso contrário, o agente geral decide. Nenhum dos modelos tem conhecimento prévio do tipo de cenário — a decisão baseia-se exclusivamente no estado instantâneo das filas, exatamente como um semáforo real operaria.

---

## Validação e Resultados

O desempenho do agente foi avaliado de forma quantitativa em **32 cenários de avaliação** com sementes distintas das usadas no treino, garantindo que os resultados não refletem memorização. As métricas foram comparadas com o benchmark de semáforo de tempo fixo e validadas estatisticamente por reamostragem de **Monte Carlo** (bootstrap com 10.000 iterações) e teste t para significância.

Principais resultados do DualAgent frente ao semáforo de tempo fixo:

| Métrica | Variação |
|---|---|
| Espera média de pedestres | **−55,9%** |
| Espera média de veículos | **−25,9%** |
| Espera máxima de veículos | **−27,9%** |
| Violações de tempo limite de pedestres | **−83,5%** |
| Espera média de veículos (cenários vencidos) | **32 de 32** |

A melhoria na espera de pedestres é estatisticamente significativa (p < 0,001). A melhoria na espera de veículos, embora consistente em todos os cenários, apresenta intervalos de confiança que se sobrepõem parcialmente, indicando que a confirmação estatística plena demandaria um conjunto de avaliação mais amplo — limitação documentada como trabalho futuro.

Os resultados completos, gráficos comparativos, análise de Monte Carlo e relatório final encontram-se na pasta `machine-learning/results/`.

---

## Como Executar

### Clonar o repositório

```bash
git clone git@github.com:SEU-USUARIO/smart-traffic-light.git
```

### Entrar na pasta do projeto

```bash
cd smart-traffic-light
```

### Reproduzir o módulo de Machine Learning

A partir da pasta `machine-learning/`:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
pip install -r requirements.txt

python scripts/generate_scenarios.py    # gera cenários sintéticos
python scripts/run_benchmark.py         # roda o benchmark de tempo fixo
python scripts/train_agent.py           # treina o agente geral
python scripts/train_agent_pico_veic.py # treina o especialista
python scripts/evaluate_dual_agent.py   # avalia o DualAgent
python scripts/analyze_results.py       # gera gráficos e tabelas
```

Os modelos treinados (`.zip`) não são versionados por questões de tamanho; devem ser gerados localmente pelos scripts acima.

---

## Stack Tecnológica

| Módulo | Tecnologias |
|---|---|
| Visão Computacional | Python · OpenCV · YOLOv8 (Ultralytics) · Roboflow |
| Machine Learning | Python · Stable-Baselines3 (PPO) · Gymnasium · NumPy · Pandas · Matplotlib · SciPy |
| Controle Físico | ESP32 · Arduino (C++) · comunicação MQTT |
| Hardware | Raspberry Pi · câmera USB · maquete com LEDs |

---

## Equipe

Projeto desenvolvido para fins acadêmicos e educacionais.

---

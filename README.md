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

## Organização das Branches

O desenvolvimento de cada módulo foi conduzido em branches específicas, permitindo trabalho paralelo entre os integrantes e revisão incremental antes da integração à `main`. As branches do módulo de Machine Learning estão descritas abaixo.

**`feature/rl-training`** — Contém as Fases 4 e 5 do módulo de ML. Implementa o ambiente de Reinforcement Learning (`TrafficLightEnv`, compatível com o padrão Gymnasium), a suíte de testes de validação e o treinamento inicial do agente com o algoritmo PPO (Proximal Policy Optimization) via Stable-Baselines3. Resultado: o primeiro agente treinado, capaz de controlar o semáforo de forma adaptativa.

**`fase6-dual-agent`** — Contém a Fase 6 do módulo de ML. Adiciona a avaliação comparativa contra o benchmark de tempo fixo, a análise crítica dos resultados (incluindo validação estatística por Monte Carlo) e a arquitetura final **DualAgent**: dois modelos especializados que se alternam conforme o estado das filas, combinando um agente geral com um especialista em cenários de alto volume de veículos. Resultado: superação do semáforo de tempo fixo em cinco das seis métricas principais, com redução de 55,9% na espera média de pedestres e 25,9% na espera média de veículos.

---

## Metodologia do Módulo de Machine Learning

O sistema de decisão foi desenvolvido com **Reinforcement Learning**, abordagem em que um agente aprende a melhor política de controle por meio de interações repetidas com um ambiente de simulação, sem regras programadas manualmente. O desenvolvimento seguiu seis fases com critérios de aceite explícitos:

1. **Setup e engine de simulação** — modelagem do cruzamento, filas lógicas e fases.
2. **Visualização e benchmark** — controlador de tempo fixo que serve como base de comparação.
3. **Gerador de cenários** — criação de cenários sintéticos de tráfego com distribuição de Poisson, cobrindo diferentes horários e padrões de fluxo.
4. **Ambiente Gymnasium** — interface padronizada entre a simulação e o algoritmo de RL.
5. **Treinamento PPO** — aprendizado do agente em múltiplos ambientes paralelos.
6. **Avaliação e arquitetura final** — comparação rigorosa com o benchmark e desenvolvimento do DualAgent.

Cada decisão do agente é tomada a partir apenas do estado atual das filas — quantidade de veículos e pedestres aguardando, fase atual e tempos de espera — sem qualquer conhecimento prévio do tipo de cenário, exatamente como um semáforo real operaria.

---

## Validação e Resultados

O desempenho do agente foi avaliado de forma quantitativa em **32 cenários de avaliação** com sementes distintas das usadas no treino, garantindo que os resultados não refletem memorização. As métricas foram comparadas com o benchmark de semáforo de tempo fixo e validadas estatisticamente.

Principais resultados do DualAgent frente ao semáforo de tempo fixo:

* Espera média de pedestres: **−55,9%**
* Espera média de veículos: **−25,9%**
* Espera máxima de veículos: **−27,9%**
* Violações de tempo limite de pedestres: **−83,5%**
* Superação do benchmark na espera média de veículos em **todos os 32 cenários avaliados**

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

python scripts/generate_scenarios.py   # gera cenários sintéticos
python scripts/run_benchmark.py        # roda o benchmark de tempo fixo
python scripts/train_agent.py          # treina o agente geral
python scripts/train_agent_pico_veic.py # treina o especialista
python scripts/evaluate_dual_agent.py  # avalia o DualAgent
python scripts/analyze_results.py      # gera gráficos e tabelas
```

Os modelos treinados (`.zip`) não são versionados por questões de tamanho; devem ser gerados localmente pelos scripts acima.

---

## Equipe

Projeto desenvolvido para fins acadêmicos e educacionais.

---

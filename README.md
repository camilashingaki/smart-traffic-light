# Semáforo Inteligente

Sistema de semáforo adaptativo utilizando Visão Computacional, Machine Learning e ESP32 para o trabalho de PCS

---

## Visão Geral

Este projeto tem como objetivo desenvolver um sistema de semáforo inteligente capaz de se adaptar em tempo real de acordo com o fluxo de veículos e pedestres.

O sistema utiliza visão computacional para detectar carros e pessoas, processa essas informações por meio de um modelo de machine learning e controla uma maquete física de semáforo utilizando um ESP32.

---

## Arquitetura do Sistema

```text id="b0ptjq"
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

```text id="s7e2f8"
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

## Como Executar

### Clonar o repositório

```bash id="qr3ew3"
git clone git@github.com:SEU-USUARIO/smart-traffic-light.git
```

### Entrar na pasta do projeto

```bash id="ev1x0q"
cd smart-traffic-light
```

---

## Equipe

Projeto desenvolvido para fins acadêmicos e educacionais.

---

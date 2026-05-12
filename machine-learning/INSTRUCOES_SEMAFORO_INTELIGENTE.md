# INSTRUCOES_SEMAFORO_INTELIGENTE.md

> **Documento de instruções permanentes do projeto Semáforo Inteligente.**
> Toda nova sessão do Claude Code deve começar lendo este arquivo na íntegra.
> Mudanças de design são registradas aqui *antes* de mexer no código.

---

## 1. Objetivo do projeto

Treinar um agente de **Reinforcement Learning** capaz de controlar um semáforo de forma adaptativa, reduzindo o tempo médio e o tempo máximo de espera de carros e pedestres em comparação com um semáforo de **tempo fixo** (benchmark).

O modelo treinado será posteriormente embarcado em hardware (Raspberry Pi + câmera + Arduino) por **outro grupo / em outra etapa**. **Esta tarefa não inclui integração com hardware** — apenas geração de dados sintéticos, simulação visual, e treinamento + avaliação do agente.

**Critério de sucesso:** métricas (espera média, espera máxima, distribuição) significativamente melhores que o benchmark de tempo fixo, em base sintética suficientemente diversa.

---

## 2. Layout físico do cruzamento

- **Rua única** no sentido **Norte → Sul**, com **2 faixas paralelas** (mesmo sentido, não é mão dupla).
- **1 travessia de pedestres** no eixo **Leste–Oeste**, atravessando ambas as faixas.
- **Duas fases possíveis** de semáforo (não existe fase "tudo vermelho"):
  - **Fase A:** verde para carros (N→S) / vermelho para pedestres
  - **Fase B:** verde para pedestres (L↔O) / vermelho para carros

---

## 3. Conceitos fundamentais

### 3.1 Tick
Unidade fundamental de tempo simulado. **1 tick = 5 segundos**. Todo parâmetro temporal do projeto é expresso em ticks ou em segundos múltiplos de 5.
- 1 hora = 720 ticks
- 1 dia inteiro = 17.280 ticks

### 3.2 Filas lógicas
São **3 filas lógicas** no sistema:
- `veh_ns` — carros chegando no sentido N→S (a simulação distribui fisicamente entre as 2 faixas pela regra "faixa menos cheia").
- `ped_l` — pedestres acumulados no lado **leste** querendo atravessar para o oeste.
- `ped_o` — pedestres acumulados no lado **oeste** querendo atravessar para o leste.

Quando a Fase B abre, ambas as filas de pedestres escoam simultaneamente.

### 3.3 Fase
Estado atual do semáforo: A (verde-carros) ou B (verde-pedestres).

### 3.4 Amarelo (apenas estético)
**O amarelo NÃO tem efeito funcional.** Ele opera como verde. Quando o agente decide trocar de fase no tick T:
- Durante todo o tick T (5s), o tráfego da fase atual continua passando normalmente.
- Visualmente, os primeiros 2s do tick T mostram verde, e os últimos 3s mostram amarelo.
- A partir do tick T+1, a outra fase entra em verde.

Não há penalidade na recompensa por causa do amarelo, e a simulação não muda comportamento durante ele.

### 3.5 Episódio
Pedaço de simulação rodado de uma vez, do reset ao fim. Durante **treino**, episódios são curtos e amostrados de pontos aleatórios do dia (default: 30 minutos = 360 ticks). Durante **avaliação final**, episódios são dias inteiros completos.

### 3.6 Cenário
Um dia sintético gerado pelo gerador, salvo em CSV com 17.280 linhas (uma por tick). Cada linha registra quantas chegadas houve em cada fila lógica naquele tick. Um cenário é determinado por (tipo de dia, família, seed).

---

## 4. Decisões técnicas congeladas

### 4.1 Stack
- **Linguagem:** Python 3.10+
- **Simulação visual:** Pygame
- **Gráficos em tempo real:** Matplotlib (modo interativo)
- **RL framework:** Stable-Baselines3
- **Algoritmo:** PPO
- **Interface RL:** Gymnasium-compatible
- **Configuração:** YAML
- **Dados:** Pandas + NumPy
- **Versionamento:** Git + GitHub
- **Repositório:** https://github.com/camilashingaki/smart-traffic-light (público; trabalho do Lucca fica dentro de machine-learning/)

### 4.2 Valores temporais (defaults; todos editáveis em `configs/config.yaml`)

| Parâmetro | Valor default | Em ticks |
|---|---|---|
| Tick | 5s | — |
| Tempo mínimo de verde para carros | 15s | 3 |
| Tempo mínimo de verde para pedestres | 10s | 2 |
| Amarelo visual (dentro do último tick antes da troca) | 3s | — |
| Frequência de decisão do agente | 1 tick | 1 |
| Episódio de treino | 30 min | 360 |
| Episódio de avaliação | 1 dia | 17.280 |
| Cenários na avaliação final | 30 dias | — |

### 4.3 Tetos de espera (compõem a recompensa, configuráveis)

| Tipo | Teto |
|---|---|
| Carros | 90s |
| Pedestres | 60s |

### 4.4 Detecção
**Perfeita.** A simulação é a fonte da verdade. Não modelamos ruído de câmera nesta etapa.

---

## 5. Filosofia do gerador de cenários

O gerador segue a lógica de **duas camadas**, herdada de trabalho anterior do usuário e adaptada ao novo layout.

### 5.1 Camada 1 — Perfil-base por contexto
Para cada combinação **(tipo de dia × faixa horária × fila lógica)** existe uma **média esperada de chegadas** explicitamente calibrada em `configs/scenarios.yaml`.

- **Tipos de dia:** `util`, `fds`
- **Faixas horárias** (8 ao todo):
  `madrugada`, `manha_tranquila`, `pico_manha`, `meio_manha`, `pico_tarde`, `tarde`, `pico_noite`, `noite`

### 5.2 Camada 2 — Modulação por família
O perfil-base é deformado pela família do cenário. **5 famílias:**
- `equilibrado` — perfil-base puro, sem viés.
- `pico_veic` — multiplica veículos para cima, pedestres relativamente para baixo.
- `pico_ped` — inverso de `pico_veic`.
- `baixa_mov` — reduz movimentação geral.
- `imprevisivel` — preserva a média mas amplifica a variabilidade.

Cada família tem multiplicadores próprios para veículos, pedestres, e variabilidade — todos em `configs/scenarios.yaml`.

### 5.3 Sorteio final
Após perfil + família + variabilidade, cada chegada é sorteada com **distribuição Poisson** parametrizada pela média contextual final. Tudo controlado por **seed** para reprodutibilidade.

### 5.4 O que o gerador NÃO faz
- Não atualiza filas acumuladas.
- Não conhece a fase do semáforo.
- Não decide quem atravessa.
- Apenas produz "quem chegou em cada tick".

### 5.5 Saída
Cada cenário gera dois arquivos em `scenarios/`:
- **CSV** com colunas: `tick, hora, minuto, day_type, time_band, family, veh_ns, ped_l, ped_o`
- **JSON** com metadados: seed, família, tipo de dia, duração total, versão do calibrador.

### 5.6 Versatilidade exigida
O gerador deve ser totalmente parametrizado. Trocar o número de cenários, a duração de cada um, ou as taxas de chegada deve ser feito **apenas editando YAMLs**, nunca código.

---

## 6. Especificação da recompensa do RL

A recompensa a cada tick é uma **combinação ponderada** de penalizações. Todos os pesos são editáveis em `configs/rl.yaml` para permitir rebalanceamento durante experimentação.

```yaml
reward_weights:
  espera_acumulada: -1.0      # soma dos tempos de espera de todos na fila
  tamanho_filas:    -0.5      # tamanho total das filas neste tick
  max_espera:       -2.0      # penalidade extra para a maior espera atual
  desequilibrio:    -0.3      # |carga_veiculos - carga_pedestres| neste tick
  excedeu_teto:    -10.0      # penalidade por agente (carro/pedestre) acima do teto

teto_espera_carros:    90     # segundos
teto_espera_pedestres: 60     # segundos
```

**Regra de segurança operacional:** restrições duras (ex: trocar fase antes do tempo mínimo de verde) são **impostas pelo ambiente**, não pela recompensa. Ações inválidas são **silenciosamente ignoradas** pelo ambiente. O agente nunca recebe penalidade por tentar uma ação inválida — ele simplesmente vê que o estado não mudou. Isso evita que o agente aprenda a "evitar uma punição artificial" em vez de aprender a estratégia ótima.

---

## 7. Plano de fases

O projeto é dividido em **6 fases**. Cada fase tem um **critério de aceite** explícito.

> **REGRA CRÍTICA:** o Claude Code não avança para a próxima fase sem aprovação explícita do usuário, mesmo que o critério pareça atendido.

### Fase 1 — Setup e esqueleto
- Estrutura de pastas (ver §8) criada.
- Repositório git inicializado e conectado ao GitHub remote.
- `configs/config.yaml`, `configs/scenarios.yaml`, `configs/rl.yaml` iniciais.
- Lógica básica de simulação (filas, fases, ticks) sem visualização.
- Testes unitários básicos das primitivas.
- **Aceite:** simulação roda no terminal por N ticks com prints de estado e filas evoluindo coerentemente.

### Fase 2 — Visualização Pygame + benchmark de tempo fixo
- Janela Pygame com cruzamento, carros (em 2 faixas), pedestres, semáforo (verde/amarelo/vermelho).
- Painel matplotlib lateral com gráficos de filas em tempo real.
- Controlador de tempo fixo parametrizável.
- **Aceite:** assistir uma simulação visual completa com semáforo trocando em intervalos fixos e veículos/pedestres se acumulando coerentemente.

### Fase 3 — Gerador de cenários completo
- Implementação das duas camadas (perfil + família).
- Calibração inicial em `scenarios.yaml`.
- Geração de N cenários cobrindo todas as combinações `(tipo de dia × família)`.
- Validação visual: gráfico de chegadas por hora confirmando que `pico_manha` realmente concentra mais carros, etc.
- Rodar benchmark de tempo fixo em todos os cenários e salvar tabela de métricas baseline em `results/`.
- **Aceite:** base de cenários gerada + tabela de métricas do benchmark.

### Fase 3.5 — Checkpoint pré-ML

Pausa obrigatória antes do RL. Não tem entregas técnicas; é uma parada estruturada para garantir que o usuário esteja confortável com tudo que veio antes do machine learning começar.

- Claude Code apresenta um relatório consolidado: estado da engine, comportamento do gerador, métricas do benchmark de tempo fixo, gráficos comparativos das 5 famílias de cenário em diferentes faixas horárias.
- Usuário roda a simulação visual em pelo menos 3 cenários distintos (ex: `pico_manha` útil, `baixa_mov` fds, `imprevisivel`) e valida visualmente que tudo faz sentido.
- Usuário levanta dúvidas conceituais sobre RL, recompensa, ambiente Gymnasium, ou qualquer coisa do plano que ainda esteja confusa.
- **Aceite:** aprovação explícita do usuário com a frase `Fase 3.5 aprovada, pode iniciar Fase 4`. Sem essa frase exata, o Claude Code não inicia a Fase 4.

### Fase 4 — Ambiente Gymnasium
- Envolver a simulação em `gym.Env`.
- Definir espaço de observação, espaço de ação (binário: manter/trocar), função de recompensa.
- Testar com agente aleatório.
- **Aceite:** `env.reset()` e `env.step()` funcionam sem erro; recompensa varia coerentemente; smoke test com agente aleatório completa um episódio.

### Fase 5 — Treinamento PPO
- Configuração do PPO via Stable-Baselines3.
- Treinamento (pode rodar por horas).
- Logs em TensorBoard.
- Checkpoints periódicos em `models/`.
- **Aceite:** curva de recompensa convergindo, agente treinado salvo.

### Fase 6 — Avaliação e comparação
- Rodar agente treinado em conjunto de avaliação (default: 30 dias com seeds diferentes, todas as combinações família × tipo de dia).
- Rodar benchmark de tempo fixo no mesmo conjunto.
- Gerar gráficos comparativos: espera média, espera máxima, distribuição, comportamento ao longo do dia.
- Relatório final em `results/relatorio.md`.
- **Aceite:** evidência clara e quantificada de melhora sobre o benchmark.

---

## 8. Estrutura de pastas

> Toda a estrutura abaixo vive dentro de **`machine-learning/`** no repositório do grupo (`camilashingaki/smart-traffic-light`). As pastas `computer-vision/` e `esp32-controller/` na raiz do repositório são responsabilidade de outros membros do grupo e **não devem ser tocadas**.

```
semaforo-inteligente/
├── INSTRUCOES_SEMAFORO_INTELIGENTE.md   ← este arquivo (raiz)
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── config.yaml                      # parâmetros gerais do sistema
│   ├── scenarios.yaml                   # calibração do gerador
│   └── rl.yaml                          # hiperparâmetros e pesos da recompensa
├── src/
│   ├── __init__.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── crossing.py                  # cruzamento, filas, fases
│   │   ├── controllers.py               # controlador de tempo fixo
│   │   ├── metrics.py                   # cálculo de métricas
│   │   └── simulation_loop.py           # orquestra avanço de ticks, controla velocidade/pausa, conecta engine a renderer e plots sem que esses se conheçam
│   ├── generator/
│   │   ├── __init__.py
│   │   └── scenario_generator.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── pygame_renderer.py
│   │   └── live_plots.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── env.py                       # ambiente Gymnasium
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       └── config_loader.py
├── scripts/
│   ├── run_benchmark.py
│   ├── generate_scenarios.py
│   ├── run_visual_demo.py               # demo visual Fase 2 (Pygame + matplotlib)
│   ├── train_agent.py
│   └── evaluate_agent.py
├── scenarios/                           # CSVs gerados (gitignored)
├── models/                              # checkpoints (gitignored)
├── results/                             # gráficos, tabelas, relatório
└── tests/
    └── test_*.py
```

---

## 9. Convenções de código

- **Idioma:** identificadores em **inglês** (variáveis, funções, classes), docstrings e comentários em **português**.
- **Type hints obrigatórios** em assinaturas de funções públicas.
- **Sem números mágicos** no código — todo parâmetro vem de YAML.
- **Logging estruturado:** usar `logging`, não `print` (exceto em scripts de demo curtos).
- **Reprodutibilidade:** toda função que use aleatoriedade aceita parâmetro `seed`.
- **Funções pequenas e nomeadas claramente** (limite informal: ~50 linhas).
- **Docstring no formato:** breve descrição + parâmetros + retorno.

---

## 10. Convenções de processo (Claude Code)

- **Faseamento estrito:** não começar uma fase nova sem o usuário aprovar a anterior.
- **Validação visual antes de avançar:** sempre que houver visualização possível, mostrar para o usuário antes de pedir aceite.
- **Commits semânticos** em português, descrevendo a mudança (ex: `feat: gerador aplica modulação por família`).
- **Mudanças de design vão neste documento PRIMEIRO**, código depois.
- **Quando em dúvida, perguntar ao usuário** — não inventar requisitos.
- **Confirmação explícita** antes de operações destrutivas (deletar arquivos, sobrescrever cenários, etc).

---

## 11. O que NÃO fazer

- Não implementar integração com câmera real, OpenCV, Raspberry Pi, Arduino ou qualquer hardware.
- Não usar bibliotecas pesadas desnecessariamente (ex: TensorFlow se SB3 com PyTorch já resolve).
- Não criar arquivos fora da estrutura definida sem documentar aqui antes.
- Não pular a fase 3 (benchmark) — sem ele, não há comparação.
- Não tratar o amarelo como pausa funcional — durante o amarelo o tráfego ainda passa.
- Não hard-codar valores temporais ou pesos da recompensa — tudo via YAML.
- Não puxar dependências sem atualizar `requirements.txt`.

---

## 12. Histórico de revisões deste documento

| Data | Mudança |
|---|---|
| Versão inicial | Documento criado com decisões fechadas das fases de planejamento. |
| 2026-05-09 | Fase 1 concluída. Migração para repositório do grupo (camilashingaki/smart-traffic-light, subpasta machine-learning/). Adicionada Fase 3.5 — Checkpoint pré-ML. |
| 2026-05-12 | Fase 1 aprovada pelo usuário. Critério de aceite validado em dois testes: (a) demo com taxa 2.5 veh/tick mostrando filas crescentes como esperado em cenário sobrealimentado; (b) demo com taxa 1.0 veh/tick mostrando filas estabilizadas e espera máxima de carros convergindo para ~35s, coerente com o ciclo do controlador fixo. |
| 2026-05-12 | Início da Fase 2. Adicionado `src/simulation/simulation_loop.py` à estrutura de pastas (orquestra ticks, velocidade/pausa, desacopla engine de renderer e plots). Adicionado `scripts/run_visual_demo.py`. |

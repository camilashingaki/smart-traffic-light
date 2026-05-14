# Checkpoint Pré-ML — Fase 3.5

> Documento de revisão consolidada antes do início da Fase 4 (Ambiente Gymnasium).
> Gerado em 2026-05-14. Não contém entregas técnicas — é uma pausa estruturada de alinhamento.

---

## 1. Estado do projeto

As Fases 1, 2 e 3 estão concluídas e aprovadas pelo usuário.

**Fase 1** entregou a engine de simulação do cruzamento: as três filas lógicas (`veh_ns`, `ped_l`, `ped_o`), o modelo de ticks de 5 segundos, as duas fases do semáforo (A = verde carros, B = verde pedestres), o controlador de tempo fixo e a camada de métricas. Critério de aceite validado com demo interativo mostrando filas crescentes e estabilizadas sob diferentes taxas de chegada.

**Fase 2** entregou a visualização Pygame e o loop de simulação desacoplado: HUD com contadores numéricos de fila, indicadores piscantes de chegada (+N) e escoamento (−N), controles de velocidade 1×/5×/10×/50×, pausa e reset. Também entregou gráficos ao vivo via Matplotlib (Agg). Aprovada após rodada visual do usuário.

**Fase 3** entregou o gerador de cenários sintéticos de duas camadas, os 82 cenários (50 treino + 32 avaliação), o registro configurável de 13 métricas, o benchmark do controlador de tempo fixo e as validações visuais A/B/C. Detalhes nas seções seguintes.

**Decisões arquiteturais fechadas:**
- Tick = 5 segundos. Dia simulado = 17.280 ticks.
- O semáforo tem apenas duas fases; o amarelo é visual sem efeito funcional.
- Todos os parâmetros operacionais vivem em YAML; nada é hard-coded.
- O agente RL recebe o estado como dict e emite ação binária (0 = manter, 1 = trocar fase).
- Recompensa é negativa e depende de espera acumulada, tamanho de fila, máxima espera, desequilíbrio entre carros/pedestres e penalidade por violação de teto.

**Estrutura de pastas:**

```
configs/
├── config.yaml          # parâmetros de simulação, geração, métricas, thresholds
├── rl.yaml              # pesos da recompensa e hiperparâmetros do PPO
└── scenarios.yaml       # taxas base calibradas (SP 2023) e multiplicadores de família

scripts/
├── demo_terminal.py     # demo sem Pygame (terminal)
├── evaluate_agent.py    # avalia agente treinado nos cenários de avaliação
├── generate_scenarios.py
├── run_benchmark.py     # benchmark do controlador fixo em todos os cenários
├── run_simulation_demo.py
├── run_visual_demo.py   # demo Pygame interativo
├── train_agent.py       # treino PPO via Stable-Baselines3
└── validate_scenarios.py

src/
├── generator/
│   └── scenario_generator.py
├── rl/
│   ├── env.py           # ambiente Gymnasium (esqueleto)
│   ├── evaluate.py
│   └── train.py
├── simulation/
│   ├── controllers.py   # FixedTimeController
│   ├── crossing.py      # Crossing + TrafficQueue
│   ├── metrics.py       # registry de 13 métricas + compute_tick_reward
│   └── simulation_loop.py
├── utils/
│   └── config_loader.py
└── visualization/
    ├── live_plots.py
    └── pygame_renderer.py

tests/
├── test_config_loader.py
├── test_controllers.py
├── test_crossing.py
├── test_metrics.py
├── test_scenario_generator.py
└── test_visualization.py
```

---

## 2. Engine de simulação

O núcleo da simulação é a classe `Crossing` em `src/simulation/crossing.py`. Ela gerencia três filas lógicas independentes — `veh_ns` (veículos norte-sul), `ped_l` (pedestres lado leste) e `ped_o` (pedestres lado oeste) — implementadas pela classe `TrafficQueue`. Cada fila mantém um `deque` de inteiros, onde cada elemento representa o número de ticks que a entidade correspondente já aguarda. Isso permite calcular o tempo máximo de espera, o tempo total acumulado e o throughput servido por tick com custo O(1) nas consultas.

A simulação avança um tick por chamada ao método `step(arrivals, action)`. A ordem das operações é determinística: primeiro aplica uma troca de fase pendente do tick anterior; depois registra a nova ação do controlador (se a troca for válida dado o tempo mínimo de verde); em seguida adiciona as chegadas às filas; escoa entidades conforme a capacidade da fase ativa; e por fim incrementa o tempo de espera de quem ficou. Esse sequenciamento garante que uma troca de fase solicitada no tick T só surte efeito no tick T+1, evitando que o agente aprenda a explorar artefatos de ordenação.

O escoamento de veículos usa drenagem fracionada com crédito acumulado por faixa. Cada faixa acumula `saturation_flow_veh_per_lane_per_tick = 1.5` carros por tick e drena `floor(crédito)` entidades, mantendo o resíduo para o próximo tick. Com duas faixas, a capacidade efetiva alterna entre 2 e 4 carros por tick em ticks consecutivos, resultando em média de 3 carros/tick — mais realista do que uma capacidade inteira fixa. O crédito é zerado ao trocar de fase. Para pedestres, a capacidade é inteira: `saturation_flow_ped_per_side_per_tick = 4` pedestres/tick por lado quando a Fase B está ativa.

Cada fase tem um tempo mínimo de verde configurável (`min_green_cars_ticks = 3` → 15 s; `min_green_pedestrians_ticks = 2` → 10 s). Uma solicitação de troca chegada antes desse mínimo é silenciosamente descartada — o agente RL precisa aprender que ações prematuras não têm efeito. O estado retornado por `get_state()` inclui a fase atual, os ticks acumulados nessa fase, o flag de troca pendente e, para cada fila, o tamanho, o maior tempo de espera e a soma total dos tempos de espera. O campo `served_this_tick` expõe a lista de tempos de espera (em ticks) de cada entidade servida naquele tick, necessário para o cálculo de p95 e contagem de violações de teto.

---

## 3. Gerador de cenários

O gerador implementa duas camadas de modulação sobre um perfil-base.

**Camada 1 — Perfil-base:** as taxas de chegada por tick são definidas em `configs/scenarios.yaml` para cada combinação de tipo de dia (`util` ou `fds`) e faixa horária. Há oito faixas horárias calibradas com referências de volumes viários de São Paulo (2023):

| Faixa | Horário | Características |
|---|---|---|
| madrugada | 0h–5h | Tráfego residual |
| manha_tranquila | 5h–7h | Aquecimento pré-pico |
| pico_manha | 7h–10h | Pico AM (maior taxa de veículos útil: 1.67/tick ≈ 1202/h) |
| meio_manha | 10h–12h | Vale entre picos |
| pico_tarde | 12h–14h | Almoço — pico de pedestres |
| tarde | 14h–17h | Fluxo sustentado |
| pico_noite | 17h–20h | Pico PM (maior taxa de veículos útil: 1.81/tick ≈ 1303/h) |
| noite | 20h–24h | Declínio; FDS tem mais tráfego noturno que útil |

**Camada 2 — Família:** multiplicadores `veh_multiplier` e `ped_multiplier` escalam as taxas-base. Um ruído gaussiano `N(0, 0.1 × var_multiplier)` é aplicado antes do sorteio Poisson, introduzindo variabilidade realista tick a tick.

| Família | veh_mult | ped_mult | var_mult | Uso |
|---|---|---|---|---|
| equilibrado | 1.0 | 1.0 | 1.0 | Referência |
| pico_veic | 1.5 | 0.7 | 1.0 | Alta demanda de carros |
| pico_ped | 0.7 | 1.6 | 1.0 | Alta demanda de pedestres |
| baixa_mov | 0.4 | 0.4 | 1.0 | Baixo movimento |
| imprevisivel | 1.0 | 1.0 | 2.5 | Alta variabilidade (só treino) |

A reprodutibilidade é garantida por `np.random.default_rng(seed)` instanciado uma única vez por cenário — todos os sorteios dos 17.280 ticks × 3 filas derivam do mesmo gerador em ordem determinística, produzindo CSVs byte-idênticos para a mesma tripla `(family, day_type, seed)`. Isso foi validado por hash SHA-256 em três combinações distintas (Validação C: PASS).

**Resultado:** 82 cenários gerados, armazenados em `scenarios/train/` (50) e `scenarios/eval/` (32). A família `imprevisivel` é excluída da avaliação — reservada como stress test durante o treino.

**Gráficos de validação** em `results/scenario_validation/`:
- `validation_A_veh_ns.png`, `validation_A_ped_l.png`, `validation_A_ped_o.png` — perfis horários família equilibrado (util vs fds). Confirmam dois picos para carros em útil (pico_noite ligeiramente mais forte), pico_tarde mais pronunciado para pedestres, e curva de FDS achatada sem picos definidos.
- `validation_B_familias_pico_manha.png` — barplot comparando as 5 famílias na faixa pico_manha/util. Multiplicadores batem matematicamente com os valores de `scenarios.yaml`.

---

## 4. Baseline quantitativo

O benchmark executa o `FixedTimeController` (Fase A = 9 ticks / 45 s; Fase B = 5 ticks / 25 s) em cada um dos 82 cenários durante um dia completo (17.280 ticks). Os resultados estão em `results/benchmark_baseline.csv` e `results/benchmark_baseline.md`.

### Resumo geral (média ± std sobre 82 cenários)

| Métrica | Média | Std | Mín | Máx |
|---|---|---|---|---|
| espera_media_carros (s) | 23.7 | 42.4 | 7.0 | 161.5 |
| espera_media_pedestres (s) | 22.0 | 9.1 | 17.2 | 55.6 |
| espera_maxima_carros (s) | 139.9 | 286.2 | 35.0 | 1035.0 |
| espera_maxima_pedestres (s) | 82.8 | 74.9 | 50.0 | 425.0 |
| violacoes_teto_carros | 669.1 | 1958.6 | 0 | 7953 |
| violacoes_teto_pedestres | 656.2 | 1908.7 | 0 | 7604 |
| throughput_total_carros | 10682.5 | 4563.7 | 4282 | 18847 |
| throughput_total_pedestres | 14611.2 | 6715.0 | 5609 | 27561 |

**Total de violações de teto:** 669.1 × 82 + 656.2 × 82 ≈ **54.866 (carros) + 53.808 (pedestres) = 108.674 violações** no conjunto completo.

### Por família — leitura interpretativa

| Família | Pior dimensão | viol_teto_carros | viol_teto_pedestres | Interpretação |
|---|---|---|---|---|
| **pico_veic** | Carros | **3048 ±3255/cenário** | 0 | Controlador fixo falha estruturalmente: a alta demanda de veículos satura a Fase A antes de o ciclo terminar. Espera média de carros = **77.4 s** — próxima do teto de 90 s. |
| **pico_ped** | Pedestres | 0 | **2988 ±3159/cenário** | Espelho do pico_veic: a Fase B de 25 s é insuficiente para escoar a demanda pedestre. Espera máxima chega a 425 s em alguns cenários. |
| **imprevisivel** | Nenhuma estrutural | 0 | 1.2 ±2.1/cenário | Apesar da alta variabilidade (var_mult = 2.5), a demanda média é igual ao equilibrado. O controlador fixo lida bem com variabilidade quando a média é compatível. |
| **equilibrado** | Nenhuma | 0 | 0.6 ±1.5/cenário | Funcionamento esperado. Ciclo 45s/25s adequado para as taxas calibradas. |
| **baixa_mov** | — | **0** | **0** | Cenário trivial. Nenhuma violação. O controlador fixo é sobredimensionado para este perfil. |

**Leitura para o RL:** `pico_veic` e `pico_ped` são os casos onde um agente adaptativo tem maior oportunidade de superar o baseline — basta aprender a estender a fase saturada e encurtar a fase ociosa. `baixa_mov` não é um bom discriminador entre controladores. `imprevisivel` serve de stress test de robustez.

---

## 5. Cobertura de testes

Execução de `pytest tests/ -v` em 2026-05-14 (pós-correção do bug de interval):

| Módulo | Testes | Resultado |
|---|---|---|
| `test_config_loader.py` | 5 | 5 PASS |
| `test_controllers.py` | 5 | 5 PASS |
| `test_crossing.py` | 18 | 18 PASS |
| `test_metrics.py` | 35 | 35 PASS |
| `test_scenario_generator.py` | 34 | 34 PASS |
| `test_visualization.py` | 19 | 19 PASS |
| **Total** | **116** | **116 PASS / 0 FAIL** |

Durante a Fase 3.5 foi identificado e corrigido um bug real em `src/visualization/live_plots.py`: `_last_drawn` era inicializado como `-999`, fazendo `update_surface(interval)` sempre renderizar na primeira chamada independente do interval — inclusive quando os dados acumulados eram insuficientes. A correção muda o valor inicial para `None` e trata o caso `_last_drawn is None` explicitamente como "primeira renderização, sempre executa". Um segundo teste (`test_first_render_is_immediate`) foi adicionado para cobrir o comportamento desejável de não mostrar tela vazia ao usuário.

**Cobertura por área:**
- Engine de simulação (`crossing`, `controllers`, `metrics`): 58 testes cobrindo filas, fases, drenagem fracionada, mínimos de verde, todas as 13 métricas e cálculo de recompensa.
- Gerador de cenários: 34 testes cobrindo reprodutibilidade (SHA-256), estrutura CSV/JSON, 10 transições de faixa horária, efeito dos multiplicadores de família e isolamento de seeds.
- Visualização: 19 testes cobrindo renderer Pygame, loop de simulação e gráficos ao vivo (incluindo comportamento de interval e primeira renderização imediata).
- Infraestrutura: 5 testes de carregamento de YAML.

---

## 6. O que vem a seguir

### Fase 4 — Ambiente Gymnasium

Transformar o cruzamento em um ambiente compatível com a API `gymnasium.Env`, de modo que qualquer algoritmo de RL do ecossistema Stable-Baselines3 possa treiná-lo sem modificações. As responsabilidades da Fase 4 são:

- Implementar `TrafficEnv(gymnasium.Env)` em `src/rl/env.py`, encapsulando `Crossing` e o carregamento dos CSVs de cenário.
- Definir o espaço de observação (`observation_space`): vetor contínuo com as variáveis de estado relevantes do cruzamento (tamanhos de fila, tempos máximos de espera, fase atual, ticks na fase, ticks do dia).
- Definir o espaço de ação (`action_space`): `Discrete(2)` — manter (0) ou solicitar troca (1).
- Implementar `reset()` (sorteia um cenário de treino aleatoriamente ou determinísticamente por seed) e `step()` (avança um tick, retorna obs, reward, terminated, truncated, info).
- Conectar `compute_tick_reward()` de `metrics.py` para produzir a recompensa escalar.
- Validar com `gymnasium.utils.env_checker.check_env()`.

**Pontos de decisão ainda abertos antes de começar a Fase 4:**

1. **Espaço de observação exato:** quais variáveis incluir e se normalizar. Candidatos: `(veh_ns.size, ped_l.size, ped_o.size, veh_ns.max_wait_ticks, ped_l.max_wait_ticks, ped_o.max_wait_ticks, phase_encoded, ticks_in_phase, tick_of_day)`. A inclusão de `tick_of_day` (sinal de tempo) permite que o agente aprenda padrões horários, mas aumenta a complexidade de generalização.
2. **Política de exploração:** o PPO padrão usa entropia (`ent_coef = 0.01` em `rl.yaml`). Se o agente convergir rapidamente para nunca trocar de fase, pode ser necessário aumentar o coeficiente de entropia ou usar exploração curricular.
3. **Hiperparâmetros iniciais do PPO:** os valores em `rl.yaml` são razoáveis (lr = 3e-4, γ = 0.99, n_steps = 2048), mas ainda não foram validados empiricamente para este ambiente. O horizonte de 360 ticks por episódio de treino (30 min simulados) precisa ser verificado contra o `n_steps`.

### Fase 5 — Treino PPO

Treinar o agente com Stable-Baselines3 PPO sobre os 50 cenários de treino. O loop em `scripts/train_agent.py` usará `n_envs = 4` ambientes paralelos (VecEnv), cada um sorteando cenários independentemente. Checkpoints a cada 50.000 passos, mantendo os últimos 5.

### Fase 6 — Avaliação comparativa

Executar o agente treinado nos 32 cenários de avaliação (sem `imprevisivel`) e comparar as 13 métricas contra o baseline do controlador fixo (`results/benchmark_baseline.csv`). O critério de sucesso é redução de violações de teto em `pico_veic` e `pico_ped` sem degradação em `equilibrado` e `baixa_mov`.

---

## 7. Riscos e questões em aberto

**Risco 1 — Política trivial de "nunca trocar".**
O agente pode aprender que a ação 0 (manter) é sempre segura, pois a recompensa piora abruptamente quando há troca prematura (viola mínimo de verde sem efeito). Mitigação: verificar entropia da política nas primeiras iterações; se colapsar para ação única, aumentar `ent_coef` (de 0.01 para 0.05–0.1) ou usar curriculum — começar com cenários `baixa_mov` onde trocas são baratas.

**Risco 2 — Recompensa mal balanceada entre carros e pedestres.**
Os pesos atuais (`espera_acumulada: -1.0`, `max_espera: -2.0`, `desequilibrio: -0.3`) não têm calibração empírica. Em `pico_veic`, a penalidade de fila de carros vai dominar a recompensa e o agente pode sacrificar pedestres sistematicamente. Mitigação: monitorar a razão de violações carros/pedestres durante o treino e ajustar os pesos via grid search sobre `rl.yaml`.

**Risco 3 — Overfitting aos cenários de treino.**
Com apenas 50 cenários de treino (e seeds fixas 100–104), o agente pode memorizar padrões de chegada específicos ao invés de generalizar. Mitigação: durante o treino, randomizar a ordem dos cenários e usar `imprevisivel` como ruído adicional; avaliar nos 32 cenários de avaliação (seeds diferentes: 1000–1003) para detectar overfitting.

**Risco 4 — Instabilidade do treino PPO com horizonte curto.**
O episódio de treino é de 360 ticks (30 min), mas os padrões horários que diferenciam as faixas só ficam visíveis ao longo de horas. O agente pode ter dificuldade em aprender comportamentos de longo prazo. Mitigação: experimentar aumentar `episode_ticks` do treino gradualmente (360 → 1080 → 4320), monitorando o tempo de convergência.

**Risco 5 — Escala de observação não normalizada.**
`veh_ns.size` pode variar de 0 a dezenas (em `pico_veic` severo), enquanto `phase_encoded` é 0 ou 1. Redes neurais do PPO são sensíveis à escala dos inputs. Mitigação: normalizar todas as observações para [0, 1] ou z-score no `reset()`/`step()` do ambiente, usando limites conservadores definidos em `config.yaml`.

# Handoff — Fases 4 e 5 (Ambiente RL + Treinamento PPO)

**De:** Lucca (responsável pelas Fases 1–3.5)  
**Para:** Leticia  
**Data:** 14 de maio de 2026  
**Branch de trabalho:** `feature/rl-training`

---

## Parte 1 — Setup em 5 minutos

### 1.1 Clonar e preparar

```bash
# Se ainda não clonou o repositório do grupo:
git clone https://github.com/camilashingaki/smart-traffic-light.git
cd smart-traffic-light/machine-learning

# Criar e entrar na branch de trabalho (NUNCA empurre direto na main)
git checkout -b feature/rl-training
```

### 1.2 Criar e ativar o ambiente virtual

```bash
# Python 3.11+ recomendado (evita problema com anotações de tipo; 3.10 também funciona)
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### 1.3 Instalar dependências e verificar

```bash
pip install -r requirements.txt
pytest                          # esperado: 116 passed
python scripts/run_visual_demo.py   # abre janela Pygame com simulação visual
```

Se o pytest mostrar 116/116 verdes e a janela do demo abrir sem erros, o ambiente está 100% funcional.

### 1.4 Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `from __future__ import annotations` quebrando em runtime | Python < 3.10 | Usar Python 3.10+ |
| `pygame.error: No available video device` | Ambiente sem display (SSH sem X11) | Rodar o demo localmente, não em servidor remoto |
| `ModuleNotFoundError: gymnasium` | Venv não ativado ou install incompleto | Ativar venv e rodar `pip install -r requirements.txt` de novo |
| `torch` demorando muito para instalar | Baixando build com CUDA | Usar `pip install torch --index-url https://download.pytorch.org/whl/cpu` se não tiver GPU |
| Testes falhando em `test_visualization.py` | Pygame sem display | Normal em CI; localmente deve passar |

---

## Parte 2 — O que você está recebendo

### 2.1 Status do projeto

| Fase | Descrição | Status |
|---|---|---|
| **1** | Setup, esqueleto, primitivas de simulação | Concluída |
| **2** | Visualização Pygame + benchmark de tempo fixo | Concluída |
| **3** | Gerador de cenários completo + calibração | Concluída |
| **3.5** | Checkpoint pré-ML (validação e revisão) | Concluída e aprovada |
| **4** | Ambiente Gymnasium (sua tarefa) | Pendente |
| **5** | Treinamento PPO via Stable-Baselines3 (sua tarefa) | Pendente |
| **6** | Avaliação final + relatório | Pendente |

**Leitura obrigatória antes de começar:** `INSTRUCOES_SEMAFORO_INTELIGENTE.md` na raiz de `machine-learning/`. É o documento de design do projeto inteiro — recompensa, conceitos de tick/fase/episódio, critérios de aceite de cada fase. Não é longo e vai responder 80% das dúvidas que surgem.

### 2.2 Inventário do que existe

**`src/simulation/`** — Engine de simulação completa e validada.
- `crossing.py` — coração do sistema: filas lógicas (`veh_ns`, `ped_l`, `ped_o`), lógica de escoamento por fase, tempos mínimos de verde, contadores de espera por agente. É o que o ambiente Gymnasium vai envolver.
- `controllers.py` — controlador de tempo fixo (o baseline) e interface base para controladores.
- `simulation_loop.py` — loop principal que amarra crossing + controlador + métricas.
- `metrics.py` — cálculo de espera média, máxima, p95, violações de teto, throughput.

**`src/generator/`** — Gerador de cenários completo.
- `scenario_generator.py` — gera dias sintéticos em CSV via distribuição Poisson parametrizada por família + faixa horária + seed. Totalmente configurável via YAML.

**`src/visualization/`** — Demo visual funcional.
- `pygame_renderer.py` — renderizador do cruzamento (carros em 2 faixas, pedestres, semáforo com amarelo visual).
- `live_plots.py` — painel matplotlib com filas em tempo real.

**`configs/`** — Três YAMLs calibrados, não tocar sem motivo.
- `config.yaml` — parâmetros temporais (tick = 5s, mínimos de verde, duração de episódios).
- `scenarios.yaml` — médias de chegada por contexto (tipo de dia × faixa horária × fila) e multiplicadores por família.
- `rl.yaml` — pesos da função de recompensa e tetos de espera. **Este você pode (e provavelmente vai querer) ajustar durante o treino.**

**`scenarios/train/`** — 50 cenários de treino (CSV + JSON cada), cobrindo todas as combinações família × tipo de dia com seeds diferentes.

**`scenarios/eval/`** — 32 cenários de avaliação com seeds distintas do treino. São estes que definem se o agente superou o baseline.

**`results/benchmark_baseline.csv`** — Métricas do controlador de tempo fixo (Fase A = 45s, Fase B = 25s) rodando nos 82 cenários. É o número que o agente treinado precisa bater.

**`results/scenario_validation/`** — Gráficos de validação visual das curvas de chegada por família e hora, gerados durante a Fase 3. Útil para entender o comportamento dos cenários.

### 2.3 O que NÃO mexer (zona vermelha)

As seguintes partes foram validadas e calibradas. Qualquer mudança invalida a comparação com o baseline e pode quebrar os 116 testes que cobrem essas camadas:

- **`src/simulation/crossing.py`** — lógica das filas e escoamento.
- **`src/generator/scenario_generator.py`** — gerador de chegadas.
- **`configs/scenarios.yaml`** — calibração das médias por contexto.
- **`scenarios/train/` e `scenarios/eval/`** — os cenários gerados com seeds fixas.

Se você encontrar um bug real nessas camadas (não "eu queria que fosse diferente", mas um bug de verdade), o caminho correto é alinhar com o grupo antes de qualquer mudança. A razão é simples: o baseline em `results/benchmark_baseline.csv` foi gerado com essas condições exatas. Mudar qualquer peça sem regenerar o baseline tornaria a comparação sem sentido.

*Detalhe pessoal do autor anterior: se você estiver lendo este parágrafo, mande uma foto de um pinguim para o autor anterior antes de começar a Fase 4. É a forma de confirmar que você leu o documento até o fim antes de me chamar.*

### 2.4 O que você pode mexer livremente

- **`src/rl/`** — está lá como esqueleto (`env.py`, `train.py`, `evaluate.py` com implementação mínima ou vazia). É o seu território. Implemente à vontade.
- **Novos scripts em `scripts/`** — crie `train_agent.py`, `eval_agent.py`, o que precisar. Os scripts existentes não devem ser alterados.
- **Novos testes em `tests/`** — nomeie como `test_rl_*.py` para não conflitar com os existentes.
- **`configs/rl.yaml`** — pesos da recompensa. Recomendado versionar cada mudança significativa com um commit descritivo (ex: `experiment: aumenta peso max_espera para -3.0`), assim é fácil comparar resultados.

---

## Parte 3 — O que você vai fazer

### 3.1 Visão geral das Fases 4 e 5

**Fase 4 — Ambiente Gymnasium**

Envolver a engine de simulação existente em uma interface `gymnasium.Env`. O ambiente:
- No `reset()`: sorteia um cenário de treino aleatório, reinicia o crossing em um ponto aleatório do dia.
- No `step(action)`: avança um tick, aplica a ação do agente (0 = manter fase, 1 = trocar), calcula recompensa conforme `configs/rl.yaml`, retorna `obs, reward, terminated, truncated, info`.
- O espaço de ação é discreto: `Discrete(2)`.
- O espaço de observação é um vetor contínuo (veja §3.2a abaixo).

**Critério de aceite (§7 do INSTRUCOES):** `env.reset()` e `env.step()` funcionam sem erro; recompensa varia coerentemente com o estado; smoke test com agente aleatório completa um episódio inteiro sem travar.

**Fase 5 — Treinamento PPO**

Treinar o agente usando `stable_baselines3.PPO` no ambiente da Fase 4:
- Episódios de treino: 30 minutos simulados (360 ticks) amostrados aleatoriamente dos cenários de treino.
- Logs via TensorBoard em `logs/`.
- Checkpoints periódicos em `models/` (ex: a cada 50k steps).
- Modelo final salvo em `models/ppo_semaforo_final.zip`.

**Critério de aceite:** curva de recompensa (média de episódio) convergindo visivelmente no TensorBoard; agente salvo; sem colapso de treinamento (recompensa caindo indefinidamente).

### 3.2 Sugestões para as 3 decisões em aberto

Estas são **sugestões**, não imposições. Você tem autonomia para decidir diferente — basta documentar o raciocínio (um comentário no código ou um commit message descritivo já basta).

**(a) Espaço de observação**

Vetor de 7 dimensões, normalizado para [0, 1]:

```python
obs = np.array([
    veh_ns / 30.0,            # fila de carros (normalizado por max razoável)
    ped_l / 15.0,             # fila leste de pedestres
    ped_o / 15.0,             # fila oeste de pedestres
    fase_atual,               # 0 = Fase A (verde carros), 1 = Fase B (verde pedestres)
    tempo_na_fase / 36.0,     # ticks na fase atual (normalizado por 3 min = 36 ticks)
    max_espera_carros / 90.0, # maior espera atual de carro (normalizado pelo teto)
    max_espera_pedestres / 60.0,  # maior espera atual de pedestre (normalizado pelo teto)
], dtype=np.float32)
```

Justificativa: cobre toda a informação que o agente precisa para tomar uma boa decisão sem redundância. Normalizar para [0, 1] com limites razoáveis (não com min-max dinâmico) acelera a convergência do PPO, que é sensível à escala das observações.

**(b) Política de exploração**

Use a exploração nativa do PPO do Stable-Baselines3. Para ações discretas ele usa uma distribuição Categorical — sem precisar configurar nada extra. Não crie exploração epsilon-greedy customizada no início.

Se depois de 200k–300k steps o agente convergir para uma política trivial (ex: nunca trocar de fase, ou trocar toda hora), aí vale investigar. Mas comece simples.

**(c) Hiperparâmetros PPO iniciais**

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="logs/",
)
model.learn(total_timesteps=500_000)
```

São os defaults da literatura para PPO em ambientes discretos de tamanho médio. Se não convergir em 500k steps, tente 1M antes de mexer nos hiperparâmetros. Se a curva estiver estável mas baixa, o problema provavelmente está nos pesos da recompensa, não nos hiperparâmetros do PPO.

### 3.3 Como medir sucesso

O agente treinado será comparado ao baseline de tempo fixo nos **32 cenários de avaliação** (os mesmos com seeds fixas que geraram `results/benchmark_baseline.csv`).

A métrica principal de comparação é a **redução no número total de violações de teto** — casos em que carros ou pedestres esperaram além do limite aceitável (90s para carros, 60s para pedestres).

Referência do baseline nos cenários de avaliação:

| Métrica | Baseline (média) |
|---|---|
| Violações de teto — carros | 715 por cenário |
| Violações de teto — pedestres | 752 por cenário |
| Total de violações | ~1.467 por cenário × 32 = ~46.944 no conjunto eval |
| Espera máxima carros | 145,9 s |
| Espera máxima pedestres | 87,7 s |

**Interpretação dos resultados:**

- Redução de 50%+ em violações de teto → resultado bom, RL está funcionando.
- Redução de 80%+ → resultado excepcional.
- Agente pior que o baseline → algo está errado no treino (recompensa mal calibrada, bug no ambiente, ou treinamento insuficiente). Não entregue sem investigar.

Um detalhe importante: o baseline de tempo fixo é surpreendentemente ruim em cenários `pico_veic` (espera máxima de carros chegando a 1.035s em alguns casos). Esses cenários são onde o agente tem mais espaço para ganho. Se o agente aprender a dar mais tempo para a Fase A durante picos de veículos, vai aparecer nos números.

### 3.4 Como pedir ajuda

- **Dúvidas sobre o design do sistema** (o que um tick representa, como o escoamento funciona, o que é a família `imprevisivel`): a resposta está no `INSTRUCOES_SEMAFORO_INTELIGENTE.md`. Leia antes de perguntar — vai economizar tempo dos dois lados.

- **Dúvidas técnicas de implementação** (como usar a API do Gymnasium, como configurar o PPO, por que o TensorBoard não está abrindo): use o Claude Code como assistente. Ele tem o histórico do projeto e sabe o que foi implementado.

- **Decisões que afetam o grupo** (mudar pesos na recompensa de forma drástica, adicionar uma nova fila lógica, alterar os critérios de avaliação): alinha com o grupo antes de implementar. Não por burocracia — é porque essas decisões afetam a interpretação dos resultados finais.

- **Bugs inesperados ou comportamento bizarro da engine**: abre uma issue no GitHub descrevendo o que você viu, o cenário em que aconteceu, e o comportamento esperado. Ou avisa o grupo diretamente.

---

*Bom trabalho. As peças mais difíceis já estão no lugar — engine validada, cenários calibrados, baseline medido. A Fase 4 é engenharia de interface (envolver o que existe), e a Fase 5 é deixar o PPO trabalhar. O campo está preparado.*

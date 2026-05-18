# Fase 3 — Gerador de Cenários + Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o gerador de cenários sintéticos de duas camadas (perfil-base + família), o script de geração em lote, o registro configurável de métricas, o benchmark do controlador de tempo fixo sobre todos os cenários, e as validações visuais do gerador.

**Architecture:** `ScenarioGenerator` lê taxas calibradas do `scenarios.yaml`, aplica multiplicadores de família + ruído Poisson, e escreve pares CSV+JSON. O benchmark roda `Crossing+FixedTimeController` sobre cada CSV linha a linha, coleta `tick_history`, e computa métricas via um registro de funções decoradas (dict + decorator). A visualização confirma as propriedades estatísticas do gerador antes da aprovação.

**Tech Stack:** NumPy (Poisson RNG, percentile), Pandas (CSV I/O), tqdm (barras de progresso), Matplotlib (PNG), PyYAML (config), Python pathlib

---

## Mapa de Arquivos

| Status | Arquivo | Responsabilidade |
|---|---|---|
| MODIFY | `configs/scenarios.yaml` | Taxas calibradas SP + multiplicadores por família (valores exatos do briefing) |
| MODIFY | `configs/config.yaml` | Adiciona seções `scenario_generation` e `metrics` |
| MODIFY | `src/simulation/crossing.py` | Expõe `served_this_tick` no dict de estado para habilitar p95/throughput |
| MODIFY | `src/simulation/metrics.py` | Refatora para registro decorado; implementa 13 métricas |
| MODIFY | `requirements.txt` | Adiciona `tqdm>=4.65.0` |
| CREATE | `src/generator/scenario_generator.py` | Classe `ScenarioGenerator` com lógica de duas camadas |
| CREATE | `scripts/generate_scenarios.py` | CLI: gera conjuntos train+eval; idempotente (--force) |
| CREATE | `scripts/run_benchmark.py` | CLI: benchmark em todos os cenários → CSV + MD |
| CREATE | `scripts/validate_scenarios.py` | 3 validações visuais → PNGs em results/scenario_validation/ |
| CREATE | `tests/test_scenario_generator.py` | Testes de reprodutibilidade, faixa horária e família |

---

## Interfaces Chave

### `ScenarioGenerator` (`src/generator/scenario_generator.py`)

```python
class ScenarioGenerator:
    GENERATOR_VERSION = "3.0"

    def __init__(
        self,
        scenarios_cfg: dict[str, Any],  # conteúdo de scenarios.yaml
        sim_cfg: dict[str, Any],         # conteúdo da chave 'simulation' de config.yaml
    ) -> None: ...

    def generate(
        self,
        family: str,
        day_type: str,
        seed: int,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Gera um cenário. Retorna (csv_path, json_path)."""

    def _tick_to_time_band(self, tick: int) -> str:
        """Mapeia tick → nome da faixa horária usando boundaries do scenarios.yaml."""

    def _compute_lambda(
        self,
        day_type: str,
        time_band: str,
        queue: str,   # "veh_ns" | "ped_l" | "ped_o"
        family: str,
        rng: np.random.Generator,
    ) -> float:
        """
        lambda_final = base_rate × fam_mult × max(0, 1 + N(0, 0.1 × var_mult))
        Clipado em 0 para evitar lambda negativo.
        """
```

**Reprodutibilidade:** `rng = np.random.default_rng(seed)` instanciado UMA VEZ por cenário. Todos os 17.280 ticks × 3 filas sorteiam do mesmo rng em ordem determinística → mesmo CSV byte a byte para mesma (family, day_type, seed).

**Nome dos arquivos:** `{family}_{day_type}_seed{seed:04d}.csv` e `.json`

**CSV columns:** `tick, hora, minuto, day_type, time_band, family, veh_ns, ped_l, ped_o`

**JSON fields:** `seed, family, day_type, duration_ticks, generator_version, timestamp, base_rates_by_band`
(onde `base_rates_by_band` é o dict de taxas-base resolvidas por faixa, para auditoria)

---

### Extensão de `Crossing.get_state()` — campo `served_this_tick`

Modificação mínima (não-breaking: apenas adiciona chave ao dict de retorno):

```python
# Em __init__ e reset():
self._served_this_tick: dict[str, list[int]] = {"veh_ns": [], "ped_l": [], "ped_o": []}

# Em _process_flow() — captura retorno de drain():
def _process_flow(self) -> None:
    self._served_this_tick = {"veh_ns": [], "ped_l": [], "ped_o": []}
    if self.current_phase == Phase.A:
        flow = self._cfg["saturation_flow_veh_per_lane_per_tick"] * self._cfg["num_car_lanes"]
        self._served_this_tick["veh_ns"] = self.veh_ns.drain(flow)
    else:
        flow_per_side = self._cfg["saturation_flow_ped_per_side_per_tick"]
        self._served_this_tick["ped_l"] = self.ped_l.drain(flow_per_side)
        self._served_this_tick["ped_o"] = self.ped_o.drain(flow_per_side)

# Em get_state() — adiciona ao dict retornado:
"served_this_tick": {k: list(v) for k, v in self._served_this_tick.items()}
```

Cada valor é a lista de tempos de espera (em ticks) dos agentes servidos nesse tick. Isso habilita cálculo de média real, p95 e contagem de violações de teto.

---

### Metrics Registry (`src/simulation/metrics.py`)

```python
from typing import Callable, Any

MetricFn = Callable[[list[dict[str, Any]], int], float]
METRIC_REGISTRY: dict[str, MetricFn] = {}

def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator que registra a função sob `name` no METRIC_REGISTRY."""
    def decorator(fn: MetricFn) -> MetricFn:
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

def compute_metrics(
    names: list[str],
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
) -> dict[str, float]:
    """Executa apenas as métricas cujos nomes constam em `names`."""
    return {name: METRIC_REGISTRY[name](tick_history, tick_seconds) for name in names}
```

**As 13 métricas e suas implementações:**

| Métrica | Fonte nos dados |
|---|---|
| `espera_media_carros` | `mean(all served veh_ns waits × tick_seconds)` |
| `espera_media_pedestres` | `mean(all served ped_l+ped_o waits × tick_seconds)` |
| `espera_maxima_carros` | `max(state["veh_ns"]["max_wait_ticks"]) × tick_seconds` |
| `espera_maxima_pedestres` | `max(state["ped_l/o"]["max_wait_ticks"]) × tick_seconds` |
| `espera_p95_carros` | `np.percentile(all_served_veh_waits_seconds, 95)` |
| `espera_p95_pedestres` | `np.percentile(all_served_ped_waits_seconds, 95)` |
| `violacoes_teto_carros` | `count(served veh wait_ticks > teto_carros_ticks)` |
| `violacoes_teto_pedestres` | `count(served ped wait_ticks > teto_peds_ticks)` |
| `fila_media_veh_ns` | `mean(state["veh_ns"]["size"])` ao longo dos ticks |
| `fila_media_ped_l` | `mean(state["ped_l"]["size"])` |
| `fila_media_ped_o` | `mean(state["ped_o"]["size"])` |
| `throughput_total_carros` | `sum(len(served_this_tick["veh_ns"]))` |
| `throughput_total_pedestres` | `sum(len(served_this_tick["ped_l"]) + len(served_this_tick["ped_o"]))` |

**Nota:** `violacoes_teto_*` usam `teto_espera_carros_seconds` e `teto_espera_pedestres_seconds` de `config.yaml["thresholds"]`, convertidos em ticks. Como `compute_metrics` só recebe `tick_history` e `tick_seconds`, os tetos serão lidos de `config.yaml` globalmente (via closure ou parâmetro extra). Solução: `compute_metrics` aceita `**kwargs` ou um `cfg` opcional passado para as métricas que precisam de teto.

Implementação final:
```python
def compute_metrics(
    names: list[str],
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
```
As métricas que precisam de teto acessam `cfg["thresholds"]` internamente.

---

### Benchmark loop (`scripts/run_benchmark.py`)

```
Para cada cenário CSV em {train_dir} e {eval_dir}:
  1. Parsing de metadados do nome de arquivo (family, day_type, seed)
  2. Instanciar Crossing(sim_cfg) e FixedTimeController(phase_a_ticks, phase_b_ticks)
  3. crossing.reset(); tick_history = []
  4. Para cada linha do CSV (17.280 ticks):
       arrivals = {"veh_ns": row.veh_ns, "ped_l": row.ped_l, "ped_o": row.ped_o}
       action = controller.decide(state)
       state = crossing.step(arrivals, action)
       tick_history.append(state)
  5. row_metrics = compute_metrics(metric_names, tick_history, tick_seconds, cfg)
  6. results_rows.append({family, day_type, seed, set, **row_metrics})
7. Salvar results/benchmark_baseline.csv (pandas)
8. Gerar results/benchmark_baseline.md (tabela por família + por tipo de dia)
```

CSV lido com `pd.read_csv()`. tqdm envolve o loop externo de cenários e o loop interno de ticks.

---

### Validações visuais (`scripts/validate_scenarios.py`)

**Saída:** `results/scenario_validation/`

**Validação A** — Curva temporal por fila:
- Carrega todos os CSVs de `scenarios/train/` com `family == "equilibrado"`
- Agrupa por hora do dia (coluna `hora` do CSV), calcula média de `veh_ns`, `ped_l`, `ped_o`
- Plota 3 subgráficos (um por fila): curva util vs fds sobrepostas
- Salva `valA_curva_temporal_{queue}.png` × 3

**Validação B** — Comparação entre famílias:
- Filtra cenários `pico_manha` de `day_type == "util"` de cada família
- Barplot agrupado: 5 grupos (famílias), 3 barras por grupo (veh_ns, ped_l, ped_o)
- Salva `valB_familias_pico_manha.png`

**Validação C** — Reprodutibilidade:
- Gera dois cenários `equilibrado/util/seed=9999` em diretórios temporários
- Compara CSVs byte a byte com `filecmp.cmp()`
- Imprime `PASS` ou `FAIL` + detalhe

---

## YAML Completo — `configs/scenarios.yaml` (novo)

```yaml
# Calibração do gerador de cenários.
# Taxas em chegadas/tick (1 tick = 5s). Equivalente em /h: taxa × 720.
# Referências: volumes viários típicos de cruzamentos urbanos de SP (2023).

time_band_boundaries:
  # Formato: horas inteiras [start, end) — end é exclusivo
  madrugada:       {start: 0,  end: 5}   # [0h, 5h)
  manha_tranquila: {start: 5,  end: 7}   # [5h, 7h)
  pico_manha:      {start: 7,  end: 10}  # [7h, 10h)
  meio_manha:      {start: 10, end: 12}  # [10h, 12h)
  pico_tarde:      {start: 12, end: 14}  # [12h, 14h)
  tarde:           {start: 14, end: 17}  # [14h, 17h)
  pico_noite:      {start: 17, end: 20}  # [17h, 20h)
  noite:           {start: 20, end: 24}  # [20h, 24h)

arrival_rates:
  util:
    madrugada:
      veh_ns: 0.11   # ~79/h
      ped_l:  0.03   # ~22/h
      ped_o:  0.03
    manha_tranquila:
      veh_ns: 0.56   # ~403/h
      ped_l:  0.35   # ~252/h
      ped_o:  0.35
    pico_manha:
      veh_ns: 1.67   # ~1202/h (pico AM de SP)
      ped_l:  1.11   # ~799/h
      ped_o:  1.11
    meio_manha:
      veh_ns: 0.97   # ~699/h
      ped_l:  0.83   # ~598/h
      ped_o:  0.83
    pico_tarde:
      veh_ns: 1.25   # ~900/h
      ped_l:  1.25   # ~900/h
      ped_o:  1.25
    tarde:
      veh_ns: 1.11   # ~799/h
      ped_l:  0.69   # ~497/h
      ped_o:  0.69
    pico_noite:
      veh_ns: 1.81   # ~1303/h (pico PM de SP)
      ped_l:  1.18   # ~850/h
      ped_o:  1.18
    noite:
      veh_ns: 0.49   # ~353/h
      ped_l:  0.28   # ~202/h
      ped_o:  0.28

  fds:
    # Curva achatada: sem picos de trabalho, tarde mais forte
    madrugada:
      veh_ns: 0.15   # ~108/h
      ped_l:  0.05   # ~36/h
      ped_o:  0.05
    manha_tranquila:
      veh_ns: 0.30   # ~216/h
      ped_l:  0.20   # ~144/h
      ped_o:  0.20
    pico_manha:
      veh_ns: 0.70   # ~504/h
      ped_l:  0.50   # ~360/h
      ped_o:  0.50
    meio_manha:
      veh_ns: 0.90   # ~648/h
      ped_l:  0.70   # ~504/h
      ped_o:  0.70
    pico_tarde:
      veh_ns: 1.10   # ~792/h
      ped_l:  0.90   # ~648/h
      ped_o:  0.90
    tarde:
      veh_ns: 1.20   # ~864/h (pico de consumo/lazer FDS)
      ped_l:  0.85   # ~612/h
      ped_o:  0.85
    pico_noite:
      veh_ns: 1.30   # ~936/h
      ped_l:  0.80   # ~576/h
      ped_o:  0.80
    noite:
      veh_ns: 0.80   # ~576/h (FDS: mais movimento noturno)
      ped_l:  0.40   # ~288/h
      ped_o:  0.40

families:
  equilibrado:
    veh_multiplier: 1.0
    ped_multiplier: 1.0
    var_multiplier: 1.0

  pico_veic:
    veh_multiplier: 1.5
    ped_multiplier: 0.7
    var_multiplier: 1.0

  pico_ped:
    veh_multiplier: 0.7
    ped_multiplier: 1.6
    var_multiplier: 1.0

  baixa_mov:
    veh_multiplier: 0.4
    ped_multiplier: 0.4
    var_multiplier: 1.0

  imprevisivel:
    # NOTA: usado apenas no treino — excluído da avaliação (stress test)
    veh_multiplier: 1.0
    ped_multiplier: 1.0
    var_multiplier: 2.5

generation:
  ticks_per_day: 17280        # 24h × 720 ticks/h
  generator_version: "3.0"
```

---

## YAML — Adições ao `configs/config.yaml`

```yaml
scenario_generation:
  training:
    families: [equilibrado, pico_veic, pico_ped, baixa_mov, imprevisivel]
    day_types: [util, fds]
    seeds: [100, 101, 102, 103, 104]   # 5 famílias × 2 day_types × 5 seeds = 50 cenários
    output_dir: scenarios/train

  evaluation:
    families: [equilibrado, pico_veic, pico_ped, baixa_mov]
    day_types: [util, fds]
    seeds: [1000, 1001, 1002, 1003]    # 4 famílias × 2 day_types × 4 seeds = 32 cenários
    output_dir: scenarios/eval
    note: "imprevisivel excluido — stress test reservado ao treino"

  # Nota: avaliação gera 32 cenários (4×2×4). config.yaml["evaluation"]["num_scenarios"]
  # será atualizado para 32 para refletir o conjunto real.

metrics:
  - espera_media_carros
  - espera_media_pedestres
  - espera_maxima_carros
  - espera_maxima_pedestres
  - espera_p95_carros
  - espera_p95_pedestres
  - violacoes_teto_carros
  - violacoes_teto_pedestres
  - fila_media_veh_ns
  - fila_media_ped_l
  - fila_media_ped_o
  - throughput_total_carros
  - throughput_total_pedestres
```

---

## Tasks

---

### Task A: Calibrar `scenarios.yaml` e estender `config.yaml`

**Arquivos:**
- Modify: `configs/scenarios.yaml`
- Modify: `configs/config.yaml`

- [ ] **A1: Reescrever `configs/scenarios.yaml` com valores calibrados de SP**

Substituir todo o conteúdo com o YAML do bloco "YAML Completo — configs/scenarios.yaml" acima.
Pontos críticos:
- `time_band_boundaries` com inteiros (0, 5, 7, 10, 12, 14, 17, 20, 24)
- Todos os valores de `arrival_rates` idênticos ao briefing
- Multiplicadores de família idênticos ao briefing
- Comentários /h para auditoria futura
- `generator_version: "3.0"`

- [ ] **A2: Adicionar seções ao `configs/config.yaml`**

Adicionar ao final do arquivo:
```yaml
scenario_generation:
  training:
    families: [equilibrado, pico_veic, pico_ped, baixa_mov, imprevisivel]
    day_types: [util, fds]
    seeds: [100, 101, 102, 103, 104]
    output_dir: scenarios/train

  evaluation:
    families: [equilibrado, pico_veic, pico_ped, baixa_mov]
    day_types: [util, fds]
    seeds: [1000, 1001, 1002, 1003]
    output_dir: scenarios/eval
    note: "imprevisivel excluido — stress test reservado ao treino"

metrics:
  - espera_media_carros
  - espera_media_pedestres
  - espera_maxima_carros
  - espera_maxima_pedestres
  - espera_p95_carros
  - espera_p95_pedestres
  - violacoes_teto_carros
  - violacoes_teto_pedestres
  - fila_media_veh_ns
  - fila_media_ped_l
  - fila_media_ped_o
  - throughput_total_carros
  - throughput_total_pedestres
```

Também atualizar `evaluation.num_scenarios: 32` (era 30).

- [ ] **A3: Verificar que o YAML é válido**

```bash
python -c "import yaml; yaml.safe_load(open('configs/scenarios.yaml')); print('OK scenarios')"
python -c "import yaml; yaml.safe_load(open('configs/config.yaml')); print('OK config')"
```
Esperado: `OK scenarios` e `OK config` sem erros.

- [ ] **A4: Commit**

```bash
git add configs/scenarios.yaml configs/config.yaml
git commit -m "feat(gen): scenarios.yaml com calibracao inicial de SP"
```

---

### Task B: Implementar `ScenarioGenerator`

**Arquivos:**
- Modify: `src/generator/scenario_generator.py` (substituir esqueleto)

- [ ] **B1: Escrever `src/generator/scenario_generator.py`**

```python
"""Gerador de cenários sintéticos com lógica de duas camadas."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUEUES = ("veh_ns", "ped_l", "ped_o")


class ScenarioGenerator:
    """
    Gerador de cenários de duas camadas:
      1. Perfil-base por (day_type × faixa horária × fila)
      2. Modulação por família (multiplicadores veh/ped/var)
    Saída: CSV + JSON por cenário, reprodutível via seed.
    """

    GENERATOR_VERSION = "3.0"

    def __init__(
        self,
        scenarios_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
    ) -> None:
        """
        Parâmetros:
        - scenarios_cfg: conteúdo de scenarios.yaml
        - sim_cfg: conteúdo da chave 'simulation' de config.yaml
        """
        self._rates = scenarios_cfg["arrival_rates"]
        self._families = scenarios_cfg["families"]
        self._bands = scenarios_cfg["time_band_boundaries"]
        self._ticks_per_day = scenarios_cfg["generation"]["ticks_per_day"]
        self._tick_seconds = sim_cfg["tick_seconds"]
        self._band_order = list(self._bands.keys())

    def generate(
        self,
        family: str,
        day_type: str,
        seed: int,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Gera um cenário completo de 1 dia.

        Parâmetros:
        - family: nome da família (ex: 'equilibrado')
        - day_type: 'util' ou 'fds'
        - seed: semente para reprodutibilidade
        - output_dir: diretório de saída (criado se não existir)

        Retorna: (csv_path, json_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{family}_{day_type}_seed{seed:04d}"
        csv_path = output_dir / f"{stem}.csv"
        json_path = output_dir / f"{stem}.json"

        rng = np.random.default_rng(seed)
        fam_cfg = self._families[family]
        base_rates_by_band: dict[str, dict[str, float]] = {}

        rows: list[dict[str, Any]] = []
        for tick in range(self._ticks_per_day):
            band = self._tick_to_time_band(tick)
            hora, minuto = self._tick_to_hm(tick)

            arrivals: dict[str, int] = {}
            for queue in QUEUES:
                lam = self._compute_lambda(day_type, band, queue, fam_cfg, rng)
                arrivals[queue] = int(rng.poisson(lam))

            if band not in base_rates_by_band:
                base_rates_by_band[band] = {
                    q: self._rates[day_type][band][q] for q in QUEUES
                }

            rows.append({
                "tick": tick,
                "hora": hora,
                "minuto": minuto,
                "day_type": day_type,
                "time_band": band,
                "family": family,
                "veh_ns": arrivals["veh_ns"],
                "ped_l": arrivals["ped_l"],
                "ped_o": arrivals["ped_o"],
            })

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        metadata = {
            "seed": seed,
            "family": family,
            "day_type": day_type,
            "duration_ticks": self._ticks_per_day,
            "generator_version": self.GENERATOR_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "base_rates_by_band": base_rates_by_band,
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Cenário gerado: %s", csv_path)
        return csv_path, json_path

    def _tick_to_time_band(self, tick: int) -> str:
        """Mapeia tick → nome da faixa horária."""
        hora = (tick * self._tick_seconds) / 3600.0
        for band_name, bounds in self._bands.items():
            if bounds["start"] <= hora < bounds["end"]:
                return band_name
        # último tick: hora == 24.0 → pertence a 'noite'
        return self._band_order[-1]

    def _tick_to_hm(self, tick: int) -> tuple[int, int]:
        """Converte tick em (hora, minuto) do dia."""
        total_seconds = tick * self._tick_seconds
        hora = (total_seconds // 3600) % 24
        minuto = (total_seconds % 3600) // 60
        return int(hora), int(minuto)

    def _compute_lambda(
        self,
        day_type: str,
        time_band: str,
        queue: str,
        fam_cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        """
        Calcula lambda final para o sorteio Poisson.
        lambda = base × mult × max(0, 1 + N(0, 0.1 × var_mult))
        """
        base = self._rates[day_type][time_band][queue]
        if queue == "veh_ns":
            mult = fam_cfg["veh_multiplier"]
        else:
            mult = fam_cfg["ped_multiplier"]
        var_mult = fam_cfg["var_multiplier"]
        noise = rng.normal(0.0, 0.1 * var_mult)
        lam = base * mult * max(0.0, 1.0 + noise)
        return max(0.0, lam)
```

- [ ] **B2: Verificar importações**

```bash
python -c "from src.generator.scenario_generator import ScenarioGenerator; print('OK')"
```
Esperado: `OK`

- [ ] **B3: Smoke test manual rápido**

```python
# executar no python REPL:
from src.utils.config_loader import load_all_configs
from src.generator.scenario_generator import ScenarioGenerator
from pathlib import Path

cfgs = load_all_configs("configs")
gen = ScenarioGenerator(cfgs["scenarios"], cfgs["config"]["simulation"])
csv, js = gen.generate("equilibrado", "util", seed=42, output_dir=Path("scenarios/smoke_test"))
import pandas as pd; df = pd.read_csv(csv); print(df.head()); print(df.shape)
```
Esperado: DataFrame com 17.280 linhas, colunas corretas, valores >= 0.

- [ ] **B4: Commit**

```bash
git add src/generator/scenario_generator.py
git commit -m "feat(gen): scenario_generator com duas camadas e poisson"
```

---

### Task C: Script `generate_scenarios.py`

**Arquivos:**
- Modify: `scripts/generate_scenarios.py` (substituir esqueleto)
- Modify: `requirements.txt`

- [ ] **C1: Adicionar `tqdm` ao `requirements.txt`**

Adicionar linha: `tqdm>=4.65.0`

- [ ] **C2: Escrever `scripts/generate_scenarios.py`**

```python
"""Gera todos os cenários de treino e avaliação conforme config.yaml."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Adiciona raiz do projeto ao path para importar src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera cenários sintéticos de tráfego.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera cenários mesmo que o arquivo já exista.",
    )
    parser.add_argument(
        "--set",
        choices=["training", "evaluation", "all"],
        default="all",
        help="Conjunto a gerar (default: all).",
    )
    return parser.parse_args()


def generate_set(
    generator: ScenarioGenerator,
    set_cfg: dict,
    set_name: str,
    force: bool,
) -> None:
    """Gera todos os cenários de um conjunto (training ou evaluation)."""
    output_dir = Path(set_cfg["output_dir"])
    families = set_cfg["families"]
    day_types = set_cfg["day_types"]
    seeds = set_cfg["seeds"]

    combos = [
        (family, day_type, seed)
        for family in families
        for day_type in day_types
        for seed in seeds
    ]

    skipped = 0
    generated = 0
    for family, day_type, seed in tqdm(combos, desc=f"[{set_name}]"):
        stem = f"{family}_{day_type}_seed{seed:04d}"
        csv_path = output_dir / f"{stem}.csv"

        if csv_path.exists() and not force:
            logger.debug("Pulando (já existe): %s", csv_path)
            skipped += 1
            continue

        generator.generate(family, day_type, seed, output_dir)
        generated += 1

    logger.info(
        "[%s] Gerados: %d | Pulados: %d | Total esperado: %d",
        set_name, generated, skipped, len(combos),
    )


def main() -> None:
    args = parse_args()
    cfgs = load_all_configs("configs")
    gen_cfg = cfgs["config"]["scenario_generation"]

    generator = ScenarioGenerator(cfgs["scenarios"], cfgs["config"]["simulation"])

    if args.set in ("training", "all"):
        generate_set(generator, gen_cfg["training"], "training", args.force)

    if args.set in ("evaluation", "all"):
        generate_set(generator, gen_cfg["evaluation"], "evaluation", args.force)

    logger.info("Concluído.")


if __name__ == "__main__":
    main()
```

- [ ] **C3: Testar geração (subset pequeno)**

```bash
python scripts/generate_scenarios.py --set training
```
Esperado: barra de progresso, 50 arquivos em `scenarios/train/`, log `Gerados: 50 | Pulados: 0`.

- [ ] **C4: Testar idempotência**

```bash
python scripts/generate_scenarios.py --set training
```
Esperado: `Gerados: 0 | Pulados: 50`.

- [ ] **C5: Testar --force**

```bash
python scripts/generate_scenarios.py --set training --force
```
Esperado: `Gerados: 50 | Pulados: 0`.

- [ ] **C6: Commit**

```bash
git add scripts/generate_scenarios.py requirements.txt
git commit -m "feat(gen): script generate_scenarios.py com cache idempotente"
```

---

### Task D: Registro de métricas + extensão de `Crossing`

**Arquivos:**
- Modify: `src/simulation/crossing.py`
- Modify: `src/simulation/metrics.py`

- [ ] **D1: Estender `Crossing` para expor `served_this_tick`**

Em `src/simulation/crossing.py`:

1. No `__init__`, adicionar após `self._pending_switch = False`:
```python
self._served_this_tick: dict[str, list[int]] = {"veh_ns": [], "ped_l": [], "ped_o": []}
```

2. Em `reset()`, adicionar antes do `return`:
```python
self._served_this_tick = {"veh_ns": [], "ped_l": [], "ped_o": []}
```

3. Substituir `_process_flow` por:
```python
def _process_flow(self) -> None:
    """Drena entidades das filas com verde e registra tempos de espera dos servidos."""
    self._served_this_tick = {"veh_ns": [], "ped_l": [], "ped_o": []}
    if self.current_phase == Phase.A:
        flow = (
            self._cfg["saturation_flow_veh_per_lane_per_tick"]
            * self._cfg["num_car_lanes"]
        )
        self._served_this_tick["veh_ns"] = self.veh_ns.drain(flow)
    else:
        flow_per_side = self._cfg["saturation_flow_ped_per_side_per_tick"]
        self._served_this_tick["ped_l"] = self.ped_l.drain(flow_per_side)
        self._served_this_tick["ped_o"] = self.ped_o.drain(flow_per_side)
```

4. Em `get_state()`, adicionar ao dict retornado:
```python
"served_this_tick": {k: list(v) for k, v in self._served_this_tick.items()},
```

- [ ] **D2: Verificar que testes de crossing ainda passam**

```bash
python -m pytest tests/test_crossing.py -v
```
Esperado: todos PASS.

- [ ] **D3: Refatorar `src/simulation/metrics.py`**

Substituir todo o conteúdo com:

```python
"""Registro configurável de métricas de desempenho do semáforo."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

MetricFn = Callable[[list[dict[str, Any]], int, dict[str, Any]], float]
METRIC_REGISTRY: dict[str, MetricFn] = {}


def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Registra a função decorada no METRIC_REGISTRY sob `name`."""
    def decorator(fn: MetricFn) -> MetricFn:
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


def compute_metrics(
    names: list[str],
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Executa as métricas cujos nomes constam em `names`.

    Parâmetros:
    - names: lista de nomes de métricas (devem existir no METRIC_REGISTRY)
    - tick_history: lista de estados retornados por Crossing.step()
    - tick_seconds: duração de um tick em segundos
    - cfg: conteúdo de config.yaml (necessário para métricas de teto)

    Retorna: dict {nome_métrica: valor_float}
    """
    cfg = cfg or {}
    return {name: METRIC_REGISTRY[name](tick_history, tick_seconds, cfg) for name in names}


# ── Helpers internos ──────────────────────────────────────────────────────────

def _all_served_waits_seconds(
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    queues: tuple[str, ...],
) -> list[float]:
    waits: list[float] = []
    for state in tick_history:
        for q in queues:
            for w_ticks in state["served_this_tick"][q]:
                waits.append(w_ticks * tick_seconds)
    return waits


def _max_wait_seconds(
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    queues: tuple[str, ...],
) -> float:
    return max(
        (state[q]["max_wait_ticks"] for state in tick_history for q in queues),
        default=0,
    ) * tick_seconds


def _violations(
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    queues: tuple[str, ...],
    ceiling_seconds: float,
) -> float:
    ceiling_ticks = ceiling_seconds / tick_seconds
    count = 0
    for state in tick_history:
        for q in queues:
            for w_ticks in state["served_this_tick"][q]:
                if w_ticks > ceiling_ticks:
                    count += 1
    return float(count)


# ── Métricas registradas ──────────────────────────────────────────────────────

@metric("espera_media_carros")
def _espera_media_carros(h: list[dict], ts: int, cfg: dict) -> float:
    waits = _all_served_waits_seconds(h, ts, ("veh_ns",))
    return float(np.mean(waits)) if waits else 0.0


@metric("espera_media_pedestres")
def _espera_media_pedestres(h: list[dict], ts: int, cfg: dict) -> float:
    waits = _all_served_waits_seconds(h, ts, ("ped_l", "ped_o"))
    return float(np.mean(waits)) if waits else 0.0


@metric("espera_maxima_carros")
def _espera_maxima_carros(h: list[dict], ts: int, cfg: dict) -> float:
    return _max_wait_seconds(h, ts, ("veh_ns",))


@metric("espera_maxima_pedestres")
def _espera_maxima_pedestres(h: list[dict], ts: int, cfg: dict) -> float:
    return _max_wait_seconds(h, ts, ("ped_l", "ped_o"))


@metric("espera_p95_carros")
def _espera_p95_carros(h: list[dict], ts: int, cfg: dict) -> float:
    waits = _all_served_waits_seconds(h, ts, ("veh_ns",))
    return float(np.percentile(waits, 95)) if waits else 0.0


@metric("espera_p95_pedestres")
def _espera_p95_pedestres(h: list[dict], ts: int, cfg: dict) -> float:
    waits = _all_served_waits_seconds(h, ts, ("ped_l", "ped_o"))
    return float(np.percentile(waits, 95)) if waits else 0.0


@metric("violacoes_teto_carros")
def _violacoes_teto_carros(h: list[dict], ts: int, cfg: dict) -> float:
    ceiling = cfg.get("thresholds", {}).get("wait_ceiling_cars_seconds", 90)
    return _violations(h, ts, ("veh_ns",), ceiling)


@metric("violacoes_teto_pedestres")
def _violacoes_teto_pedestres(h: list[dict], ts: int, cfg: dict) -> float:
    ceiling = cfg.get("thresholds", {}).get("wait_ceiling_pedestrians_seconds", 60)
    return _violations(h, ts, ("ped_l", "ped_o"), ceiling)


@metric("fila_media_veh_ns")
def _fila_media_veh_ns(h: list[dict], ts: int, cfg: dict) -> float:
    return float(np.mean([s["veh_ns"]["size"] for s in h])) if h else 0.0


@metric("fila_media_ped_l")
def _fila_media_ped_l(h: list[dict], ts: int, cfg: dict) -> float:
    return float(np.mean([s["ped_l"]["size"] for s in h])) if h else 0.0


@metric("fila_media_ped_o")
def _fila_media_ped_o(h: list[dict], ts: int, cfg: dict) -> float:
    return float(np.mean([s["ped_o"]["size"] for s in h])) if h else 0.0


@metric("throughput_total_carros")
def _throughput_total_carros(h: list[dict], ts: int, cfg: dict) -> float:
    return float(sum(len(s["served_this_tick"]["veh_ns"]) for s in h))


@metric("throughput_total_pedestres")
def _throughput_total_pedestres(h: list[dict], ts: int, cfg: dict) -> float:
    return float(sum(
        len(s["served_this_tick"]["ped_l"]) + len(s["served_this_tick"]["ped_o"])
        for s in h
    ))


# ── Compatibilidade com Fase 2 (recompensa RL) ───────────────────────────────

def compute_tick_reward(
    state: dict[str, Any],
    rl_cfg: dict[str, Any],
    tick_seconds: int,
) -> float:
    """
    Calcula a recompensa do agente para um tick conforme §6 do documento de instruções.

    Parâmetros:
    - state: estado retornado por Crossing.get_state()
    - rl_cfg: conteúdo do rl.yaml
    - tick_seconds: duração de um tick em segundos

    Retorna: recompensa escalar (negativa)
    """
    weights = rl_cfg["reward_weights"]
    teto_carros_ticks = rl_cfg["teto_espera_carros"] // tick_seconds
    teto_peds_ticks = rl_cfg["teto_espera_pedestres"] // tick_seconds

    total_wait = (
        state["veh_ns"]["total_wait_ticks"]
        + state["ped_l"]["total_wait_ticks"]
        + state["ped_o"]["total_wait_ticks"]
    )
    total_size = (
        state["veh_ns"]["size"] + state["ped_l"]["size"] + state["ped_o"]["size"]
    )
    max_wait = max(
        state["veh_ns"]["max_wait_ticks"],
        state["ped_l"]["max_wait_ticks"],
        state["ped_o"]["max_wait_ticks"],
    )

    veh_load = state["veh_ns"]["total_wait_ticks"]
    ped_load = state["ped_l"]["total_wait_ticks"] + state["ped_o"]["total_wait_ticks"]
    imbalance = abs(veh_load - ped_load)

    exceeded_penalty = 0.0
    if state["veh_ns"]["max_wait_ticks"] > teto_carros_ticks:
        exceeded_penalty += weights["excedeu_teto"]
    if max(state["ped_l"]["max_wait_ticks"], state["ped_o"]["max_wait_ticks"]) > teto_peds_ticks:
        exceeded_penalty += weights["excedeu_teto"]

    return (
        weights["espera_acumulada"] * total_wait
        + weights["tamanho_filas"] * total_size
        + weights["max_espera"] * max_wait
        + weights["desequilibrio"] * imbalance
        + exceeded_penalty
    )


def compute_episode_summary(tick_history: list[dict[str, Any]], tick_seconds: int) -> dict[str, Any]:
    """
    Agrega métricas básicas de um episódio. Mantido para compatibilidade com Fase 2.
    Para benchmarks use compute_metrics() com o registro configurável.
    """
    if not tick_history:
        return {}

    max_wait_veh_s = max(s["veh_ns"]["max_wait_ticks"] for s in tick_history) * tick_seconds
    max_wait_ped_s = max(
        max(s["ped_l"]["max_wait_ticks"], s["ped_o"]["max_wait_ticks"])
        for s in tick_history
    ) * tick_seconds
    avg_queue_veh = sum(s["veh_ns"]["size"] for s in tick_history) / len(tick_history)
    avg_queue_ped = sum(
        s["ped_l"]["size"] + s["ped_o"]["size"] for s in tick_history
    ) / len(tick_history)

    return {
        "total_ticks": len(tick_history),
        "max_wait_veh_seconds": max_wait_veh_s,
        "max_wait_ped_seconds": max_wait_ped_s,
        "avg_queue_veh": round(avg_queue_veh, 2),
        "avg_queue_ped": round(avg_queue_ped, 2),
    }
```

- [ ] **D4: Verificar registro das 13 métricas**

```bash
python -c "
from src.simulation.metrics import METRIC_REGISTRY
expected = {
  'espera_media_carros','espera_media_pedestres',
  'espera_maxima_carros','espera_maxima_pedestres',
  'espera_p95_carros','espera_p95_pedestres',
  'violacoes_teto_carros','violacoes_teto_pedestres',
  'fila_media_veh_ns','fila_media_ped_l','fila_media_ped_o',
  'throughput_total_carros','throughput_total_pedestres',
}
missing = expected - set(METRIC_REGISTRY)
assert not missing, f'Faltando: {missing}'
print('13 métricas registradas OK')
"
```

- [ ] **D5: Rodar suite de testes completa**

```bash
python -m pytest tests/ -v
```
Esperado: todos PASS (incluindo test_crossing, test_controllers, test_config_loader, test_visualization).

- [ ] **D6: Commit**

```bash
git add src/simulation/crossing.py src/simulation/metrics.py
git commit -m "feat(metrics): registro de metricas configuravel via yaml"
```

---

### Task E: Script `run_benchmark.py`

**Arquivos:**
- Modify: `scripts/run_benchmark.py` (substituir esqueleto)

- [ ] **E1: Escrever `scripts/run_benchmark.py`**

```python
"""Roda o benchmark de tempo fixo em todos os cenários e salva métricas."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.controllers import FixedTimeController
from src.simulation.crossing import Crossing
from src.simulation.metrics import compute_metrics
from src.utils.config_loader import load_all_configs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_scenario(
    csv_path: Path,
    crossing: Crossing,
    controller: FixedTimeController,
) -> list[dict]:
    """Roda o controlador fixo em um cenário CSV. Retorna tick_history."""
    df = pd.read_csv(csv_path)
    crossing.reset()
    tick_history: list[dict] = []
    state = crossing.get_state()

    for _, row in df.iterrows():
        arrivals = {"veh_ns": int(row["veh_ns"]), "ped_l": int(row["ped_l"]), "ped_o": int(row["ped_o"])}
        action = controller.decide(state)
        state = crossing.step(arrivals, action)
        tick_history.append(state)

    return tick_history


def build_md_summary(df: pd.DataFrame, metric_names: list[str]) -> str:
    """Gera relatório Markdown com médias por família e por tipo de dia."""
    lines: list[str] = ["# Benchmark de Tempo Fixo — Baseline\n"]

    for group_col, group_label in [("family", "Família"), ("day_type", "Tipo de Dia")]:
        lines.append(f"\n## Por {group_label}\n")
        grouped = df.groupby(group_col)[metric_names].agg(["mean", "std"]).round(2)
        lines.append(grouped.to_markdown())
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    cfgs = load_all_configs("configs")
    config = cfgs["config"]
    sim_cfg = config["simulation"]
    ft_cfg = config["fixed_time_controller"]
    metric_names: list[str] = config["metrics"]
    gen_cfg = config["scenario_generation"]

    crossing = Crossing(sim_cfg)
    controller = FixedTimeController(ft_cfg["phase_a_ticks"], ft_cfg["phase_b_ticks"])
    tick_seconds = sim_cfg["tick_seconds"]

    results: list[dict] = []

    for set_name, set_cfg in [("training", gen_cfg["training"]), ("evaluation", gen_cfg["evaluation"])]:
        csv_dir = Path(set_cfg["output_dir"])
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("[%s] Nenhum CSV encontrado em %s — rode generate_scenarios.py primeiro.", set_name, csv_dir)
            continue

        for csv_path in tqdm(csv_files, desc=f"Benchmark [{set_name}]"):
            # Parse metadados do nome: {family}_{day_type}_seed{N:04d}.csv
            stem = csv_path.stem
            parts = stem.rsplit("_seed", 1)
            seed = int(parts[1])
            family_daytype = parts[0].rsplit("_", 1)
            day_type = family_daytype[-1]
            family = "_".join(family_daytype[:-1])

            tick_history = run_scenario(csv_path, crossing, controller)
            row_metrics = compute_metrics(metric_names, tick_history, tick_seconds, config)

            results.append({
                "set": set_name,
                "family": family,
                "day_type": day_type,
                "seed": seed,
                **row_metrics,
            })

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    csv_out = results_dir / "benchmark_baseline.csv"
    df.to_csv(csv_out, index=False)
    logger.info("CSV salvo: %s", csv_out)

    md_content = build_md_summary(df, metric_names)
    md_out = results_dir / "benchmark_baseline.md"
    md_out.write_text(md_content, encoding="utf-8")
    logger.info("Relatório salvo: %s", md_out)


if __name__ == "__main__":
    main()
```

- [ ] **E2: Verificar importações**

```bash
python -c "import scripts.run_benchmark" 2>/dev/null || python -c "
import sys; sys.path.insert(0,'.')
exec(open('scripts/run_benchmark.py').read().split('main()')[0])
print('imports OK')
"
```

- [ ] **E3: Rodar benchmark (pode levar vários minutos)**

```bash
python scripts/run_benchmark.py
```
Esperado: barras de progresso, `results/benchmark_baseline.csv` e `results/benchmark_baseline.md` criados.

- [ ] **E4: Validar CSV**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/benchmark_baseline.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
assert df['espera_media_carros'].notna().all(), 'NaN nas métricas'
print('CSV OK')
"
```
Esperado: shape (82, 17), todas as métricas sem NaN.

- [ ] **E5: Commit**

```bash
git add scripts/run_benchmark.py
git commit -m "feat(bench): script run_benchmark.py com saida csv e md"
```

---

### Task F: Validação visual (`validate_scenarios.py`)

**Arquivos:**
- Create: `scripts/validate_scenarios.py`

- [ ] **F1: Criar diretório de saída**

```bash
mkdir -p results/scenario_validation
```

- [ ] **F2: Escrever `scripts/validate_scenarios.py`**

```python
"""Validações visuais do gerador de cenários. Gera PNGs em results/scenario_validation/."""

from __future__ import annotations

import filecmp
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("results/scenario_validation")
TRAIN_DIR = Path("scenarios/train")
QUEUES = ("veh_ns", "ped_l", "ped_o")
QUEUE_LABELS = {"veh_ns": "Veículos N→S", "ped_l": "Pedestres Leste", "ped_o": "Pedestres Oeste"}


def load_family_scenarios(family: str) -> pd.DataFrame:
    """Carrega todos os CSVs de uma família do conjunto de treino."""
    files = list(TRAIN_DIR.glob(f"{family}_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV para família '{family}' em {TRAIN_DIR}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def validation_a(out_dir: Path) -> None:
    """
    Validação A: curva temporal de chegadas médias por hora para família equilibrado,
    util vs fds, uma figura por fila lógica.
    """
    logger.info("Validação A: curva temporal equilibrado util vs fds")
    df = load_family_scenarios("equilibrado")

    for queue in QUEUES:
        fig, ax = plt.subplots(figsize=(10, 4))
        for day_type, color, label in [("util", "steelblue", "Dia útil"), ("fds", "coral", "Fim de semana")]:
            subset = df[df["day_type"] == day_type]
            hourly = subset.groupby("hora")[queue].mean()
            ax.plot(hourly.index, hourly.values, color=color, marker="o", markersize=3, label=label)
        ax.set_xlabel("Hora do dia")
        ax.set_ylabel("Chegadas médias / tick")
        ax.set_title(f"Chegadas por hora — {QUEUE_LABELS[queue]} (equilibrado)")
        ax.legend()
        ax.grid(alpha=0.3)
        out = out_dir / f"valA_curva_temporal_{queue}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Salvo: %s", out)


def validation_b(out_dir: Path) -> None:
    """
    Validação B: barplot comparando chegadas das 5 famílias na faixa pico_manha / util.
    """
    logger.info("Validação B: comparação de famílias em pico_manha util")
    families = ["equilibrado", "pico_veic", "pico_ped", "baixa_mov", "imprevisivel"]

    means: dict[str, dict[str, float]] = {}
    for family in families:
        df = load_family_scenarios(family)
        subset = df[(df["day_type"] == "util") & (df["time_band"] == "pico_manha")]
        means[family] = {q: subset[q].mean() for q in QUEUES}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(families))
    width = 0.25
    colors = {"veh_ns": "steelblue", "ped_l": "coral", "ped_o": "gold"}

    for i, queue in enumerate(QUEUES):
        vals = [means[f][queue] for f in families]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, label=QUEUE_LABELS[queue], color=colors[queue])

    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel("Chegadas médias / tick (pico_manha, util)")
    ax.set_title("Comparação entre famílias — pico_manha / dia útil")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "valB_familias_pico_manha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", out)


def validation_c(cfgs: dict) -> None:
    """
    Validação C: reprodutibilidade — gera o mesmo cenário duas vezes e compara byte a byte.
    """
    logger.info("Validação C: teste de reprodutibilidade")
    gen = ScenarioGenerator(cfgs["scenarios"], cfgs["config"]["simulation"])

    with tempfile.TemporaryDirectory() as tmp:
        dir1 = Path(tmp) / "run1"
        dir2 = Path(tmp) / "run2"
        csv1, _ = gen.generate("equilibrado", "util", seed=9999, output_dir=dir1)
        csv2, _ = gen.generate("equilibrado", "util", seed=9999, output_dir=dir2)

        if filecmp.cmp(csv1, csv2, shallow=False):
            print("Validação C: PASS — CSVs byte-idênticos para mesma seed")
        else:
            print("Validação C: FAIL — CSVs diferem para mesma seed!")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = load_all_configs("configs")

    validation_a(OUT_DIR)
    validation_b(OUT_DIR)
    validation_c(cfgs)

    logger.info("Validação concluída. PNGs em %s", OUT_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **F3: Rodar validação**

```bash
python scripts/validate_scenarios.py
```
Esperado: 4 PNGs em `results/scenario_validation/`, `Validação C: PASS`.

- [ ] **F4: Commit**

```bash
git add scripts/validate_scenarios.py results/scenario_validation/
git commit -m "feat(viz): script validate_scenarios.py com 3 validacoes visuais"
```

---

### Task G: Testes unitários do gerador

**Arquivos:**
- Create: `tests/test_scenario_generator.py`

- [ ] **G1: Escrever `tests/test_scenario_generator.py`**

```python
"""Testes unitários do ScenarioGenerator."""

from __future__ import annotations

import filecmp
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs


@pytest.fixture(scope="module")
def cfgs():
    return load_all_configs("configs")


@pytest.fixture(scope="module")
def generator(cfgs):
    return ScenarioGenerator(cfgs["scenarios"], cfgs["config"]["simulation"])


@pytest.fixture(scope="module")
def sample_csv(generator, tmp_path_factory):
    out = tmp_path_factory.mktemp("scenarios")
    csv_path, _ = generator.generate("equilibrado", "util", seed=42, output_dir=out)
    return csv_path


class TestReproducibility:
    def test_same_seed_produces_identical_csv(self, generator):
        with tempfile.TemporaryDirectory() as tmp:
            dir1, dir2 = Path(tmp) / "a", Path(tmp) / "b"
            csv1, _ = generator.generate("equilibrado", "util", seed=1234, output_dir=dir1)
            csv2, _ = generator.generate("equilibrado", "util", seed=1234, output_dir=dir2)
            assert filecmp.cmp(csv1, csv2, shallow=False), "CSVs diferem para mesma seed"

    def test_different_seeds_produce_different_csvs(self, generator):
        with tempfile.TemporaryDirectory() as tmp:
            dir1, dir2 = Path(tmp) / "a", Path(tmp) / "b"
            csv1, _ = generator.generate("equilibrado", "util", seed=1, output_dir=dir1)
            csv2, _ = generator.generate("equilibrado", "util", seed=2, output_dir=dir2)
            assert not filecmp.cmp(csv1, csv2, shallow=False), "CSVs idênticos para seeds diferentes"


class TestCsvStructure:
    def test_shape(self, sample_csv):
        df = pd.read_csv(sample_csv)
        assert df.shape == (17280, 9), f"Shape inesperado: {df.shape}"

    def test_columns(self, sample_csv):
        df = pd.read_csv(sample_csv)
        expected = {"tick", "hora", "minuto", "day_type", "time_band", "family", "veh_ns", "ped_l", "ped_o"}
        assert set(df.columns) == expected

    def test_arrivals_non_negative(self, sample_csv):
        df = pd.read_csv(sample_csv)
        for col in ("veh_ns", "ped_l", "ped_o"):
            assert (df[col] >= 0).all(), f"Coluna {col} tem valores negativos"

    def test_tick_sequence(self, sample_csv):
        df = pd.read_csv(sample_csv)
        assert list(df["tick"]) == list(range(17280))

    def test_metadata_fields(self, generator):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            _, json_path = generator.generate("pico_veic", "fds", seed=99, output_dir=Path(tmp))
            meta = json.loads(json_path.read_text())
        required = {"seed", "family", "day_type", "duration_ticks", "generator_version", "timestamp", "base_rates_by_band"}
        assert required.issubset(meta.keys())
        assert meta["seed"] == 99
        assert meta["family"] == "pico_veic"
        assert meta["day_type"] == "fds"
        assert meta["duration_ticks"] == 17280


class TestTimeBand:
    @pytest.mark.parametrize("tick,expected_band", [
        (0, "madrugada"),         # hora 0
        (720, "manha_tranquila"), # hora 1 ... wait: tick 720 = 720*5s = 3600s = 1h → madrugada
        # Recalcular: madrugada [0,5), então tick 720 = 1h → madrugada
        # tick para 5h: 5*3600/5 = 3600
        (3600, "manha_tranquila"),  # 5h exato → manha_tranquila
        (5040, "pico_manha"),       # 7h: 7*3600/5 = 5040
        (7200, "meio_manha"),       # 10h: 10*3600/5 = 7200
        (8640, "pico_tarde"),       # 12h: 12*3600/5 = 8640
        (10080, "tarde"),           # 14h: 14*3600/5 = 10080
        (12240, "pico_noite"),      # 17h: 17*3600/5 = 12240
        (14400, "noite"),           # 20h: 20*3600/5 = 14400
        (17279, "noite"),           # último tick
    ])
    def test_tick_to_time_band(self, generator, tick, expected_band):
        result = generator._tick_to_time_band(tick)
        assert result == expected_band, f"tick={tick}: esperado '{expected_band}', obtido '{result}'"


class TestFamilyMultipliers:
    def test_pico_veic_has_more_vehicles_than_equilibrado(self, generator):
        """pico_veic deve produzir mais veículos que equilibrado em média."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_eq, _ = generator.generate("equilibrado", "util", seed=0, output_dir=tmp / "eq")
            csv_pv, _ = generator.generate("pico_veic", "util", seed=0, output_dir=tmp / "pv")
            df_eq = pd.read_csv(csv_eq)
            df_pv = pd.read_csv(csv_pv)
            assert df_pv["veh_ns"].mean() > df_eq["veh_ns"].mean()

    def test_pico_ped_has_more_pedestrians_than_equilibrado(self, generator):
        """pico_ped deve produzir mais pedestres que equilibrado em média."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_eq, _ = generator.generate("equilibrado", "util", seed=0, output_dir=tmp / "eq")
            csv_pp, _ = generator.generate("pico_ped", "util", seed=0, output_dir=tmp / "pp")
            df_eq = pd.read_csv(csv_eq)
            df_pp = pd.read_csv(csv_pp)
            assert df_pp["ped_l"].mean() > df_eq["ped_l"].mean()

    def test_baixa_mov_has_less_traffic_than_equilibrado(self, generator):
        """baixa_mov deve produzir menos tráfego que equilibrado."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_eq, _ = generator.generate("equilibrado", "util", seed=0, output_dir=tmp / "eq")
            csv_bm, _ = generator.generate("baixa_mov", "util", seed=0, output_dir=tmp / "bm")
            df_eq = pd.read_csv(csv_eq)
            df_bm = pd.read_csv(csv_bm)
            assert df_bm["veh_ns"].mean() < df_eq["veh_ns"].mean()
            assert df_bm["ped_l"].mean() < df_eq["ped_l"].mean()

    def test_util_pico_manha_higher_than_util_madrugada(self, generator):
        """pico_manha deve ter mais veículos que madrugada no mesmo cenário."""
        with tempfile.TemporaryDirectory() as tmp:
            csv, _ = generator.generate("equilibrado", "util", seed=0, output_dir=Path(tmp))
            df = pd.read_csv(csv)
            mean_pico = df[df["time_band"] == "pico_manha"]["veh_ns"].mean()
            mean_mad = df[df["time_band"] == "madrugada"]["veh_ns"].mean()
            assert mean_pico > mean_mad
```

- [ ] **G2: Rodar testes**

```bash
python -m pytest tests/test_scenario_generator.py -v
```
Esperado: todos PASS.

- [ ] **G3: Rodar suite completa**

```bash
python -m pytest tests/ -v
```
Esperado: todos PASS.

- [ ] **G4: Commit**

```bash
git add tests/test_scenario_generator.py
git commit -m "test: testes unitarios do gerador (reproducibilidade, faixa horaria, familia)"
```

---

### Task H: Atualizar §12 do INSTRUCOES

**Arquivos:**
- Modify: `INSTRUCOES_SEMAFORO_INTELIGENTE.md`

- [ ] **H1: Adicionar entrada na tabela de §12**

Adicionar linha na tabela de histórico de revisões:
```
| 2026-05-13 | Fase 3 implementada. Gerador de cenários de duas camadas com Poisson. 50 cenários de treino (5 famílias × 2 day_types × 5 seeds) + 32 de avaliação (4 famílias × 2 day_types × 4 seeds, imprevisivel excluído). Métricas refatoradas para registro configurável via config.yaml. Benchmark baseline gerado em results/benchmark_baseline.csv e .md. Validações visuais (A/B/C) em results/scenario_validation/. |
```

- [ ] **H2: Commit**

```bash
git add INSTRUCOES_SEMAFORO_INTELIGENTE.md
git commit -m "docs: atualiza §12 do INSTRUCOES com avanco da fase 3"
```

---

## Self-Review — Cobertura do Briefing

| Requisito | Coberto em |
|---|---|
| Duas camadas (perfil-base + família) | Task B — `ScenarioGenerator._compute_lambda` |
| 8 faixas horárias com mapeamento correto [0-5), [5-7), ... | Task A (YAML) + Task B (`_tick_to_time_band`) |
| 5 famílias com multiplicadores do briefing | Task A (YAML) |
| Variabilidade Poisson com ruído N(0, 0.1×var_mult) | Task B — `_compute_lambda` |
| Clipagem em [0,∞) | Task B — `max(0.0, lam)` |
| CSV com todas as colunas exigidas | Task B — dict de row |
| JSON com todos os campos exigidos + base_rates_by_band | Task B |
| Reprodutibilidade byte a byte | Task B (rng seedado) + Task G (teste) |
| 50 cenários de treino (seeds 100-104) | Task A (config) + Task C (script) |
| ~30 cenários de avaliação (32 proposto) | Task A (config) + Task C (script) |
| Idempotência com --force | Task C |
| tqdm nas gerações longas | Task C |
| 13 métricas via registry + YAML | Task D |
| Registro por decorator | Task D — `@metric(name)` |
| Benchmark CSV + MD | Task E |
| Benchmark agrupa por família e tipo de dia | Task E — `build_md_summary` |
| Validação A (curva temporal) | Task F |
| Validação B (barplot famílias) | Task F |
| Validação C (reprodutibilidade) | Task F |
| PNGs em alta resolução | Task F — `dpi=150` |
| Testes de reprodutibilidade | Task G |
| Testes de faixa horária | Task G — `TestTimeBand` |
| Testes de família | Task G — `TestFamilyMultipliers` |
| §12 atualizado | Task H |
| `crossing.py` expõe `served_this_tick` | Task D |
| requirements.txt com tqdm | Task C |
| Logs estruturados (logging) | Todos os scripts |

## Nota — Contagem de cenários de avaliação

O briefing sugere "30 exatos" mas 4×2×4=**32**. Este plano usa 32 e atualiza `evaluation.num_scenarios: 32` no config.yaml. Se o usuário quiser exatamente 30, basta remover 1 seed de 2 famílias (`seeds: [1000, 1001, 1002]` em 2 famílias e `seeds: [1000, 1001, 1002, 1003]` nas outras 2) — isso requer confirmação antes de implementar.

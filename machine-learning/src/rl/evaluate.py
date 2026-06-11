"""
Avaliação comparativa do agente PPO treinado — Fase 6.

Roda o agente treinado nos 32 cenários de avaliação e compara os
resultados com o baseline de tempo fixo (results/benchmark_baseline.csv).

Como usar:
    python scripts/evaluate_agent.py

Saídas geradas em results/:
    agent_results.csv       — métricas do agente por cenário
    comparison.csv          — comparação direta agente vs baseline
    relatorio.md            — relatório final com gráficos e análise
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.simulation.crossing import Crossing
from src.simulation.metrics import compute_metrics
from src.utils.config_loader import load_all_configs

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    "espera_media_carros",
    "espera_media_pedestres",
    "espera_maxima_carros",
    "espera_maxima_pedestres",
    "espera_p95_carros",
    "espera_p95_pedestres",
    "violacoes_teto_carros",
    "violacoes_teto_pedestres",
    "fila_media_veh_ns",
    "fila_media_ped_l",
    "fila_media_ped_o",
    "throughput_total_carros",
    "throughput_total_pedestres",
]

# Denominadores de normalização — iguais aos do TrafficLightEnv
_NORM = {
    "veh_ns":         30.0,
    "ped_l":          15.0,
    "ped_o":          15.0,
    "ticks_in_phase": 36.0,
    "max_wait_car_s": 90.0,
    "max_wait_ped_s": 60.0,
}
_TICK_S = 5


def _parse_stem(stem: str) -> tuple[str, str, int]:
    """
    Extrai (family, day_type, seed) do nome-base do arquivo.
    Formato: {family}_{day_type}_seed{NNNN}
    """
    left, seed_str = stem.rsplit("_seed", 1)
    seed = int(seed_str)
    family_part, day_type = left.rsplit("_", 1)
    return family_part, day_type, seed


def _state_to_obs(state: dict) -> np.ndarray:
    """
    Converte o state dict do crossing no vetor de observação do agente.

    Mesmo formato usado no TrafficLightEnv._get_obs().

    Parâmetros
    ----------
    state : dict
        Retorno de Crossing.get_state().

    Retorna
    -------
    obs : np.ndarray, shape (1, 7), dtype float32
        Vetor normalizado pronto para model.predict().
    """
    max_wait_car_s = state["veh_ns"]["max_wait_ticks"] * _TICK_S
    max_wait_ped_s = max(
        state["ped_l"]["max_wait_ticks"],
        state["ped_o"]["max_wait_ticks"],
    ) * _TICK_S

    obs = np.array([
        state["veh_ns"]["size"]  / _NORM["veh_ns"],
        state["ped_l"]["size"]   / _NORM["ped_l"],
        state["ped_o"]["size"]   / _NORM["ped_o"],
        float(state["phase"] == "B"),
        state["ticks_in_phase"]  / _NORM["ticks_in_phase"],
        max_wait_car_s           / _NORM["max_wait_car_s"],
        max_wait_ped_s           / _NORM["max_wait_ped_s"],
    ], dtype=np.float32).reshape(1, -1)

    return obs.clip(0.0, 1.0)


def _run_scenario_agent(
    csv_path: Path,
    crossing: Crossing,
    model: PPO,
) -> list[dict]:
    """
    Roda o agente PPO em um cenário CSV completo (dia inteiro).

    Parâmetros
    ----------
    csv_path : Path
        Caminho para o CSV do cenário.
    crossing : Crossing
        Instância da engine de simulação (será resetada).
    model : PPO
        Modelo treinado.

    Retorna
    -------
    tick_history : list[dict]
        Estado do cruzamento após cada tick — mesmo formato do benchmark.
    """
    crossing.reset()
    tick_history: list[dict] = []
    state = crossing.get_state()

    df = pd.read_csv(csv_path)
    for row in df.itertuples(index=False):
        arrivals = {
            "veh_ns": int(row.veh_ns),
            "ped_l":  int(row.ped_l),
            "ped_o":  int(row.ped_o),
        }
        obs    = _state_to_obs(state)
        action, _ = model.predict(obs, deterministic=True)
        state  = crossing.step(arrivals, int(action.item()))
        tick_history.append(state)

    return tick_history


def _fmt(val: float) -> str:
    """Formata float para exibição."""
    return f"{val:.1f}"


def _pct(a: float, b: float) -> str:
    """Calcula variação percentual de b em relação a a."""
    if a == 0:
        return "—"
    return f"{((b - a) / a * 100):+.1f}%"


def _build_relatorio(
    df_agent: pd.DataFrame,
    df_base: pd.DataFrame,
    df_comp: pd.DataFrame,
) -> str:
    """
    Monta o relatório Markdown final comparando agente vs baseline.

    Parâmetros
    ----------
    df_agent : DataFrame
        Métricas do agente por cenário.
    df_base : DataFrame
        Métricas do baseline por cenário (apenas eval).
    df_comp : DataFrame
        Comparação direta agente vs baseline.

    Retorna
    -------
    md : str
        Conteúdo do relatório em Markdown.
    """
    n = len(df_agent)

    # Métricas principais para o resumo
    metricas_principais = [
        ("espera_media_carros",      "Espera média carros (s)"),
        ("espera_media_pedestres",   "Espera média pedestres (s)"),
        ("espera_maxima_carros",     "Espera máxima carros (s)"),
        ("espera_maxima_pedestres",  "Espera máxima pedestres (s)"),
        ("violacoes_teto_carros",    "Violações teto carros"),
        ("violacoes_teto_pedestres", "Violações teto pedestres"),
    ]

    # ── Tabela de resumo geral ─────────────────────────────────────────────
    header_resumo = "| Métrica | Baseline | Agente PPO | Variação |"
    sep_resumo    = "|---|---|---|---|"
    linhas_resumo = [header_resumo, sep_resumo]
    for col, label in metricas_principais:
        b = df_base[col].mean()
        a = df_agent[col].mean()
        linhas_resumo.append(f"| {label} | {_fmt(b)} | {_fmt(a)} | {_pct(b, a)} |")

    # ── Tabela por família ─────────────────────────────────────────────────
    familias = sorted(df_agent["family"].unique())
    header_fam = "| Família | Esp.med.carros base | Esp.med.carros agente | Var. | Viol.base | Viol.agente | Var. |"
    sep_fam    = "|---|---|---|---|---|---|---|"
    linhas_fam = [header_fam, sep_fam]
    for fam in familias:
        ag = df_agent[df_agent["family"] == fam]
        ba = df_base[df_base["family"] == fam]
        if ba.empty:
            continue
        emc_b = ba["espera_media_carros"].mean()
        emc_a = ag["espera_media_carros"].mean()
        vio_b = ba["violacoes_teto_carros"].mean() + ba["violacoes_teto_pedestres"].mean()
        vio_a = ag["violacoes_teto_carros"].mean() + ag["violacoes_teto_pedestres"].mean()
        linhas_fam.append(
            f"| {fam} | {_fmt(emc_b)} | {_fmt(emc_a)} | {_pct(emc_b, emc_a)} | "
            f"{_fmt(vio_b)} | {_fmt(vio_a)} | {_pct(vio_b, vio_a)} |"
        )

    # ── Totais de violações ────────────────────────────────────────────────
    viol_base_total  = int(df_base["violacoes_teto_carros"].sum()  + df_base["violacoes_teto_pedestres"].sum())
    viol_agent_total = int(df_agent["violacoes_teto_carros"].sum() + df_agent["violacoes_teto_pedestres"].sum())
    reducao_viol = _pct(viol_base_total, viol_agent_total)

    # ── Conclusão automática ───────────────────────────────────────────────
    esp_med_car_base  = df_base["espera_media_carros"].mean()
    esp_med_car_agent = df_agent["espera_media_carros"].mean()
    melhorou = esp_med_car_agent < esp_med_car_base

    if melhorou:
        conclusao = (
            f"O agente PPO superou o semáforo de tempo fixo em espera média de carros "
            f"({_pct(esp_med_car_base, esp_med_car_agent)}), com redução de violações de teto "
            f"de {viol_base_total} para {viol_agent_total} ({reducao_viol})."
        )
    else:
        conclusao = (
            f"O agente PPO não superou o semáforo de tempo fixo em espera média de carros "
            f"({_pct(esp_med_car_base, esp_med_car_agent)}). "
            f"Considere aumentar o tempo de treinamento ou ajustar os pesos da recompensa."
        )

    md = f"""# Relatório Final — Semáforo Inteligente

> **Agente:** PPO (Stable-Baselines3)  
> **Cenários de avaliação:** {n}  
> **Tetos:** carros ≤ 90 s | pedestres ≤ 60 s  

---

## 1. Resumo executivo

{conclusao}

---

## 2. Comparação geral — agente vs baseline

{chr(10).join(linhas_resumo)}

---

## 3. Comparação por família de cenário

{chr(10).join(linhas_fam)}

---

## 4. Violações de teto de espera

| | Baseline | Agente PPO | Variação |
|---|---|---|---|
| Violações carros | {int(df_base['violacoes_teto_carros'].sum())} | {int(df_agent['violacoes_teto_carros'].sum())} | {_pct(df_base['violacoes_teto_carros'].sum(), df_agent['violacoes_teto_carros'].sum())} |
| Violações pedestres | {int(df_base['violacoes_teto_pedestres'].sum())} | {int(df_agent['violacoes_teto_pedestres'].sum())} | {_pct(df_base['violacoes_teto_pedestres'].sum(), df_agent['violacoes_teto_pedestres'].sum())} |
| **Total** | **{viol_base_total}** | **{viol_agent_total}** | **{reducao_viol}** |

---

## 5. Arquivos gerados

- `results/agent_results.csv` — métricas do agente por cenário
- `results/comparison.csv` — comparação direta por cenário
- `results/relatorio.md` — este relatório

---

*Gerado automaticamente pelo módulo de avaliação da Fase 6.*
"""
    return md


def avaliar(
    model_path: str = "models/ppo_semaforo_final.zip",
    configs_dir: str = "configs",
    scenarios_eval_dir: str = "scenarios/eval",
    baseline_path: str = "results/benchmark_baseline.csv",
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Avalia o agente PPO nos cenários de avaliação e gera o relatório comparativo.

    Parâmetros
    ----------
    model_path : str
        Caminho para o modelo treinado.
    configs_dir : str
        Pasta com os arquivos de configuração.
    scenarios_eval_dir : str
        Pasta com os cenários de avaliação.
    baseline_path : str
        CSV com as métricas do baseline de tempo fixo.
    results_dir : str
        Pasta onde salvar os resultados.

    Retorna
    -------
    df_comp : pd.DataFrame
        DataFrame com a comparação direta agente vs baseline.
    """
    # ── Carrega configurações ──────────────────────────────────────────────
    cfgs     = load_all_configs(configs_dir)
    cfg      = cfgs["config"]
    rl_cfg   = cfgs["rl"]
    sim_cfg  = cfg["simulation"]
    tick_s   = sim_cfg["tick_seconds"]

    # ── Carrega modelo ─────────────────────────────────────────────────────
    model_p = Path(model_path)
    if not model_p.exists():
        best = Path("models/best/best_model.zip")
        if best.exists():
            model_p = best
            logger.warning("Modelo final não encontrado. Usando melhor modelo: %s", best)
        else:
            raise FileNotFoundError(
                f"Nenhum modelo encontrado em '{model_path}'. "
                "Execute scripts/train_agent.py primeiro."
            )

    logger.info("Carregando modelo: %s", model_p)
    model = PPO.load(str(model_p))

    # ── Carrega cenários de avaliação ──────────────────────────────────────
    eval_dir    = Path(scenarios_eval_dir)
    csv_paths   = sorted(eval_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"Nenhum cenário CSV encontrado em '{scenarios_eval_dir}'. "
            "Execute scripts/generate_scenarios.py primeiro."
        )
    logger.info("%d cenários de avaliação encontrados.", len(csv_paths))

    # ── Roda o agente em cada cenário ──────────────────────────────────────
    crossing = Crossing(sim_cfg)
    results: list[dict] = []

    print(f"\nAvaliando agente em {len(csv_paths)} cenários...\n")
    for i, csv_path in enumerate(csv_paths, 1):
        family, day_type, seed = _parse_stem(csv_path.stem)
        print(f"  [{i:2d}/{len(csv_paths)}] {csv_path.name}", end=" ... ", flush=True)

        tick_history = _run_scenario_agent(csv_path, crossing, model)
        metrics      = compute_metrics(METRIC_NAMES, tick_history, tick_s, rl_cfg)

        results.append({
            "scenario_id": csv_path.stem,
            "family":      family,
            "day_type":    day_type,
            "seed":        seed,
            **metrics,
        })
        print("OK")

    # ── Salva resultados do agente ─────────────────────────────────────────
    out_dir = Path(results_dir)
    out_dir.mkdir(exist_ok=True)

    df_agent = pd.DataFrame(results)
    cols     = ["scenario_id", "family", "day_type", "seed"] + METRIC_NAMES
    df_agent = df_agent[cols]

    agent_csv = out_dir / "agent_results.csv"
    df_agent.to_csv(agent_csv, index=False)
    logger.info("Resultados do agente salvos: %s", agent_csv)

    # ── Carrega baseline e faz comparação ─────────────────────────────────
    base_p = Path(baseline_path)
    if not base_p.exists():
        raise FileNotFoundError(
            f"Baseline não encontrado em '{baseline_path}'. "
            "Execute scripts/run_benchmark.py primeiro."
        )

    df_base_full = pd.read_csv(base_p)
    df_base      = df_base_full[df_base_full["set"] == "eval"].copy()

    if df_base.empty:
        logger.warning("Baseline não tem cenários de avaliação (set='eval'). Usando todos.")
        df_base = df_base_full.copy()

    # Comparação por cenário: merge nos campos comuns
    df_comp = df_agent.merge(
        df_base[["scenario_id"] + METRIC_NAMES],
        on="scenario_id",
        suffixes=("_agente", "_baseline"),
        how="inner",
    )

    # Adiciona colunas de variação percentual para métricas principais
    for m in ["espera_media_carros", "espera_media_pedestres",
              "espera_maxima_carros", "violacoes_teto_carros", "violacoes_teto_pedestres"]:
        col_base  = f"{m}_baseline"
        col_agent = f"{m}_agente"
        df_comp[f"{m}_variacao_pct"] = (
            (df_comp[col_agent] - df_comp[col_base])
            / df_comp[col_base].replace(0, np.nan)
            * 100
        ).round(1)

    comp_csv = out_dir / "comparison.csv"
    df_comp.to_csv(comp_csv, index=False)
    logger.info("Comparação salva: %s", comp_csv)

    # ── Gera relatório Markdown ────────────────────────────────────────────
    md_content = _build_relatorio(df_agent, df_base, df_comp)
    md_path    = out_dir / "relatorio.md"
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Relatório salvo: %s", md_path)

    # ── Imprime resumo no terminal ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AVALIAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"\n  Cenários avaliados : {len(df_agent)}")

    for col, label in [
        ("espera_media_carros",      "Esp. média carros (s) "),
        ("espera_media_pedestres",   "Esp. média peds   (s) "),
        ("espera_maxima_carros",     "Esp. máxima carros (s)"),
        ("violacoes_teto_carros",    "Violações carros      "),
        ("violacoes_teto_pedestres", "Violações pedestres   "),
    ]:
        b = df_base[col].mean()
        a = df_agent[col].mean()
        sinal = "✓" if a <= b else "✗"
        print(f"  {sinal}  {label}: baseline={_fmt(b)}  agente={_fmt(a)}  ({_pct(b, a)})")

    print(f"\n  Relatório completo : {md_path}")
    print("=" * 60 + "\n")

    return df_comp

"""
Avalia o DualAgent nos 32 cenários de avaliação e compara com o baseline.

Como usar:
    python scripts/evaluate_dual_agent.py

Requisitos:
    - models/ppo_semaforo_final.zip      (agente geral)
    - models/ppo_pico_veic_final.zip     (agente especialista)
    - results/benchmark_baseline.csv     (baseline)
    - scenarios/eval/*.csv               (cenários de avaliação)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

import pandas as pd
import numpy as np

from src.simulation.crossing import Crossing
from src.simulation.metrics import compute_metrics
from src.rl.dual_agent import DualAgent
from src.utils.config_loader import load_all_configs

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


def _parse_stem(stem: str) -> tuple[str, str, int]:
    left, seed_str = stem.rsplit("_seed", 1)
    seed = int(seed_str)
    family_part, day_type = left.rsplit("_", 1)
    return family_part, day_type, seed


def fmt(v: float) -> str:
    return f"{v:.1f}"


def pct(b: float, a: float) -> str:
    if b == 0:
        return "—"
    p = (a - b) / b * 100
    sinal = "+" if p > 0 else ""
    return f"{sinal}{p:.1f}%"


def main() -> None:
    print("\n" + "=" * 60)
    print("  Avaliação — DualAgent (geral + especialista pico_veic)")
    print("=" * 60)

    # Verifica pré-requisitos
    erros = []
    if not Path("models/ppo_semaforo_final.zip").exists():
        erros.append("models/ppo_semaforo_final.zip não encontrado")
    if not Path("models/ppo_pico_veic_final.zip").exists():
        erros.append("models/ppo_pico_veic_final.zip não encontrado")
        erros.append("  → Execute: python scripts/train_agent_pico_veic.py")
    if not Path("results/benchmark_baseline.csv").exists():
        erros.append("results/benchmark_baseline.csv não encontrado")
    if not any(Path("scenarios/eval").glob("*.csv")):
        erros.append("scenarios/eval/ sem cenários CSV")

    if erros:
        print("\nERRO — pré-requisitos não atendidos:")
        for e in erros:
            print(f"  {e}")
        sys.exit(1)

    # Carrega configurações
    cfgs    = load_all_configs("configs")
    cfg     = cfgs["config"]
    rl_cfg  = cfgs["rl"]
    sim_cfg = cfg["simulation"]
    tick_s  = sim_cfg["tick_seconds"]

    # Carrega DualAgent
    print("\nCarregando agentes...")
    agente = DualAgent()

    # Cenários de avaliação
    csv_paths = sorted(Path("scenarios/eval").glob("*.csv"))
    crossing  = Crossing(sim_cfg)
    results   = []

    print(f"\nAvaliando DualAgent em {len(csv_paths)} cenários...\n")

    for i, csv_path in enumerate(csv_paths, 1):
        family, day_type, seed = _parse_stem(csv_path.stem)
        print(f"  [{i:2d}/{len(csv_paths)}] {csv_path.name}", end=" ... ", flush=True)

        crossing.reset()
        tick_history = []
        state = crossing.get_state()
        df = pd.read_csv(csv_path)

        for row in df.itertuples(index=False):
            arrivals = {
                "veh_ns": int(row.veh_ns),
                "ped_l":  int(row.ped_l),
                "ped_o":  int(row.ped_o),
            }
            action = agente.decide(state)
            state  = crossing.step(arrivals, action)
            tick_history.append(state)

        metrics = compute_metrics(METRIC_NAMES, tick_history, tick_s, rl_cfg)
        results.append({
            "scenario_id": csv_path.stem,
            "family":      family,
            "day_type":    day_type,
            "seed":        seed,
            **metrics,
        })
        print("OK")

    # Salva resultados
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df_dual = pd.DataFrame(results)
    df_dual.to_csv(out_dir / "dual_agent_results.csv", index=False)

    # Carrega baseline e agente geral para comparação
    df_base  = pd.read_csv("results/benchmark_baseline.csv")
    df_base  = df_base[df_base["set"] == "eval"].copy()

    df_geral = None
    if Path("results/agent_results.csv").exists():
        df_geral = pd.read_csv("results/agent_results.csv")

    # ── Resumo no terminal ────────────────────────────────────────────────────
    metricas = [
        ("espera_media_carros",      "Esp. média carros (s) "),
        ("espera_media_pedestres",   "Esp. média peds   (s) "),
        ("espera_maxima_carros",     "Esp. máxima carros (s)"),
        ("violacoes_teto_carros",    "Violações carros      "),
        ("violacoes_teto_pedestres", "Violações pedestres   "),
    ]

    print("\n" + "=" * 72)
    print("  RESULTADOS COMPARATIVOS")
    print("=" * 72)

    if df_geral is not None:
        print(f"  {'Métrica':<26} {'Baseline':>10} {'Ag.Geral':>10} {'DualAgent':>10} {'Melhora':>8}")
        print("  " + "-" * 66)
        for col, label in metricas:
            b  = df_base[col].mean()
            g  = df_geral[col].mean()
            d  = df_dual[col].mean()
            sinal = "✓" if d <= b else "✗"
            melhor_que_geral = "↑" if d < g else ("=" if abs(d-g)<0.1 else "↓")
            print(f"  {sinal}  {label} {fmt(b):>10} {fmt(g):>10} {fmt(d):>10} {melhor_que_geral:>8} {pct(b,d):>8}")
    else:
        print(f"  {'Métrica':<26} {'Baseline':>10} {'DualAgent':>10} {'Variação':>10}")
        print("  " + "-" * 58)
        for col, label in metricas:
            b = df_base[col].mean()
            d = df_dual[col].mean()
            sinal = "✓" if d <= b else "✗"
            print(f"  {sinal}  {label} {fmt(b):>10} {fmt(d):>10} {pct(b,d):>10}")

    # Por família
    print("\n  POR FAMÍLIA:")
    print(f"  {'Família':<15} {'Base carros':>12} {'Dual carros':>12} {'Var.':>8} {'Base viol.':>11} {'Dual viol.':>11} {'Var.':>8}")
    print("  " + "-" * 80)
    for fam in sorted(df_dual["family"].unique()):
        b_sub = df_base[df_base["family"]==fam]
        d_sub = df_dual[df_dual["family"]==fam]
        if b_sub.empty:
            continue
        bc = b_sub["espera_media_carros"].mean()
        dc = d_sub["espera_media_carros"].mean()
        bv = b_sub["violacoes_teto_carros"].mean()
        dv = d_sub["violacoes_teto_carros"].mean()
        sinal = "✓" if dc <= bc else "✗"
        print(f"  {sinal}  {fam:<13} {fmt(bc):>12} {fmt(dc):>12} {pct(bc,dc):>8} {fmt(bv):>11} {fmt(dv):>11} {pct(bv,dv):>8}")

    # Stats do DualAgent
    stats = agente.stats
    print(f"\n  Uso dos agentes ao longo da avaliação:")
    print(f"    Agente geral      : {stats['ticks_geral']:,} ticks ({100-stats['pct_especialista']:.1f}%)")
    print(f"    Agente especialista: {stats['ticks_especialista']:,} ticks ({stats['pct_especialista']:.1f}%)")

    print(f"\n  Resultados salvos: results/dual_agent_results.csv")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

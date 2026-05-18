"""Roda o benchmark de tempo fixo em todos os cenários e salva métricas baseline."""

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
    """
    Extrai (family, day_type, seed) do nome-base do arquivo.
    Formato: {family}_{day_type}_seed{NNNN}
    """
    left, seed_str = stem.rsplit("_seed", 1)
    seed = int(seed_str)
    family_part, day_type = left.rsplit("_", 1)
    return family_part, day_type, seed


def _run_scenario(
    csv_path: Path,
    crossing: Crossing,
    controller: FixedTimeController,
) -> list[dict]:
    """Roda o controlador fixo em um cenário CSV. Retorna tick_history."""
    crossing.reset()
    tick_history: list[dict] = []
    state = crossing.get_state()

    df = pd.read_csv(csv_path)
    for row in df.itertuples(index=False):
        arrivals = {
            "veh_ns": int(row.veh_ns),
            "ped_l": int(row.ped_l),
            "ped_o": int(row.ped_o),
        }
        action = controller.decide(state)
        state = crossing.step(arrivals, action)
        tick_history.append(state)

    return tick_history


def _fmt(val: float) -> str:
    """Formata float para exibição: sem casas para inteiros, 1 casa para decimais."""
    return f"{val:.1f}"


def _md_grouped_table(df: pd.DataFrame, group_col: str, metric_names: list[str]) -> str:
    """Gera tabela markdown com média ± desvio-padrão agrupado por `group_col`."""
    groups = sorted(df[group_col].unique())

    # Cabeçalho
    header = f"| {group_col} |"
    sep = "|---|"
    for m in metric_names:
        short = m.replace("espera_", "esp_").replace("violacoes_", "viol_").replace(
            "fila_media_", "fila_").replace("throughput_total_", "tp_")
        header += f" {short} |"
        sep += "---|"

    lines = [header, sep]
    for g in groups:
        sub = df[df[group_col] == g]
        row = f"| {g} |"
        for m in metric_names:
            mean = sub[m].mean()
            std = sub[m].std()
            row += f" {_fmt(mean)} ±{_fmt(std)} |"
        lines.append(row)

    return "\n".join(lines)


def _md_overall_table(df: pd.DataFrame, metric_names: list[str]) -> str:
    """Gera tabela markdown com média ± desvio-padrão global."""
    header = "| métrica | média | std | mín | máx |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for m in metric_names:
        mean = df[m].mean()
        std = df[m].std()
        mn = df[m].min()
        mx = df[m].max()
        lines.append(f"| {m} | {_fmt(mean)} | {_fmt(std)} | {_fmt(mn)} | {_fmt(mx)} |")
    return "\n".join(lines)


def _build_markdown(df: pd.DataFrame, metric_names: list[str]) -> str:
    """Monta o relatório Markdown completo."""
    lines: list[str] = [
        "# Benchmark de Tempo Fixo — Baseline",
        "",
        f"> Controlador: Fase A = 9 ticks (45 s) / Fase B = 5 ticks (25 s)  ",
        f"> Cenários: {len(df)} total  "
        f"({(df['set']=='train').sum()} treino + {(df['set']=='eval').sum()} avaliação)  ",
        f"> Tetos: carros ≤ 90 s | pedestres ≤ 60 s",
        "",
        "---",
        "",
        "## Resumo geral",
        "",
        _md_overall_table(df, metric_names),
        "",
        "---",
        "",
        "## Por família",
        "",
        _md_grouped_table(df, "family", metric_names),
        "",
        "---",
        "",
        "## Por tipo de dia",
        "",
        _md_grouped_table(df, "day_type", metric_names),
        "",
        "---",
        "",
        "## Por conjunto (treino vs avaliação)",
        "",
        _md_grouped_table(df, "set", metric_names),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cfgs = load_all_configs("configs")
    config = cfgs["config"]
    rl_cfg = cfgs["rl"]
    sim_cfg = config["simulation"]
    ft_cfg = config["fixed_time_controller"]
    gen_cfg = config["scenario_generation"]
    tick_seconds: int = sim_cfg["tick_seconds"]

    crossing = Crossing(sim_cfg)
    controller = FixedTimeController(ft_cfg["phase_a_ticks"], ft_cfg["phase_b_ticks"])

    sets = [
        ("train", gen_cfg["training"]["output_dir"]),
        ("eval", gen_cfg["evaluation"]["output_dir"]),
    ]

    results: list[dict] = []

    for set_name, output_dir in sets:
        csv_dir = Path(output_dir)
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(
                "[%s] Nenhum CSV em %s — rode generate_scenarios.py primeiro.",
                set_name, csv_dir,
            )
            continue

        logger.info("[%s] %d cenários encontrados em %s", set_name, len(csv_files), csv_dir)

        for csv_path in tqdm(csv_files, desc=f"Benchmark [{set_name}]", unit="cen"):
            family, day_type, seed = _parse_stem(csv_path.stem)
            tick_history = _run_scenario(csv_path, crossing, controller)
            metrics = compute_metrics(METRIC_NAMES, tick_history, tick_seconds, rl_cfg)

            results.append({
                "scenario_id": csv_path.stem,
                "set": set_name,
                "family": family,
                "day_type": day_type,
                "seed": seed,
                **metrics,
            })

    if not results:
        logger.error("Nenhum cenário processado. Abortando.")
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    cols = ["scenario_id", "set", "family", "day_type", "seed"] + METRIC_NAMES
    df = df[cols]

    csv_out = results_dir / "benchmark_baseline.csv"
    df.to_csv(csv_out, index=False)
    logger.info("CSV salvo: %s (%d linhas)", csv_out, len(df))

    md_content = _build_markdown(df, METRIC_NAMES)
    md_out = results_dir / "benchmark_baseline.md"
    md_out.write_text(md_content, encoding="utf-8")
    logger.info("Relatório salvo: %s", md_out)

    # Resumo rápido de violações para o terminal
    viol_carros = int(df["violacoes_teto_carros"].sum())
    viol_peds = int(df["violacoes_teto_pedestres"].sum())
    logger.info(
        "Violações de teto — carros: %d | pedestres: %d | total: %d",
        viol_carros, viol_peds, viol_carros + viol_peds,
    )


if __name__ == "__main__":
    main()

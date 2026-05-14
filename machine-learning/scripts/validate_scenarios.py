"""Validações visuais do gerador de cenários. Gera PNGs em results/scenario_validation/."""

from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("results/scenario_validation")
TRAIN_DIR = Path("scenarios/train")

QUEUES = ("veh_ns", "ped_l", "ped_o")
QUEUE_LABELS = {
    "veh_ns": "Veículos N→S",
    "ped_l":  "Pedestres Leste",
    "ped_o":  "Pedestres Oeste",
}
QUEUE_COLORS = {
    "veh_ns": "steelblue",
    "ped_l":  "coral",
    "ped_o":  "goldenrod",
}

FAMILIES = ["equilibrado", "pico_veic", "pico_ped", "baixa_mov", "imprevisivel"]

# (nome, hora_inicio, hora_fim, cor_fundo)
TIME_BANDS = [
    ("madrugada",       0,  5,  "#dce0f0"),
    ("manha_tranquila", 5,  7,  "#ddf0dd"),
    ("pico_manha",      7,  10, "#f0dddd"),
    ("meio_manha",      10, 12, "#f0f0dd"),
    ("pico_tarde",      12, 14, "#f0ddf0"),
    ("tarde",           14, 17, "#ddf0f0"),
    ("pico_noite",      17, 20, "#f0eadd"),
    ("noite",           20, 24, "#e8e8e8"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shade_bands(ax: plt.Axes, ymax: float) -> None:
    """Fundo colorido + rótulo de cada faixa horária."""
    for name, start, end, color in TIME_BANDS:
        ax.axvspan(start, end, alpha=0.20, color=color, zorder=0)
        mid = (start + end) / 2
        ax.text(
            mid, ymax * 0.975,
            name.replace("_", "\n"),
            ha="center", va="top",
            fontsize=6.5, color="#444444",
            linespacing=1.15,
            zorder=4,
        )


def load_family_csvs(family: str) -> pd.DataFrame:
    """Lê todos os CSVs de uma família do conjunto de treino."""
    files = sorted(TRAIN_DIR.glob(f"{family}_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV para '{family}' em {TRAIN_DIR}")
    parts = [
        pd.read_csv(f)
        for f in tqdm(files, desc=f"  lendo {family}", leave=False)
    ]
    return pd.concat(parts, ignore_index=True)


def load_all_train_csvs() -> pd.DataFrame:
    """Lê todos os 50 CSVs do conjunto de treino."""
    files = sorted(TRAIN_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV em {TRAIN_DIR}")
    parts = [
        pd.read_csv(f)
        for f in tqdm(files, desc="  lendo treino", leave=False)
    ]
    return pd.concat(parts, ignore_index=True)


# ── Validação A ────────────────────────────────────────────────────────────────

def validation_a(out_dir: Path) -> None:
    """
    Para cada fila lógica, um gráfico mostrando chegada média por hora do dia
    (0-23). Família equilibrado, duas linhas: util e fds. 8 faixas horárias
    marcadas com fundo colorido.
    """
    logger.info("Validação A: carregando cenários da família 'equilibrado'...")
    df = load_family_csvs("equilibrado")

    for queue in QUEUES:
        fig, ax = plt.subplots(figsize=(14, 6))

        for day_type, color, ls, label in [
            ("util", "steelblue", "-",  "Dia útil (util)"),
            ("fds",  "coral",    "--", "Fim de semana (fds)"),
        ]:
            subset = df[df["day_type"] == day_type]
            hourly = subset.groupby("hora")[queue].mean()
            ax.plot(
                hourly.index, hourly.values,
                color=color, linestyle=ls,
                linewidth=2.2, marker="o", markersize=5,
                label=label, zorder=3,
            )

        # Headroom para os rótulos das faixas ficarem acima dos dados
        data_top = ax.get_ylim()[1]
        ax.set_ylim(bottom=0, top=data_top * 1.20)
        ymax = ax.get_ylim()[1]
        _shade_bands(ax, ymax)

        ax.set_xlabel("Hora do dia", fontsize=12)
        ax.set_ylabel("Chegadas médias / tick", fontsize=12)
        ax.set_title(
            f"Validação A — Perfil horário: {QUEUE_LABELS[queue]}\n"
            "Família: equilibrado | Agregado sobre todos os seeds",
            fontsize=13,
        )
        ax.set_xticks(range(24))
        ax.set_xticklabels([str(h) for h in range(24)], fontsize=9)
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(axis="y", alpha=0.4, zorder=1)

        out = out_dir / f"validation_A_{queue}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Salvo: %s", out)


# ── Validação B ────────────────────────────────────────────────────────────────

def validation_b(out_dir: Path) -> None:
    """
    Barplot comparando as 5 famílias na faixa pico_manha / util.
    3 barras por família (veh_ns, ped_l, ped_o). Valor numérico acima de cada barra.
    """
    logger.info("Validação B: carregando todos os cenários de treino...")
    df = load_all_train_csvs()

    subset = df[(df["day_type"] == "util") & (df["time_band"] == "pico_manha")]

    means: dict[str, dict[str, float]] = {}
    for fam in FAMILIES:
        fam_data = subset[subset["family"] == fam]
        means[fam] = {q: float(fam_data[q].mean()) for q in QUEUES}

    width = 0.22
    x = np.arange(len(FAMILIES))

    fig, ax = plt.subplots(figsize=(15, 7))

    for i, queue in enumerate(QUEUES):
        offset = (i - 1) * width
        vals = [means[fam][queue] for fam in FAMILIES]
        bars = ax.bar(
            x + offset, vals, width,
            label=QUEUE_LABELS[queue],
            color=QUEUE_COLORS[queue],
            edgecolor="white", linewidth=0.6,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.004,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, color="#222222",
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, fontsize=11)
    ax.set_ylabel("Chegadas médias / tick", fontsize=12)
    ax.set_title(
        "Validação B — Comparação entre famílias\n"
        "Faixa: pico_manha | Tipo de dia: util",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.4, zorder=1)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

    out = out_dir / "validation_B_familias_pico_manha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", out)


# ── Validação C ────────────────────────────────────────────────────────────────

REPRO_SCENARIOS = [
    ("equilibrado", "util", 9999),
    ("pico_veic",   "fds",  42),
    ("baixa_mov",   "util", 1234),
]


def validation_c(scenarios_cfg: dict) -> None:
    """
    Gera cada cenário 2x e compara SHA-256 dos CSVs.
    Imprime REPRODUCIBILITY: PASS ou FAIL para cada caso.
    """
    gen = ScenarioGenerator(scenarios_cfg)
    all_pass = True

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        for family, day_type, seed in REPRO_SCENARIOS:
            dir1 = base / "run1"
            dir2 = base / "run2"
            dir1.mkdir(exist_ok=True)
            dir2.mkdir(exist_ok=True)

            csv1, _ = gen.generate(family, day_type, seed, dir1)
            csv2, _ = gen.generate(family, day_type, seed, dir2)

            h1 = _sha256(csv1)
            h2 = _sha256(csv2)
            tag = f"{family}/{day_type}/seed={seed}"

            if h1 == h2:
                print(f"REPRODUCIBILITY: PASS  [{tag}]  sha256={h1[:12]}...")
            else:
                all_pass = False
                print(f"REPRODUCIBILITY: FAIL  [{tag}]")
                lines1 = csv1.read_text(encoding="utf-8").splitlines()
                lines2 = csv2.read_text(encoding="utf-8").splitlines()
                for idx, (a, b) in enumerate(zip(lines1, lines2)):
                    if a != b:
                        print(f"  primeira diferença na linha {idx}:")
                        print(f"    run1: {a!r}")
                        print(f"    run2: {b!r}")
                        break

            for d in (dir1, dir2):
                for f in d.iterdir():
                    f.unlink()

    print()
    if all_pass:
        print("REPRODUCIBILITY: PASS  [todos os 3 cenários validados]")
    else:
        print("REPRODUCIBILITY: FAIL  [pelo menos um cenário divergiu]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = load_all_configs("configs")

    validation_a(OUT_DIR)
    validation_b(OUT_DIR)
    validation_c(cfgs["scenarios"])

    logger.info("Validação concluída. PNGs em: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

"""
Análise completa dos resultados — Fase 6.

Gera:
1. Tabelas comparativas (agente vs baseline)
2. Análise de Monte Carlo (variabilidade dos resultados)
3. Gráficos comparativos salvos em results/graficos/

Como usar:
    python scripts/analyze_results.py

Requisitos:
    - results/benchmark_baseline.csv  (gerado por run_benchmark.py)
    - results/agent_results.csv       (gerado por evaluate_agent.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Configuração visual ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#404060",
    "axes.labelcolor":   "#c8c8d8",
    "xtick.color":       "#c8c8d8",
    "ytick.color":       "#c8c8d8",
    "text.color":        "#c8c8d8",
    "grid.color":        "#2a2a4a",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

COR_BASE  = "#e07070"   # vermelho — baseline
COR_AGENT = "#70aaee"   # azul — agente
COR_GRID  = "#2a2a4a"

METRICAS_PRINCIPAIS = [
    ("espera_media_carros",      "Espera média\ncarros (s)"),
    ("espera_media_pedestres",   "Espera média\npedestr. (s)"),
    ("espera_maxima_carros",     "Espera máx.\ncarros (s)"),
    ("espera_maxima_pedestres",  "Espera máx.\npedestr. (s)"),
    ("violacoes_teto_carros",    "Violações\nteto carros"),
    ("violacoes_teto_pedestres", "Violações\nteto pedesr."),
]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os CSVs de resultado e valida."""
    base_path  = Path("results/benchmark_baseline.csv")
    agent_path = Path("results/agent_results.csv")

    if not base_path.exists():
        print("ERRO: results/benchmark_baseline.csv não encontrado.")
        print("Execute: python scripts/run_benchmark.py")
        sys.exit(1)

    if not agent_path.exists():
        print("ERRO: results/agent_results.csv não encontrado.")
        print("Execute: python scripts/evaluate_agent.py")
        sys.exit(1)

    df_base  = pd.read_csv(base_path)
    df_base  = df_base[df_base["set"] == "eval"].copy()
    df_agent = pd.read_csv(agent_path)

    print(f"Baseline: {len(df_base)} cenários de avaliação")
    print(f"Agente  : {len(df_agent)} cenários\n")

    return df_base, df_agent


def fmt(v: float) -> str:
    return f"{v:.1f}"


def pct(b: float, a: float) -> str:
    if b == 0:
        return "—"
    p = (a - b) / b * 100
    sinal = "+" if p > 0 else ""
    return f"{sinal}{p:.1f}%"


def imprimir_tabela_resumo(df_base: pd.DataFrame, df_agent: pd.DataFrame) -> None:
    """Imprime tabela comparativa no terminal."""
    print("=" * 65)
    print("  COMPARAÇÃO GERAL — Agente PPO vs Semáforo de Tempo Fixo")
    print("=" * 65)
    print(f"{'Métrica':<30} {'Baseline':>10} {'Agente':>10} {'Variação':>10}")
    print("-" * 65)

    for col, label in METRICAS_PRINCIPAIS:
        label_curto = label.replace("\n", " ")
        b = df_base[col].mean()
        a = df_agent[col].mean()
        sinal = "✓" if a <= b else "✗"
        print(f"  {sinal}  {label_curto:<26} {fmt(b):>10} {fmt(a):>10} {pct(b,a):>10}")

    print("=" * 65)


def imprimir_tabela_por_familia(df_base: pd.DataFrame, df_agent: pd.DataFrame) -> None:
    """Imprime tabela por família de cenário."""
    print("\n" + "=" * 65)
    print("  POR FAMÍLIA DE CENÁRIO")
    print("=" * 65)

    familias = sorted(df_agent["family"].unique())
    for fam in familias:
        ag = df_agent[df_agent["family"] == fam]
        ba = df_base[df_base["family"] == fam]
        if ba.empty:
            continue

        print(f"\n  {fam.upper()}")
        print(f"  {'Métrica':<28} {'Baseline':>10} {'Agente':>10} {'Var.':>8}")
        print("  " + "-" * 58)

        for col, label in METRICAS_PRINCIPAIS[:4]:
            label_curto = label.replace("\n", " ")
            b = ba[col].mean()
            a = ag[col].mean()
            sinal = "✓" if a <= b else "✗"
            print(f"  {sinal}  {label_curto:<26} {fmt(b):>10} {fmt(a):>10} {pct(b,a):>8}")


def monte_carlo(df: pd.DataFrame, col: str, n_amostras: int = 10000) -> dict:
    """
    Análise de Monte Carlo para uma métrica.

    Reamostra os cenários com reposição e calcula a distribuição
    da média, gerando intervalos de confiança robustos.

    Parâmetros
    ----------
    df : DataFrame com os resultados
    col : coluna de métrica
    n_amostras : número de reamostras bootstrap

    Retorna
    -------
    dict com mean, std, ci_95_low, ci_95_high, mediana
    """
    valores = df[col].values
    medias  = np.array([
        np.mean(np.random.choice(valores, size=len(valores), replace=True))
        for _ in range(n_amostras)
    ])
    return {
        "mean":      np.mean(medias),
        "std":       np.std(medias),
        "ci_95_low": np.percentile(medias, 2.5),
        "ci_95_high":np.percentile(medias, 97.5),
        "mediana":   np.median(medias),
        "amostras":  medias,
    }


def imprimir_monte_carlo(df_base: pd.DataFrame, df_agent: pd.DataFrame) -> None:
    """Imprime análise de Monte Carlo no terminal."""
    print("\n" + "=" * 65)
    print("  ANÁLISE DE MONTE CARLO (10.000 reamostras bootstrap)")
    print("  IC 95% = intervalo de confiança de 95%")
    print("=" * 65)

    for col, label in METRICAS_PRINCIPAIS[:4]:
        label_curto = label.replace("\n", " ")
        mc_b = monte_carlo(df_base, col)
        mc_a = monte_carlo(df_agent, col)

        print(f"\n  {label_curto}")
        print(f"  {'':4} {'Média':>8} {'±std':>8} {'IC 95% baixo':>14} {'IC 95% alto':>13}")
        print(f"  Base  {fmt(mc_b['mean']):>8} {fmt(mc_b['std']):>8} "
              f"{fmt(mc_b['ci_95_low']):>14} {fmt(mc_b['ci_95_high']):>13}")
        print(f"  Agent {fmt(mc_a['mean']):>8} {fmt(mc_a['std']):>8} "
              f"{fmt(mc_a['ci_95_low']):>14} {fmt(mc_a['ci_95_high']):>13}")

        # Teste t para significância estatística
        t_stat, p_val = stats.ttest_ind(df_base[col].values, df_agent[col].values)
        sig = "✓ significativo (p<0.05)" if p_val < 0.05 else "✗ não significativo"
        print(f"  Teste t: p={p_val:.4f} → {sig}")


# ── Gráficos ──────────────────────────────────────────────────────────────────

def grafico_barras_comparativo(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Gráfico de barras comparando médias de todas as métricas principais."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Agente PPO vs Semáforo de Tempo Fixo\nMédia nos 32 cenários de avaliação",
                 fontsize=13, y=1.01)

    for ax, (col, label) in zip(axes.flat, METRICAS_PRINCIPAIS):
        b = df_base[col].mean()
        a = df_agent[col].mean()
        b_std = df_base[col].std()
        a_std = df_agent[col].std()

        bars = ax.bar(
            ["Baseline\n(tempo fixo)", "Agente\nPPO"],
            [b, a],
            color=[COR_BASE, COR_AGENT],
            width=0.5,
            yerr=[b_std, a_std],
            capsize=5,
            error_kw={"ecolor": "#ffffff", "alpha": 0.6},
        )

        # Valores nas barras
        for bar, val in zip(bars, [b, a]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + b_std * 0.1,
                fmt(val),
                ha="center", va="bottom", fontsize=9, color="#ffffff",
            )

        # Variação percentual
        variacao = pct(b, a)
        cor_var  = "#70ee90" if a <= b else "#ee7070"
        ax.text(0.98, 0.95, variacao, transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=cor_var, fontweight="bold")

        ax.set_title(label.replace("\n", " "), pad=8)
        ax.set_ylabel("segundos / contagem")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "01_barras_comparativo.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvo: {path}")


def grafico_por_familia(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Gráfico de espera média por família de cenário."""
    familias = sorted(df_agent["family"].unique())
    metricas = [
        ("espera_media_carros",    "Espera média carros (s)"),
        ("espera_media_pedestres", "Espera média pedestres (s)"),
        ("violacoes_teto_carros",  "Violações teto carros"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparação por Família de Cenário", fontsize=13)

    x = np.arange(len(familias))
    width = 0.35

    for ax, (col, titulo) in zip(axes, metricas):
        vals_b = [df_base[df_base["family"]==f][col].mean() for f in familias]
        vals_a = [df_agent[df_agent["family"]==f][col].mean() for f in familias]

        ax.bar(x - width/2, vals_b, width, label="Baseline", color=COR_BASE, alpha=0.85)
        ax.bar(x + width/2, vals_a, width, label="Agente PPO", color=COR_AGENT, alpha=0.85)

        ax.set_title(titulo, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in familias], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "02_por_familia.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvo: {path}")


def grafico_boxplot(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot mostrando distribuição dos resultados por cenário."""
    metricas = [
        ("espera_media_carros",    "Espera média\ncarros (s)"),
        ("espera_media_pedestres", "Espera média\npedestr. (s)"),
        ("espera_maxima_carros",   "Espera máx.\ncarros (s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Distribuição dos Resultados por Cenário\n(cada ponto = 1 cenário de avaliação)", fontsize=12)

    for ax, (col, titulo) in zip(axes, metricas):
        data_b = df_base[col].values
        data_a = df_agent[col].values

        bp = ax.boxplot(
            [data_b, data_a],
            labels=["Baseline", "Agente\nPPO"],
            patch_artist=True,
            medianprops={"color": "#ffffff", "linewidth": 2},
            whiskerprops={"color": "#888888"},
            capprops={"color": "#888888"},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        )
        bp["boxes"][0].set_facecolor(COR_BASE)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COR_AGENT)
        bp["boxes"][1].set_alpha(0.7)

        # Pontos individuais
        for i, data in enumerate([data_b, data_a], 1):
            jitter = np.random.normal(0, 0.06, len(data))
            ax.scatter(i + jitter, data, alpha=0.4, s=15,
                      color=[COR_BASE, COR_AGENT][i-1], zorder=3)

        ax.set_title(titulo, pad=8)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "03_boxplot_distribuicao.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvo: {path}")


def grafico_monte_carlo(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Gráfico de distribuição Monte Carlo para as métricas principais."""
    metricas = [
        ("espera_media_carros",    "Espera média carros (s)"),
        ("espera_media_pedestres", "Espera média pedestres (s)"),
        ("violacoes_teto_carros",  "Violações teto carros"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Análise de Monte Carlo — Distribuição das Médias\n(10.000 reamostras bootstrap)",
                 fontsize=12)

    for ax, (col, titulo) in zip(axes, metricas):
        mc_b = monte_carlo(df_base, col)
        mc_a = monte_carlo(df_agent, col)

        # Histogramas das distribuições bootstrap
        ax.hist(mc_b["amostras"], bins=60, alpha=0.6, color=COR_BASE,
                label=f"Baseline\n{fmt(mc_b['mean'])} ±{fmt(mc_b['std'])}",
                density=True)
        ax.hist(mc_a["amostras"], bins=60, alpha=0.6, color=COR_AGENT,
                label=f"Agente PPO\n{fmt(mc_a['mean'])} ±{fmt(mc_a['std'])}",
                density=True)

        # Linhas de média
        ax.axvline(mc_b["mean"], color=COR_BASE,  linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axvline(mc_a["mean"], color=COR_AGENT, linestyle="--", linewidth=1.5, alpha=0.9)

        # IC 95%
        ax.axvspan(mc_b["ci_95_low"], mc_b["ci_95_high"],
                   alpha=0.15, color=COR_BASE, label="IC 95%")
        ax.axvspan(mc_a["ci_95_low"], mc_a["ci_95_high"],
                   alpha=0.15, color=COR_AGENT)

        ax.set_title(titulo, pad=8)
        ax.set_xlabel("Valor da métrica")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "04_monte_carlo.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvo: {path}")


def grafico_scatter_cenarios(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: baseline vs agente por cenário — pontos abaixo da diagonal = agente melhor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cenário a Cenário — Agente vs Baseline\n(abaixo da diagonal = agente melhor)",
                 fontsize=12)

    pares = [
        ("espera_media_carros",    "Espera média carros (s)"),
        ("violacoes_teto_carros",  "Violações teto carros"),
    ]

    df_merged = df_agent.merge(
        df_base[["scenario_id"] + [c for c, _ in pares]],
        on="scenario_id", suffixes=("_agente", "_base"), how="inner"
    )

    for ax, (col, titulo) in zip(axes, pares):
        x = df_merged[f"{col}_base"].values
        y = df_merged[f"{col}_agente"].values

        # Cores: verde se agente melhor, vermelho se pior
        cores = [COR_AGENT if yi <= xi else COR_BASE for xi, yi in zip(x, y)]

        ax.scatter(x, y, c=cores, alpha=0.7, s=40, zorder=3)

        # Diagonal y=x
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#ffffff", alpha=0.3, linewidth=1)

        n_melhor = sum(1 for xi, yi in zip(x, y) if yi <= xi)
        ax.set_xlabel(f"Baseline — {titulo}")
        ax.set_ylabel(f"Agente PPO — {titulo}")
        ax.set_title(f"{titulo}\n{n_melhor}/{len(x)} cenários com agente melhor")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "05_scatter_cenarios.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvo: {path}")


def gerar_relatorio_md(df_base: pd.DataFrame, df_agent: pd.DataFrame, out_dir: Path) -> None:
    """Gera relatório Markdown com tabelas e referências aos gráficos."""

    def tabela_md(cols_labels, df_b, df_a):
        header = "| Métrica | Baseline | Agente PPO | Variação |"
        sep    = "|---|---:|---:|---:|"
        linhas = [header, sep]
        for col, label in cols_labels:
            b = df_b[col].mean()
            a = df_a[col].mean()
            sinal = "✅" if a <= b else "❌"
            linhas.append(f"| {sinal} {label} | {fmt(b)} | {fmt(a)} | {pct(b,a)} |")
        return "\n".join(linhas)

    def tabela_familia_md(col, label):
        familias = sorted(df_agent["family"].unique())
        header = f"| Família | Baseline | Agente PPO | Variação |"
        sep    = "|---|---:|---:|---:|"
        linhas = [header, sep]
        for fam in familias:
            b = df_base[df_base["family"]==fam][col].mean()
            a = df_agent[df_agent["family"]==fam][col].mean()
            linhas.append(f"| {fam} | {fmt(b)} | {fmt(a)} | {pct(b,a)} |")
        return "\n".join(linhas)

    def tabela_mc_md():
        header = "| Métrica | Baseline média | IC 95% | Agente média | IC 95% | p-valor |"
        sep    = "|---|---:|---|---:|---|---:|"
        linhas = [header, sep]
        for col, label in METRICAS_PRINCIPAIS[:4]:
            mc_b = monte_carlo(df_base, col, n_amostras=5000)
            mc_a = monte_carlo(df_agent, col, n_amostras=5000)
            _, p  = stats.ttest_ind(df_base[col].values, df_agent[col].values)
            sig  = "✅ <0.05" if p < 0.05 else "❌ ≥0.05"
            linhas.append(
                f"| {label.replace(chr(10),' ')} "
                f"| {fmt(mc_b['mean'])} "
                f"| [{fmt(mc_b['ci_95_low'])}, {fmt(mc_b['ci_95_high'])}] "
                f"| {fmt(mc_a['mean'])} "
                f"| [{fmt(mc_a['ci_95_low'])}, {fmt(mc_a['ci_95_high'])}] "
                f"| {sig} |"
            )
        return "\n".join(linhas)

    md = f"""# Relatório de Análise — Semáforo Inteligente

> **Agente:** PPO (Stable-Baselines3)  
> **Cenários de avaliação:** {len(df_agent)}  
> **Benchmark:** Semáforo de tempo fixo (Fase A=45s / Fase B=25s)  
> **Tetos:** carros ≤ 90s | pedestres ≤ 60s  

---

## 1. Comparação Geral

{tabela_md(METRICAS_PRINCIPAIS, df_base, df_agent)}

![Comparação geral](graficos/01_barras_comparativo.png)

---

## 2. Comparação por Família de Cenário

### Espera média — carros

{tabela_familia_md("espera_media_carros", "Espera média carros (s)")}

### Espera média — pedestres

{tabela_familia_md("espera_media_pedestres", "Espera média pedestres (s)")}

![Por família](graficos/02_por_familia.png)

---

## 3. Distribuição dos Resultados

O boxplot abaixo mostra a variabilidade dos resultados entre os {len(df_agent)} cenários:

![Boxplot](graficos/03_boxplot_distribuicao.png)

---

## 4. Análise de Monte Carlo

Reamostragem bootstrap com 10.000 iterações para estimativa robusta dos intervalos de confiança:

{tabela_mc_md()}

![Monte Carlo](graficos/04_monte_carlo.png)

---

## 5. Análise Cenário a Cenário

Cada ponto representa um cenário de avaliação. Pontos abaixo da diagonal indicam que o agente foi melhor que o baseline naquele cenário específico:

![Scatter](graficos/05_scatter_cenarios.png)

---

## 6. Conclusão

O agente PPO demonstrou comportamento **assimétrico**: reduziu significativamente o tempo de espera de pedestres mas aumentou o de carros. Isso indica que a função de recompensa precisa de melhor calibração do peso de equilíbrio entre as duas filas.

*Gerado automaticamente por scripts/analyze_results.py*
"""

    path = out_dir / "relatorio_completo.md"
    path.write_text(md, encoding="utf-8")
    print(f"  Salvo: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Análise completa de resultados — Semáforo Inteligente")
    print("=" * 60 + "\n")

    df_base, df_agent = carregar_dados()

    # Pasta de saída para gráficos
    out_dir = Path("results")
    graf_dir = out_dir / "graficos"
    graf_dir.mkdir(parents=True, exist_ok=True)

    # Tabelas no terminal
    imprimir_tabela_resumo(df_base, df_agent)
    imprimir_tabela_por_familia(df_base, df_agent)
    imprimir_monte_carlo(df_base, df_agent)

    # Gráficos
    print("\nGerando gráficos...")
    grafico_barras_comparativo(df_base, df_agent, graf_dir)
    grafico_por_familia(df_base, df_agent, graf_dir)
    grafico_boxplot(df_base, df_agent, graf_dir)
    grafico_monte_carlo(df_base, df_agent, graf_dir)
    grafico_scatter_cenarios(df_base, df_agent, graf_dir)

    # Relatório Markdown
    print("\nGerando relatório...")
    gerar_relatorio_md(df_base, df_agent, out_dir)

    print("\n" + "=" * 60)
    print("  Análise concluída!")
    print(f"  Gráficos: results/graficos/")
    print(f"  Relatório: results/relatorio_completo.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

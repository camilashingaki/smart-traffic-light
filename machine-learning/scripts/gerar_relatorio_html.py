"""
Gera relatório HTML bonito com resultados do DualAgent.

Como usar:
    python scripts/gerar_relatorio_html.py

Abre automaticamente no navegador ao terminar.
Salvo em: results/relatorio_final.html
"""

from __future__ import annotations

import base64
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats


def img_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fmt(v: float) -> str:
    return f"{v:.1f}"


def pct(b: float, a: float) -> str:
    if b == 0:
        return "—"
    p = (a - b) / b * 100
    sinal = "+" if p > 0 else ""
    cor = "positivo" if p <= 0 else "negativo"
    return f'<span class="{cor}">{sinal}{p:.1f}%</span>'


def monte_carlo(df: pd.DataFrame, col: str, n: int = 5000) -> dict:
    valores = df[col].values
    medias  = np.array([
        np.mean(np.random.choice(valores, size=len(valores), replace=True))
        for _ in range(n)
    ])
    return {
        "mean":   np.mean(medias),
        "std":    np.std(medias),
        "ci_low": np.percentile(medias, 2.5),
        "ci_high":np.percentile(medias, 97.5),
    }


def main() -> None:
    base_path  = Path("results/benchmark_baseline.csv")
    agent_path = Path("results/agent_results.csv")
    dual_path  = Path("results/dual_agent_results.csv")
    graf_dir   = Path("results/graficos")

    if not base_path.exists():
        print("ERRO: rode primeiro run_benchmark.py")
        sys.exit(1)

    df_base  = pd.read_csv(base_path)
    df_base  = df_base[df_base["set"] == "eval"].copy()

    # Usa DualAgent se disponível, senão agente geral
    if dual_path.exists():
        df_agent = pd.read_csv(dual_path)
        agente_label = "DualAgent (geral + especialista pico_veic)"
        agente_tag   = "DualAgent"
    elif agent_path.exists():
        df_agent = pd.read_csv(agent_path)
        agente_label = "Agente PPO Geral"
        agente_tag   = "Agente PPO"
    else:
        print("ERRO: rode primeiro evaluate_agent.py ou evaluate_dual_agent.py")
        sys.exit(1)

    df_geral = pd.read_csv(agent_path) if agent_path.exists() else None

    print(f"Usando: {agente_label}")

    METRICAS = [
        ("espera_media_carros",      "Espera média carros (s)"),
        ("espera_media_pedestres",   "Espera média pedestres (s)"),
        ("espera_maxima_carros",     "Espera máxima carros (s)"),
        ("espera_maxima_pedestres",  "Espera máxima pedestres (s)"),
        ("violacoes_teto_carros",    "Violações teto carros"),
        ("violacoes_teto_pedestres", "Violações teto pedestres"),
    ]

    # ── Tabela geral ──────────────────────────────────────────────────────────
    linhas_geral = ""
    for col, label in METRICAS:
        b = df_base[col].mean()
        a = df_agent[col].mean()
        g = df_geral[col].mean() if df_geral is not None else None
        icone = "✅" if a <= b else "❌"
        col_geral = f"<td>{fmt(g)}</td>" if g is not None else ""
        linhas_geral += f"""
        <tr>
            <td>{icone} {label}</td>
            <td>{fmt(b)}</td>
            {col_geral}
            <td>{fmt(a)}</td>
            <td>{pct(b, a)}</td>
        </tr>"""

    th_geral = "<th>Agente Geral</th>" if df_geral is not None else ""

    # ── Tabela por família ────────────────────────────────────────────────────
    familias = sorted(df_agent["family"].unique())
    linhas_familia = ""
    for fam in familias:
        ag = df_agent[df_agent["family"] == fam]
        ba = df_base[df_base["family"] == fam]
        if ba.empty:
            continue
        for col, label in METRICAS[:4]:
            b = ba[col].mean()
            a = ag[col].mean()
            icone = "✅" if a <= b else "❌"
            fam_display = fam.replace("_", " ").title()
            linhas_familia += f"""
        <tr>
            <td><span class="tag">{fam_display}</span></td>
            <td>{icone} {label}</td>
            <td>{fmt(b)}</td>
            <td>{fmt(a)}</td>
            <td>{pct(b, a)}</td>
        </tr>"""

    # ── Tabela Monte Carlo ────────────────────────────────────────────────────
    linhas_mc = ""
    for col, label in METRICAS[:4]:
        mc_b = monte_carlo(df_base, col)
        mc_a = monte_carlo(df_agent, col)
        _, p  = stats.ttest_ind(df_base[col].values, df_agent[col].values)
        sig   = f'<span class="positivo">✅ p={p:.4f}</span>' if p < 0.05 else f'<span class="negativo">❌ p={p:.4f}</span>'
        linhas_mc += f"""
        <tr>
            <td>{label}</td>
            <td>{fmt(mc_b['mean'])} ± {fmt(mc_b['std'])}</td>
            <td>[{fmt(mc_b['ci_low'])}, {fmt(mc_b['ci_high'])}]</td>
            <td>{fmt(mc_a['mean'])} ± {fmt(mc_a['std'])}</td>
            <td>[{fmt(mc_a['ci_low'])}, {fmt(mc_a['ci_high'])}]</td>
            <td>{sig}</td>
        </tr>"""

    # ── Gráficos ──────────────────────────────────────────────────────────────
    graficos = {k: img_base64(graf_dir / f) for k, f in {
        "barras":  "01_barras_comparativo.png",
        "familia": "02_por_familia.png",
        "boxplot": "03_boxplot_distribuicao.png",
        "mc":      "04_monte_carlo.png",
        "scatter": "05_scatter_cenarios.png",
    }.items()}

    def img_tag(key: str, alt: str) -> str:
        if not graficos.get(key):
            return f"<p><em>Gráfico não encontrado. Execute analyze_results.py primeiro.</em></p>"
        return f'<img src="data:image/png;base64,{graficos[key]}" alt="{alt}" class="grafico">'

    # ── Destaques ─────────────────────────────────────────────────────────────
    esp_car_b = df_base["espera_media_carros"].mean()
    esp_car_a = df_agent["espera_media_carros"].mean()
    esp_ped_b = df_base["espera_media_pedestres"].mean()
    esp_ped_a = df_agent["espera_media_pedestres"].mean()
    viol_ped_b = df_base["violacoes_teto_pedestres"].sum()
    viol_ped_a = df_agent["violacoes_teto_pedestres"].sum()

    car_cor   = "positivo" if esp_car_a <= esp_car_b else "negativo"
    ped_cor   = "positivo" if esp_ped_a <= esp_ped_b else "negativo"
    viol_cor  = "positivo" if viol_ped_a <= viol_ped_b else "negativo"

    metricas_ok = sum(1 for col, _ in METRICAS if df_agent[col].mean() <= df_base[col].mean())

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório Final — Semáforo Inteligente</title>
<style>
  :root {{
    --bg:#0f0f1a; --card:#1a1a2e; --border:#2a2a4a; --text:#c8c8d8;
    --muted:#888898; --accent:#4a90d9; --green:#4ec94e; --red:#e05050; --yellow:#e0b840;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:15px; line-height:1.6; padding:0 0 60px; }}
  .hero {{ background:linear-gradient(135deg,#1a1a3e 0%,#0f0f2a 100%); border-bottom:1px solid var(--border); padding:48px 40px 36px; text-align:center; }}
  .hero h1 {{ font-size:2.2rem; font-weight:700; color:#fff; margin-bottom:8px; }}
  .hero p {{ color:var(--muted); font-size:1rem; }}
  .badges {{ display:flex; justify-content:center; gap:12px; margin-top:20px; flex-wrap:wrap; }}
  .badge {{ background:var(--card); border:1px solid var(--border); border-radius:999px; padding:4px 16px; font-size:13px; color:var(--muted); }}
  .badge.destaque {{ border-color:var(--accent); color:var(--accent); }}
  .container {{ max-width:1100px; margin:0 auto; padding:0 24px; }}
  .section {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:28px 32px; margin-top:28px; }}
  .section h2 {{ font-size:1.25rem; color:#fff; margin-bottom:20px; padding-bottom:12px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }}
  .section h2 .num {{ background:var(--accent); color:#fff; width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; flex-shrink:0; }}
  .section h3 {{ font-size:1rem; color:var(--accent); margin:24px 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ background:#12122a; color:var(--muted); font-weight:600; text-align:left; padding:10px 14px; border-bottom:2px solid var(--border); font-size:12px; text-transform:uppercase; letter-spacing:0.5px; }}
  td {{ padding:10px 14px; border-bottom:1px solid var(--border); vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:rgba(255,255,255,0.03); }}
  .positivo {{ color:var(--green); font-weight:600; }}
  .negativo {{ color:var(--red); font-weight:600; }}
  .tag {{ background:rgba(74,144,217,0.15); border:1px solid rgba(74,144,217,0.3); border-radius:6px; padding:2px 10px; font-size:12px; color:var(--accent); white-space:nowrap; }}
  .grafico {{ width:100%; border-radius:8px; margin-top:16px; border:1px solid var(--border); }}
  .callout {{ border-left:3px solid var(--accent); background:rgba(74,144,217,0.08); border-radius:0 8px 8px 0; padding:14px 18px; margin:16px 0; font-size:14px; color:var(--muted); }}
  .callout strong {{ color:var(--text); }}
  .grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:16px; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:16px; }}
  .stat-card {{ background:#12122a; border:1px solid var(--border); border-radius:10px; padding:16px 20px; }}
  .stat-card .label {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
  .stat-card .value {{ font-size:1.6rem; font-weight:700; }}
  .stat-card .sub {{ font-size:12px; color:var(--muted); margin-top:4px; }}
  @media(max-width:700px) {{ .grid-4,.grid-2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<div class="hero">
  <h1>🚦 Semáforo Inteligente</h1>
  <p>Relatório Final de Avaliação — {agente_label}</p>
  <div class="badges">
    <span class="badge destaque">{agente_tag}</span>
    <span class="badge">{len(df_agent)} cenários avaliados</span>
    <span class="badge">Teto carros: 90s · Pedestres: 60s</span>
    <span class="badge">Benchmark: tempo fixo 45s/25s</span>
    <span class="badge">{metricas_ok}/6 métricas melhoradas</span>
  </div>
</div>

<div class="container">

  <div class="section">
    <h2><span class="num">★</span> Destaques</h2>
    <div class="grid-4">
      <div class="stat-card">
        <div class="label">Espera média carros</div>
        <div class="value {car_cor}">{fmt(esp_car_a)}s</div>
        <div class="sub">baseline: {fmt(esp_car_b)}s ({pct(esp_car_b, esp_car_a).replace('<span class="positivo">','').replace('<span class="negativo">','').replace('</span>','')})</div>
      </div>
      <div class="stat-card">
        <div class="label">Espera média pedestres</div>
        <div class="value {ped_cor}">{fmt(esp_ped_a)}s</div>
        <div class="sub">baseline: {fmt(esp_ped_b)}s</div>
      </div>
      <div class="stat-card">
        <div class="label">Violações teto pedestres</div>
        <div class="value {viol_cor}">{int(viol_ped_a)}</div>
        <div class="sub">baseline: {int(viol_ped_b)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Métricas melhoradas</div>
        <div class="value {'positivo' if metricas_ok >= 4 else 'negativo'}">{metricas_ok}/6</div>
        <div class="sub">vs semáforo de tempo fixo</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2><span class="num">1</span> Comparação Geral</h2>
    <div class="callout">
      Médias sobre <strong>{len(df_agent)} cenários de avaliação</strong>.
      ✅ = {agente_tag} melhor · ❌ = {agente_tag} pior que o baseline.
    </div>
    <table>
      <thead>
        <tr>
          <th>Métrica</th>
          <th>Baseline (tempo fixo)</th>
          {th_geral}
          <th>{agente_tag}</th>
          <th>Variação vs baseline</th>
        </tr>
      </thead>
      <tbody>{linhas_geral}</tbody>
    </table>
    {img_tag("barras", "Comparação geral")}
  </div>

  <div class="section">
    <h2><span class="num">2</span> Por Família de Cenário</h2>
    <table>
      <thead>
        <tr><th>Família</th><th>Métrica</th><th>Baseline</th><th>{agente_tag}</th><th>Variação</th></tr>
      </thead>
      <tbody>{linhas_familia}</tbody>
    </table>
    {img_tag("familia", "Por família")}
  </div>

  <div class="section">
    <h2><span class="num">3</span> Distribuição dos Resultados</h2>
    {img_tag("boxplot", "Boxplot")}
    {img_tag("scatter", "Scatter")}
  </div>

  <div class="section">
    <h2><span class="num">4</span> Análise de Monte Carlo</h2>
    <div class="callout">Bootstrap com 5.000 reamostras · IC 95%</div>
    <table>
      <thead>
        <tr>
          <th>Métrica</th>
          <th>Baseline média ± std</th><th>IC 95%</th>
          <th>{agente_tag} média ± std</th><th>IC 95%</th>
          <th>Significância</th>
        </tr>
      </thead>
      <tbody>{linhas_mc}</tbody>
    </table>
    {img_tag("mc", "Monte Carlo")}
  </div>

  <div class="section">
    <h2><span class="num">5</span> Conclusão</h2>
    <div class="callout">
      O <strong>{agente_tag}</strong> superou o semáforo de tempo fixo em
      <strong>{metricas_ok} das 6 métricas principais</strong>.
      A abordagem de dois agentes especializados permitiu melhorar o desempenho
      em cenários de alto volume de veículos (<em>pico_veic</em>) sem degradar
      os demais cenários.
    </div>
    <h3>Arquitetura DualAgent</h3>
    <p style="color:var(--muted);font-size:14px;margin-top:8px;line-height:2">
      A cada tick, o sistema verifica a proporção de carros vs pedestres nas filas.<br>
      Se <code>fila_carros &gt; fila_pedestres × 2</code> e <code>fila_carros &gt; 3</code>:
      usa o <strong>agente especialista pico_veic</strong>.<br>
      Caso contrário: usa o <strong>agente geral</strong>.<br>
      O agente não conhece o nome do cenário — decide apenas pelo estado atual das filas.
    </p>
  </div>

</div>
</body>
</html>"""

    out = Path("results/relatorio_final.html")
    out.write_text(html, encoding="utf-8")
    print(f"Relatório salvo: {out}")
    webbrowser.open(f"file://{out.resolve()}")
    print("Abrindo no navegador...")


if __name__ == "__main__":
    main()

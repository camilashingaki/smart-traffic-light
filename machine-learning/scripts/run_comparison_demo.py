"""
Demo visual do DualAgent com comparação progressiva vs benchmark em tempo real.

Mostra o cruzamento animado à esquerda e três gráficos comparativos
à direita, todos crescendo tick a tick:
  - Espera média acumulada (agente vs benchmark)
  - Violações de teto (agente vs benchmark)
  - Tamanho das filas (agente vs benchmark)

Como usar:
    python scripts/run_comparison_demo.py

Controles:
    ESPAÇO  — pausa / retoma
    1/2/3/4 — velocidade 1x / 5x / 10x / 50x
    R       — reinicia o cenário
    F       — troca de cenário aleatório
    Q / ESC — encerra
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from stable_baselines3 import PPO
from src.simulation.crossing import Crossing
from src.simulation.controllers import FixedTimeController
from src.simulation.simulation_loop import SimulationLoop
from src.utils.config_loader import load_config
from src.visualization.fonts import load_ui_font
from src.visualization.pygame_renderer import CrossingRenderer

# ── Layout ────────────────────────────────────────────────────────────────────
RENDERER_W = 720
PLOTS_W    = 560
WIN_H      = 720
FPS        = 60
HISTORY    = 720   # ticks de histórico nos gráficos (~1 hora)
BANNER_H   = 40    # faixa reservada no topo do painel direito (controlador/cenário)
LEGEND_Y   = 26    # distância da legenda inferior à base da janela

# ── Cores dos gráficos ────────────────────────────────────────────────────────
COR_AGENTE    = "#5b8df5"   # azul
COR_BENCHMARK = "#f56565"   # vermelho
COR_AG_FILL   = "#2dd4a8"   # verde-azulado — área quando o agente está melhor
COR_BM_FILL   = "#f56565"   # vermelho claro — área quando o benchmark está melhor
COR_BG        = "#0f0f1a"
COR_CARD      = "#12121f"
COR_BORDER    = "#1e1e35"
COR_TEXT      = "#c8c8d8"
COR_TICKS     = "#9ea3bd"   # tick labels com mais contraste que o antigo
COR_GRID      = "#2a2a48"
FILL_ALPHA    = 0.20
LINE_W        = 2.0
# RGB para os elementos pygame do painel direito
RGB_AGENTE    = (91, 141, 245)
RGB_BENCHMARK = (245, 101, 101)
RGB_MUTED     = (100, 100, 150)
RGB_BANNER_BG = (26, 26, 44)
RGB_BANNER_HL = (44, 44, 72)


# ── Controlador PPO adaptado para SimulationLoop ──────────────────────────────

class DualAgentController:
    """Adapta o DualAgent para a interface decide(state) do SimulationLoop."""

    _NORM = {
        "veh_ns": 30.0, "ped_l": 15.0, "ped_o": 15.0,
        "ticks_in_phase": 36.0, "max_wait_car_s": 90.0, "max_wait_ped_s": 60.0,
    }
    _TICK_S = 5

    def __init__(self, model_geral: PPO, model_pico: PPO | None, limiar: float = 2.0) -> None:
        self._geral  = model_geral
        self._pico   = model_pico
        self._limiar = limiar
        self.usando_especialista = False

    def decide(self, state: dict) -> int:
        fila_car = state["veh_ns"]["size"]
        fila_ped = state["ped_l"]["size"] + state["ped_o"]["size"]
        usar_pico = (self._pico is not None
                     and fila_car > fila_ped * self._limiar
                     and fila_car > 3)
        self.usando_especialista = usar_pico

        n = self._NORM
        ts = self._TICK_S
        max_car_s = state["veh_ns"]["max_wait_ticks"] * ts
        max_ped_s = max(state["ped_l"]["max_wait_ticks"],
                        state["ped_o"]["max_wait_ticks"]) * ts

        obs = np.array([
            state["veh_ns"]["size"] / n["veh_ns"],
            state["ped_l"]["size"]  / n["ped_l"],
            state["ped_o"]["size"]  / n["ped_o"],
            float(state["phase"] == "B"),
            state["ticks_in_phase"] / n["ticks_in_phase"],
            max_car_s / n["max_wait_car_s"],
            max_ped_s / n["max_wait_ped_s"],
        ], dtype=np.float32).reshape(1, -1).clip(0, 1)

        model = self._pico if usar_pico else self._geral
        action, _ = model.predict(obs, deterministic=True)
        return int(action.item())


# ── Controlador de tempo fixo para comparação ─────────────────────────────────

class BenchmarkController:
    def __init__(self, phase_a_ticks: int, phase_b_ticks: int) -> None:
        self._ctrl = FixedTimeController(phase_a_ticks, phase_b_ticks)

    def decide(self, state: dict) -> int:
        return self._ctrl.decide(state)


# ── Coletor de métricas paralelo ──────────────────────────────────────────────

class MetricsCollector:
    """Coleta métricas tick a tick para dois controladores em paralelo."""

    def __init__(self, history: int) -> None:
        self.h = history
        # Agente
        self.ag_espera_media  = deque(maxlen=history)
        self.ag_violacoes     = deque(maxlen=history)
        self.ag_fila_total    = deque(maxlen=history)
        self.ag_espera_acum   = 0.0
        self.ag_viol_acum     = 0
        # Benchmark
        self.bm_espera_media  = deque(maxlen=history)
        self.bm_violacoes     = deque(maxlen=history)
        self.bm_fila_total    = deque(maxlen=history)
        self.bm_espera_acum   = 0.0
        self.bm_viol_acum     = 0
        self.ticks            = deque(maxlen=history)
        self.tick_count       = 0
        # Tetos em ticks
        self.teto_car  = 18   # 90s ÷ 5
        self.teto_ped  = 12   # 60s ÷ 5

    def update_agente(self, state: dict) -> None:
        veh = state["veh_ns"]
        pl  = state["ped_l"]
        po  = state["ped_o"]

        total_espera = (veh["total_wait_ticks"] + pl["total_wait_ticks"]
                        + po["total_wait_ticks"]) * 5
        fila = veh["size"] + pl["size"] + po["size"]

        violacoes = sum(1 for t in veh.get("snapshot", []) if t > self.teto_car)

        self.ag_espera_acum += total_espera
        self.ag_viol_acum   += violacoes
        self.ag_espera_media.append(
            self.ag_espera_acum / max(self.tick_count + 1, 1)
        )
        self.ag_violacoes.append(self.ag_viol_acum)
        self.ag_fila_total.append(fila)

    def update_benchmark(self, state: dict) -> None:
        veh = state["veh_ns"]
        pl  = state["ped_l"]
        po  = state["ped_o"]

        total_espera = (veh["total_wait_ticks"] + pl["total_wait_ticks"]
                        + po["total_wait_ticks"]) * 5
        fila = veh["size"] + pl["size"] + po["size"]

        violacoes = sum(1 for t in veh.get("snapshot", []) if t > self.teto_car)

        self.bm_espera_acum += total_espera
        self.bm_viol_acum   += violacoes
        self.bm_espera_media.append(
            self.bm_espera_acum / max(self.tick_count + 1, 1)
        )
        self.bm_violacoes.append(self.bm_viol_acum)
        self.bm_fila_total.append(fila)

        self.ticks.append(self.tick_count)
        self.tick_count += 1

    def reset(self) -> None:
        for q in [self.ag_espera_media, self.ag_violacoes, self.ag_fila_total,
                  self.bm_espera_media, self.bm_violacoes, self.bm_fila_total,
                  self.ticks]:
            q.clear()
        self.ag_espera_acum = 0.0
        self.ag_viol_acum   = 0
        self.bm_espera_acum = 0.0
        self.bm_viol_acum   = 0
        self.tick_count     = 0


# ── Painel de gráficos comparativos ──────────────────────────────────────────

class ComparisonPlots:
    """Renderiza 3 gráficos comparativos agente vs benchmark em tempo real."""

    def __init__(self, width: int, height: int, dpi: int = 100) -> None:
        fig_w = width  / dpi
        fig_h = height / dpi
        self._fig, self._axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi)
        self._fig.patch.set_facecolor(COR_BG)
        # top reserva BANNER_H px (+ folga) para o banner de controlador/cenário
        self._fig.subplots_adjust(
            left=0.14, right=0.97,
            top=1.0 - (BANNER_H + 16) / height,
            bottom=0.085, hspace=0.55,
        )
        self._canvas  = FigureCanvasAgg(self._fig)
        self._surface: pygame.Surface | None = None
        self._dirty   = False

    def update(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        self._dirty   = True

    def render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        m   = self._metrics
        ax1, ax2, ax3 = self._axes
        bg  = COR_CARD
        txt = COR_TEXT

        ticks = list(m.ticks)
        if len(ticks) < 2:
            return

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor(bg)
            ax.tick_params(colors=COR_TICKS, labelsize=8, length=3)
            # bordas quase invisíveis: só esquerda/baixo, finas e escuras
            for side, sp in ax.spines.items():
                sp.set_visible(side in ("left", "bottom"))
                sp.set_edgecolor(COR_BORDER)
                sp.set_linewidth(0.6)

        def plot_pair(ax, ag_data, bm_data, titulo, ylabel, fmt=".1f"):
            ag = list(ag_data)
            bm = list(bm_data)
            n  = min(len(ticks), len(ag), len(bm))
            if n < 2:
                return
            t = ticks[:n]
            ax.plot(t, bm[:n], color=COR_BENCHMARK, lw=LINE_W)
            ax.plot(t, ag[:n], color=COR_AGENTE,    lw=LINE_W)
            ax.fill_between(t, bm[:n], ag[:n],
                            where=[a < b for a, b in zip(ag[:n], bm[:n])],
                            alpha=FILL_ALPHA, color=COR_AG_FILL, lw=0)
            ax.fill_between(t, bm[:n], ag[:n],
                            where=[a >= b for a, b in zip(ag[:n], bm[:n])],
                            alpha=FILL_ALPHA, color=COR_BM_FILL, lw=0)
            ax.set_title(titulo, color="white", fontsize=11, pad=6,
                         fontweight="bold", loc="left")
            ax.set_ylabel(ylabel, color=txt, fontsize=7)
            ax.grid(alpha=0.5, color=COR_GRID, lw=0.5)
            ax.set_axisbelow(True)

            # legenda compacta numa linha: fundo translúcido + números em negrito
            ax.add_patch(plt.Rectangle(
                (0.015, 0.80), 0.70, 0.17, transform=ax.transAxes,
                facecolor=COR_CARD, edgecolor=COR_BORDER,
                alpha=0.78, zorder=4,
            ))
            ax.text(0.035, 0.885, f"Agente $\\mathbf{{{ag[-1]:{fmt}}}}$",
                    transform=ax.transAxes, color=COR_AGENTE,
                    fontsize=8, va="center", zorder=5)
            ax.text(0.345, 0.885, "•", transform=ax.transAxes,
                    color=COR_TICKS, fontsize=8, va="center", zorder=5)
            ax.text(0.385, 0.885, f"Benchmark $\\mathbf{{{bm[-1]:{fmt}}}}$",
                    transform=ax.transAxes, color=COR_BENCHMARK,
                    fontsize=8, va="center", zorder=5)

        plot_pair(ax1, m.ag_espera_media, m.bm_espera_media,
                  "Espera Média Acumulada", "segundos")
        plot_pair(ax2, m.ag_violacoes,    m.bm_violacoes,
                  "Violações de Teto (acumulado)", "contagem", fmt=".0f")
        plot_pair(ax3, m.ag_fila_total,   m.bm_fila_total,
                  "Tamanho Total das Filas", "agentes", fmt=".0f")

        ax3.set_xlabel("tick", color=COR_TICKS, fontsize=7)

        self._canvas.draw()
        raw  = self._canvas.buffer_rgba()
        size = self._canvas.get_width_height()
        self._surface = pygame.image.frombuffer(raw, size, "RGBA").copy()

    def get_surface(self) -> pygame.Surface | None:
        return self._surface

    def close(self) -> None:
        plt.close(self._fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Configurações
    cfg    = load_config("configs/config.yaml")
    sim    = cfg["simulation"]
    ft_cfg = cfg["fixed_time_controller"]
    tick_s = sim["tick_seconds"]

    # Carrega modelos
    model_geral = None
    model_pico  = None

    geral_path = Path("models/ppo_semaforo_final.zip")
    pico_path  = Path("models/ppo_pico_veic_final.zip")

    if not geral_path.exists():
        best = Path("models/best/best_model.zip")
        if best.exists():
            geral_path = best
        else:
            print("ERRO: modelo não encontrado. Execute train_agent.py primeiro.")
            sys.exit(1)

    print(f"Carregando agente geral: {geral_path}")
    model_geral = PPO.load(str(geral_path))

    if pico_path.exists():
        print(f"Carregando especialista pico_veic: {pico_path}")
        model_pico = PPO.load(str(pico_path))
    else:
        print("Especialista não encontrado — usando só agente geral.")

    # Cenários
    scenarios_dir  = Path("scenarios/eval")
    scenario_paths = sorted(scenarios_dir.glob("*.csv"))
    if not scenario_paths:
        scenarios_dir  = Path("scenarios/train")
        scenario_paths = sorted(scenarios_dir.glob("*.csv"))
    if not scenario_paths:
        print("ERRO: nenhum cenário encontrado.")
        sys.exit(1)

    def sortear() -> tuple[pd.DataFrame, str]:
        p = random.choice(scenario_paths)
        return pd.read_csv(p), p.name

    scenario_df, scenario_name = sortear()

    def arrivals_fn(tick: int) -> dict[str, int]:
        if tick >= len(scenario_df):
            return {"veh_ns": 0, "ped_l": 0, "ped_o": 0}
        row = scenario_df.iloc[tick]
        return {
            "veh_ns": int(row["veh_ns"]),
            "ped_l":  int(row["ped_l"]),
            "ped_o":  int(row["ped_o"]),
        }

    # Controladores
    agent_ctrl = DualAgentController(model_geral, model_pico)
    bench_ctrl = BenchmarkController(ft_cfg["phase_a_ticks"], ft_cfg["phase_b_ticks"])

    # Dois crossings — um para cada controlador
    crossing_agent = Crossing(cfg=sim)
    crossing_bench = Crossing(cfg=sim)

    # Loop principal (agente — para renderização)
    loop = SimulationLoop(
        crossing=crossing_agent,
        controller=agent_ctrl,
        arrivals_fn=arrivals_fn,
        tick_seconds=tick_s,
    )

    # Estado do benchmark (roda em paralelo, tick a tick)
    bench_state = crossing_bench.reset()

    # Métricas
    metrics   = MetricsCollector(history=HISTORY)
    plot_tick = 0

    def on_tick(state: dict) -> None:
        nonlocal bench_state, plot_tick
        # Avança benchmark com as mesmas chegadas
        tick_atual = state["tick"] - 1
        arr = arrivals_fn(tick_atual)
        bench_action = bench_ctrl.decide(bench_state)
        bench_state  = crossing_bench.step(arr, bench_action)

        # Coleta métricas
        metrics.update_agente(state)
        metrics.update_benchmark(bench_state)
        plot_tick += 1

    loop._on_tick = on_tick

    # Pygame
    pygame.init()
    screen = pygame.display.set_mode((RENDERER_W + PLOTS_W, WIN_H))
    pygame.display.set_caption(
        f"Semáforo Inteligente — DualAgent vs Benchmark | {scenario_name}"
    )
    fps_clock = pygame.time.Clock()

    renderer = CrossingRenderer(cfg=cfg, panel_width=RENDERER_W, panel_height=WIN_H)
    plots    = ComparisonPlots(width=PLOTS_W, height=WIN_H)

    speed_map   = {"1x": 1.0, "5x": 5.0, "10x": 10.0, "50x": 50.0}
    active_spd  = "1x"
    loop.set_speed(1.0)

    plot_surface = pygame.Surface((PLOTS_W, WIN_H))
    font         = load_ui_font(13)
    font_bold    = load_ui_font(13, bold=True)

    def do_reset() -> None:
        nonlocal bench_state
        loop.reset()
        bench_state = crossing_bench.reset()
        metrics.reset()
        plots.update(metrics)

    def do_new_scenario() -> None:
        nonlocal scenario_df, scenario_name, bench_state
        scenario_df, scenario_name = sortear()
        loop._arrivals_fn = arrivals_fn
        loop.reset()
        bench_state = crossing_bench.reset()
        metrics.reset()
        plots.update(metrics)
        pygame.display.set_caption(
            f"Semáforo Inteligente — DualAgent vs Benchmark | {scenario_name}"
        )

    print(f"\nDemo iniciado! Cenário: {scenario_name}")
    print("Q/ESC=sair  ESPAÇO=pausa  R=reiniciar  F=novo cenário  1-4=velocidade\n")

    running = True
    render_interval = 5   # re-renderiza gráficos a cada N ticks

    while running:
        dt = fps_clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    loop.toggle_pause()
                elif event.key == pygame.K_r:
                    do_reset()
                elif event.key == pygame.K_f:
                    do_new_scenario()
                elif event.key == pygame.K_1:
                    active_spd = "1x";  loop.set_speed(1.0)
                elif event.key == pygame.K_2:
                    active_spd = "5x";  loop.set_speed(5.0)
                elif event.key == pygame.K_3:
                    active_spd = "10x"; loop.set_speed(10.0)
                elif event.key == pygame.K_4:
                    active_spd = "50x"; loop.set_speed(50.0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if pos[0] < RENDERER_W:
                    for label, rect in renderer.get_button_rects().items():
                        if rect.collidepoint(pos):
                            if label == "PAUSE":
                                loop.toggle_pause()
                            elif label == "R":
                                do_reset()
                            elif label in speed_map:
                                active_spd = label
                                loop.set_speed(speed_map[label])

        loop.update(dt)

        # Atualiza gráficos a cada N ticks
        if plot_tick % render_interval == 0:
            plots.update(metrics)
        plots.render()

        # Renderiza cruzamento
        renderer_surface = pygame.Surface((RENDERER_W, WIN_H))
        renderer.draw(
            surface=renderer_surface,
            state=loop.state,
            is_yellow=loop.is_yellow,
            is_paused=loop.paused,
            speed=loop.speed,
            active_speed_label=active_spd,
            mouse_pos=pygame.mouse.get_pos(),
            cfg_fixed=cfg["fixed_time_controller"],
            cfg_thresholds=cfg["thresholds"],
            tick_seconds=tick_s,
            arrivals=loop.last_arrivals,
            drains=loop.last_drains,
        )
        screen.blit(renderer_surface, (0, 0))

        # Renderiza gráficos
        plot_surf = plots.get_surface()
        if plot_surf:
            screen.blit(plot_surf, (RENDERER_W, 0))
        else:
            plot_surface.fill((15, 15, 26))
            lbl = font.render("Aguardando dados...", True, (100, 100, 150))
            plot_surface.blit(lbl, (PLOTS_W//2 - lbl.get_width()//2, WIN_H//2))
            screen.blit(plot_surface, (RENDERER_W, 0))

        # Banner superior do painel direito (faixa reservada — fora dos gráficos)
        pygame.draw.rect(screen, RGB_BANNER_BG,
                         pygame.Rect(RENDERER_W, 0, PLOTS_W, BANNER_H))
        pygame.draw.rect(screen, RGB_BANNER_HL,
                         pygame.Rect(RENDERER_W, BANNER_H - 1, PLOTS_W, 1))
        agente_tipo = "Especialista pico_veic" if agent_ctrl.usando_especialista else "Agente Geral"
        cor_tipo    = (159, 122, 234) if agent_ctrl.usando_especialista else RGB_AGENTE
        lbl_agente  = font_bold.render(f"Controlador: {agente_tipo}", True, cor_tipo)
        lbl_cen     = font.render(f"Cenário: {scenario_name}", True, RGB_MUTED)
        screen.blit(lbl_agente, (RENDERER_W + 12, 5))
        screen.blit(lbl_cen,   (RENDERER_W + 12, 21))

        # Legenda inferior centralizada, com bolinhas coloridas
        lbl_ag = font.render("Agente PPO", True, RGB_AGENTE)
        lbl_bm = font.render("Benchmark (tempo fixo)", True, RGB_BENCHMARK)
        dot_r, dot_gap, item_gap = 4, 8, 28
        total_w = (dot_r * 2 + dot_gap) * 2 + lbl_ag.get_width() + lbl_bm.get_width() + item_gap
        lx = RENDERER_W + (PLOTS_W - total_w) // 2
        ly = WIN_H - LEGEND_Y
        cy = ly + lbl_ag.get_height() // 2
        pygame.draw.circle(screen, RGB_AGENTE, (lx + dot_r, cy), dot_r)
        screen.blit(lbl_ag, (lx + dot_r * 2 + dot_gap, ly))
        lx += dot_r * 2 + dot_gap + lbl_ag.get_width() + item_gap
        pygame.draw.circle(screen, RGB_BENCHMARK, (lx + dot_r, cy), dot_r)
        screen.blit(lbl_bm, (lx + dot_r * 2 + dot_gap, ly))

        pygame.display.flip()

    pygame.quit()
    plots.close()
    print("Demo encerrado.")


if __name__ == "__main__":
    main()

"""
Demo visual — Fase 2.

Roda o controlador de tempo fixo com visualização Pygame do cruzamento e
painel matplotlib ao vivo. Executa indefinidamente até o usuário fechar a janela.

Uso:
    python scripts/run_visual_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # deve vir antes de qualquer import pyplot ou inicialização pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pygame

from src.simulation.crossing import Crossing
from src.simulation.controllers import FixedTimeController
from src.simulation.simulation_loop import SimulationLoop
from src.visualization.pygame_renderer import CrossingRenderer
from src.visualization.live_plots import LivePlots
from src.utils.config_loader import load_all_configs


def main() -> None:
    cfgs     = load_all_configs(ROOT / "configs")
    cfg      = cfgs["config"]
    sim_cfg  = cfg["simulation"]
    ft_cfg   = cfg["fixed_time_controller"]
    demo_cfg = cfg["demo_arrival_rate"]
    viz_cfg  = cfg["visualization"]
    thr_cfg  = cfg["thresholds"]
    tick_s   = sim_cfg["tick_seconds"]

    WIN_W   = viz_cfg["window_width"]
    WIN_H   = viz_cfg["window_height"]
    PANEL_L = int(WIN_W * 0.6)   # 768
    PANEL_R = WIN_W - PANEL_L    # 512

    # ── Painel de gráficos (antes do pygame.init para garantir Agg) ──────────
    plots = LivePlots(
        panel_width    = PANEL_R,
        panel_height   = WIN_H,
        history_length = viz_cfg["history_length_ticks"],
        tick_seconds   = tick_s,
        ceil_cars_s    = thr_cfg["wait_ceiling_cars_seconds"],
        ceil_peds_s    = thr_cfg["wait_ceiling_pedestrians_seconds"],
    )

    # ── Função de chegadas (placeholder Poisson até Fase 3) ──────────────────
    rng    = np.random.default_rng(42)
    rate_v = demo_cfg["vehicles"]
    rate_e = demo_cfg["pedestrians_east"]
    rate_w = demo_cfg["pedestrians_west"]

    def arrivals_fn(tick: int) -> dict[str, int]:
        return {
            "veh_ns": int(rng.poisson(rate_v)),
            "ped_l":  int(rng.poisson(rate_e)),
            "ped_o":  int(rng.poisson(rate_w)),
        }

    # ── Engine de simulação ──────────────────────────────────────────────────
    crossing   = Crossing(sim_cfg)
    controller = FixedTimeController(ft_cfg["phase_a_ticks"], ft_cfg["phase_b_ticks"])
    loop       = SimulationLoop(
        crossing     = crossing,
        controller   = controller,
        arrivals_fn  = arrivals_fn,
        tick_seconds = tick_s,
        on_tick      = plots.record,
    )
    loop.set_speed(5.0)

    # ── Pygame ───────────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Semáforo Inteligente — Demo Visual (Fase 2)")
    clock = pygame.time.Clock()

    renderer   = CrossingRenderer(cfg, PANEL_L, WIN_H)
    cx_surface = pygame.Surface((PANEL_L, WIN_H))

    # ── Controle de velocidade ───────────────────────────────────────────────
    _SPEEDS = {"1x": 1.0, "5x": 5.0, "10x": 10.0, "50x": 50.0}
    active_speed_label = "5x"

    def set_speed(label: str) -> None:
        nonlocal active_speed_label
        active_speed_label = label
        loop.set_speed(_SPEEDS[label])

    def do_reset() -> None:
        loop.reset()
        plots.clear_history()

    # ── Loop principal ───────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    loop.toggle_pause()
                elif event.key == pygame.K_1:
                    set_speed("1x")
                elif event.key == pygame.K_2:
                    set_speed("5x")
                elif event.key == pygame.K_3:
                    set_speed("10x")
                elif event.key == pygame.K_4:
                    set_speed("50x")
                elif event.key in (pygame.K_r, pygame.K_R):
                    do_reset()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for lbl, rect in renderer.get_button_rects().items():
                    if rect.collidepoint(event.pos):
                        if lbl == "PAUSE":
                            loop.toggle_pause()
                        elif lbl == "R":
                            do_reset()
                        elif lbl in _SPEEDS:
                            set_speed(lbl)
                        break

        loop.update(dt)
        plots.update_surface(viz_cfg["plot_update_interval_ticks"])

        # painel esquerdo — cruzamento
        renderer.draw(
            surface            = cx_surface,
            state              = loop.state,
            is_yellow          = loop.is_yellow,
            is_paused          = loop.paused,
            speed              = loop.speed,
            active_speed_label = active_speed_label,
            mouse_pos          = mouse_pos,
            cfg_fixed          = ft_cfg,
            cfg_thresholds     = thr_cfg,
            tick_seconds       = tick_s,
        )
        screen.blit(cx_surface, (0, 0))

        # painel direito — gráficos
        plot_surf = plots.get_surface()
        if plot_surf is not None:
            screen.blit(plot_surf, (PANEL_L, 0))
        else:
            pygame.draw.rect(screen, (18, 18, 28),
                             pygame.Rect(PANEL_L, 0, PANEL_R, WIN_H))

        # separador vertical
        pygame.draw.line(screen, (50, 50, 72), (PANEL_L, 0), (PANEL_L, WIN_H))

        pygame.display.flip()

    # ── Encerramento limpo ───────────────────────────────────────────────────
    plots.close()
    pygame.quit()


if __name__ == "__main__":
    main()
